"""Best-effort hardware inventory; CUDA is never a dependency."""
import argparse
import ctypes
import ctypes.util
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_environment():
    cpu = {"model": platform.processor() or "unknown", "logical_cores": os.cpu_count(), "physical_cores": "unknown"}
    if platform.system() == "Windows":
        raw = command(["powershell", "-NoProfile", "-Command",
                       "Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json -Compress"])
        try:
            items = json.loads(raw) if raw else None
            if items:
                items = items if isinstance(items, list) else [items]
                cpu = {"model": "; ".join(i["Name"].strip() for i in items),
                       "physical_cores": sum(i["NumberOfCores"] for i in items),
                       "logical_cores": sum(i["NumberOfLogicalProcessors"] for i in items)}
        except (ValueError, KeyError, TypeError):
            pass
    elif Path("/proc/cpuinfo").exists():
        try:
            info = Path("/proc/cpuinfo").read_text()
            match = re.search(r"model name\s*:\s*(.+)", info)
            if match:
                cpu["model"] = match.group(1)
            pairs = set()
            for block in info.split("\n\n"):
                physical = re.search(r"physical id\s*:\s*(\d+)", block)
                core = re.search(r"core id\s*:\s*(\d+)", block)
                if physical and core:
                    pairs.add((physical.group(1), core.group(1)))
            if pairs:
                cpu["physical_cores"] = len(pairs)
        except OSError:
            pass
    gpus = []
    raw = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"])
    if raw:
        for line in raw.splitlines():
            items = [s.strip() for s in line.split(",")]
            if len(items) == 3:
                gpus.append({"model": items[0], "vram_mib": int(items[1]) if items[1].isdigit() else "unknown", "driver_version": items[2]})
    header = command(["nvidia-smi"]) if raw else None
    match = re.search(r"CUDA Version:\s*([\d.]+)", header or "")
    nvcc = shutil.which("nvcc")
    if not nvcc and os.environ.get("CUDA_PATH"):
        candidate = Path(os.environ["CUDA_PATH"]) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
        if candidate.exists():
            nvcc = str(candidate)
    toolkit = command([nvcc, "--version"]) if nvcc else None
    toolkit_match = re.search(r"release\s+([\d.]+)", toolkit or "")
    runtime, runtime_library = "unavailable or not discoverable", None
    candidates = []
    library = ctypes.util.find_library("cudart")
    if library:
        candidates.append(library)
    if os.environ.get("CUDA_PATH"):
        candidates.extend(str(p) for p in (Path(os.environ["CUDA_PATH"]) / "bin").glob("cudart64*.dll"))
    for library in candidates:
        try:
            dll = ctypes.CDLL(library)
            version = ctypes.c_int()
            if dll.cudaRuntimeGetVersion(ctypes.byref(version)) == 0:
                runtime = f"{version.value // 1000}.{(version.value % 1000) // 10}"
                runtime_library = library
                break
        except (OSError, AttributeError):
            pass
    return {"detected_at": datetime.now(timezone.utc).isoformat(), "os": platform.platform(),
            "python": platform.python_version(), "cpu": cpu, "nvidia_gpus": gpus,
            "gpu_detection_status": "detected" if gpus else "unavailable or not discoverable; CPU execution unaffected",
            "driver_reported_cuda_compatibility": match.group(1) if match else "unknown",
            "installed_cuda_toolkit_version": toolkit_match.group(1) if toolkit_match else "unavailable or not discoverable",
            "installed_cuda_runtime_version": runtime, "runtime_library": runtime_library,
            "cuda_note": "Driver compatibility is not evidence of an installed CUDA toolkit/runtime."}


def model_devices(seed):
    return {"actual_device_per_model": {e: "cpu" for e in ("E1", "E2", "E3", "E4")},
            "e5_device": "reuses selected model predictions; portfolio on CPU",
            "gpu_fallback_reason": "not applicable: CPU-first sklearn models; GPU path was not attempted",
            "reproducibility": {"seed": seed, "numerical_threads": 1, "gbt_early_stopping": False,
                                "fixed_parameter_grid": True, "remaining_nondeterminism":
                                "Floating point differences across hardware/library versions; source downloads may change across vintages."}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    root = Path(parser.parse_args().run)
    path = root / "metadata.json"
    metadata = json.loads(path.read_text())
    metadata.update(model_devices(metadata["seed"]))
    metadata["environment"] = detect_environment()
    metadata["environment_capture_note"] = "Added after completed run on same host; hardware policy was added to authoritative docs during implementation. Models were CPU sklearn throughout."
    path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata["environment"], indent=2))
