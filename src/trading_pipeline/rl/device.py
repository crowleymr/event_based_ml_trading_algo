"""CUDA preflight and explicit safe device resolution for the optional RL stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time


VALID_DEVICES = {"cpu", "cuda", "auto"}


def resolve_device(requested: str) -> dict:
    if requested not in VALID_DEVICES:
        raise ValueError(f"device must be one of {sorted(VALID_DEVICES)}")
    import torch

    available = bool(torch.cuda.is_available())
    actual = "cuda" if requested in {"cuda", "auto"} and available else "cpu"
    reason = None
    if requested == "cuda" and not available:
        reason = "requested CUDA but the installed PyTorch build/runtime reports CUDA unavailable"
    elif requested == "auto" and not available:
        reason = "auto selected CPU because the installed PyTorch build/runtime reports CUDA unavailable"
    return {
        "requested_device": requested,
        "actual_device": actual,
        "fallback_reason": reason,
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": available,
        "cuda_device_name": torch.cuda.get_device_name(0) if available else None,
        "cuda_device_count": torch.cuda.device_count() if available else 0,
        "cudnn_version": torch.backends.cudnn.version() if available else None,
    }


def preflight(requested: str = "auto") -> dict:
    import torch

    result = resolve_device(requested)
    device = torch.device(result["actual_device"])
    started = time.perf_counter()
    left = torch.arange(4096, dtype=torch.float32, device=device).reshape(64, 64)
    product = left @ left.T
    if device.type == "cuda":
        torch.cuda.synchronize()
    result.update({
        "tensor_device": str(product.device),
        "tensor_checksum": float(product.sum().cpu()),
        "operation_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "passed": bool(product.shape == (64, 64) and str(product.device).startswith(result["actual_device"])),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the optional PyTorch CUDA execution path")
    parser.add_argument("--device", choices=sorted(VALID_DEVICES), default="auto")
    parser.add_argument("--output", help="Optional new JSON evidence path")
    args = parser.parse_args()
    result = preflight(args.device)
    if args.output:
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
