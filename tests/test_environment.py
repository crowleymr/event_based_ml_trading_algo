from trading_pipeline.environment import detect_environment, model_devices


def test_cpu_only_detection_never_requires_cuda(monkeypatch):
    monkeypatch.setattr("trading_pipeline.environment.command", lambda *args: None)
    monkeypatch.setattr("trading_pipeline.environment.shutil.which", lambda *args: None)
    monkeypatch.setattr("trading_pipeline.environment.ctypes.util.find_library", lambda *args: None)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    environment = detect_environment()
    assert environment["nvidia_gpus"] == []
    assert environment["driver_reported_cuda_compatibility"] == "unknown"
    assert environment["installed_cuda_toolkit_version"] == "unavailable or not discoverable"
    assert set(model_devices(42)["actual_device_per_model"].values()) == {"cpu"}
