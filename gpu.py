"""CUDA runtime setup and strict GPU validation for the Kaggle runtime."""
from __future__ import annotations

import os
import ctypes
from pathlib import Path


_CUDA_LIBRARY_DIRS = (
    Path("/usr/local/nvidia/lib64"),
    Path("/usr/local/cuda/lib64"),
    Path("/usr/local/cuda-13.0/lib64"),
)


def configure_cuda_environment() -> None:
    """Make Kaggle's CUDA libraries discoverable before importing torch."""
    existing = os.environ.get("LD_LIBRARY_PATH", "").split(":")
    paths = [str(path) for path in _CUDA_LIBRARY_DIRS if path.is_dir()]
    os.environ["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys(paths + [p for p in existing if p]))
    driver = Path("/usr/local/nvidia/lib64/libcuda.so.580.178.04")
    if driver.exists():
        ctypes.CDLL(str(driver), mode=ctypes.RTLD_GLOBAL)


def require_two_gpus(device: str = "0,1") -> str:
    """Return a validated two-GPU device string; never silently use the CPU."""
    configure_cuda_environment()
    import torch

    requested = [part.strip() for part in device.split(",") if part.strip()]
    if requested != ["0", "1"]:
        raise ValueError(f"two GPUs are required; use --device 0,1 (got {device!r})")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The Kaggle CUDA library path was configured, "
            "but torch still cannot access a GPU; refusing CPU fallback."
        )
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"two GPUs are required, but only {torch.cuda.device_count()} are visible")
    print("CUDA devices:", ", ".join(torch.cuda.get_device_name(i) for i in range(2)))
    return "0,1"
