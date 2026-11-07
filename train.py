#!/usr/bin/env python3
"""Fine-tune a pretrained YOLO11 model on the merged dataset (transfer learning).

Examples:
  python -m src.train --data data/merged/data.yaml --model yolo11n.pt --hours 2.0
  python -m src.train --data data/merged/data.yaml --model yolo11s.pt --epochs 50 --freeze 10

`--hours` uses Ultralytics' built-in time budget (it overrides --epochs and adapts the LR
schedule to finish on time) - the safest option when you only have one day.
"""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import time
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
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback.")
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"two GPUs are required, but only {torch.cuda.device_count()} are visible")
    print("CUDA devices:", ", ".join(torch.cuda.get_device_name(i) for i in range(2)))
    return "0,1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/merged/data.yaml")
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained COCO weights (yolo11n/s/m.pt)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--hours", type=float, default=None, help="time budget in hours (overrides epochs)")
    ap.add_argument("--batch", type=int, default=16, help="-1 = auto")
    ap.add_argument("--device", default="0,1", help="exactly two CUDA devices: 0,1")
    ap.add_argument("--freeze", type=int, default=0, help="freeze first N layers (10 = backbone) for speed")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache", action="store_true", help="cache images in RAM")
    ap.add_argument("--project", default="runs/train")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()

    configure_cuda_environment()
    device = require_two_gpus(args.device)
    from ultralytics import YOLO  # imported late so --help works without it

    name = args.name or Path(args.model).stem
    started = time.perf_counter()

    def report_progress(trainer):
        if os.environ.get("RANK", "-1") not in {"-1", "0"}:
            return
        epoch = int(trainer.epoch) + 1
        total = int(trainer.epochs)
        elapsed = time.perf_counter() - started
        rate = elapsed / epoch
        eta = max(0.0, rate * (total - epoch))
        width = 30
        filled = min(width, int(width * epoch / max(total, 1)))
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\nTraining progress: [{bar}] {epoch}/{total} "
            f"({epoch / max(total, 1):6.1%}) | "
            f"elapsed {time.strftime('%H:%M:%S', time.gmtime(elapsed))} | "
            f"ETA {time.strftime('%H:%M:%S', time.gmtime(eta))}",
            flush=True,
        )

    kw = dict(
        data=args.data, imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        workers=args.workers, freeze=args.freeze or None, cache=args.cache,
        project=args.project, name=name, exist_ok=True,
        patience=15, close_mosaic=5, seed=0, plots=True, amp=True,
    )
    if args.hours:
        kw["time"] = args.hours
    kw["device"] = device

    model = YOLO(args.model)
    model.add_callback("on_train_epoch_end", report_progress)
    model.train(**{k: v for k, v in kw.items() if v is not None})

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce the expected checkpoint: {best}")
    metrics = YOLO(best).val(data=args.data, imgsz=args.imgsz, device=device)
    print(f"\nmAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")
    for i, n in metrics.names.items():
        if i < len(metrics.box.maps):
            print(f"  {n:10s} mAP50-95 = {metrics.box.maps[i]:.3f}")

    Path("weights").mkdir(exist_ok=True)
    shutil.copy2(best, Path("weights") / f"{name}_best.pt")
    print(f"copied best weights to weights/{name}_best.pt")


if __name__ == "__main__":
    main()
