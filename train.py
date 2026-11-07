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
import shutil
from pathlib import Path

try:
    from .gpu import configure_cuda_environment, require_two_gpus
except ImportError:
    from gpu import configure_cuda_environment, require_two_gpus


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
    model.train(**{k: v for k, v in kw.items() if v is not None})

    best = Path(args.project) / name / "weights" / "best.pt"
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
