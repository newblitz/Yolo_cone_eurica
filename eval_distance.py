#!/usr/bin/env python3
"""Validate the distance estimators against tape-measured ground truth.

CSV columns: image,class,x1,y1,x2,y2,true_distance_m
  (box = tight box around the object in that image; take ~15-30 photos at known distances,
   e.g. 2, 4, 6, 8, 10 m, with the camera at a fixed, measured height.)

Example:
  python -m src.eval_distance --csv data/dist_gt.csv --images data/dist_photos --hfov 65 --cam-height 1.4
  # or, after calibrating the focal length from ONE photo:
  python -m src.eval_distance --csv ... --images ... --focal-px 980 --ref-width 1280
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image

try:
    from .distance import build_estimator
except ImportError:
    from distance import build_estimator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--focal-px", type=float, default=None)
    ap.add_argument("--ref-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--pitch", type=float, default=0.0)
    args = ap.parse_args()

    errs = {"height": [], "ground": [], "fused": []}
    rows = list(csv.DictReader(open(args.csv)))
    print(f"{'image':24s} {'cls':9s} {'true':>6s} {'M1':>6s} {'M2':>6s} {'fused':>6s}  conf")
    for r in rows:
        with Image.open(Path(args.images) / r["image"]) as im:
            w, h = im.size
        estimator = build_estimator(
            w, h, hfov=args.hfov, focal_px=args.focal_px, ref_width=args.ref_width,
            cam_height=args.cam_height, pitch=args.pitch,
        )
        e = estimator.estimate(r["class"], [float(r[k]) for k in ("x1", "y1", "x2", "y2")])
        t = float(r["true_distance_m"])
        for key, val in (("height", e.z_height), ("ground", e.z_ground), ("fused", e.distance_m)):
            if val is not None:
                errs[key].append((abs(val - t), abs(val - t) / t))
        f = lambda v: "  n/a " if v is None else f"{v:6.2f}"
        print(f"{r['image'][:24]:24s} {r['class']:9s} {t:6.2f} {f(e.z_height)} {f(e.z_ground)} {f(e.distance_m)}  {e.confidence}")

    print("\n| Method | N | MAE (m) | MAPE (%) |\n|---|---|---|---|")
    for key, v in errs.items():
        if v:
            mae = sum(a for a, _ in v) / len(v)
            mape = 100 * sum(p for _, p in v) / len(v)
            print(f"| {key} | {len(v)} | {mae:.2f} | {mape:.1f} |")


if __name__ == "__main__":
    main()
