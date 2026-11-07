#!/usr/bin/env python3
"""Detect cones / barriers / stop signs and annotate each box as 'Cone, 1.5m'.

Examples:
  python -m src.infer --weights weights/yolo11n_best.pt --source samples/ --out runs/infer
  python -m src.infer --weights weights/yolo11n_best.pt --source drive.mp4 --track --out runs/infer

Camera assumptions (override with your own calibration!):
  --hfov 65           horizontal field of view in degrees      (or --focal-px + --ref-width)
  --cam-height 1.4    lens height above the road in metres
  --pitch 0           camera tilt in degrees, positive = up
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from .distance import DistanceEstimator, build_estimator as shared_build_estimator, format_label
except ImportError:
    from distance import DistanceEstimator, build_estimator as shared_build_estimator, format_label

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DISPLAY = {"cone": "Cone", "barrier": "Barrier", "stop_sign": "Stop Sign"}
COLORS = {"cone": (0, 140, 255), "barrier": (0, 200, 0), "stop_sign": (0, 0, 230)}  # BGR
CONF_MARK = {"high": "", "medium": "", "low": "?", "none": ""}


def build_estimator(args, w: int, h: int) -> DistanceEstimator:
    return shared_build_estimator(
        w, h, hfov=args.hfov, focal_px=args.focal_px, ref_width=args.ref_width,
        cam_height=args.cam_height, pitch=args.pitch,
    )


def draw(img, box, text, color):
    import cv2
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    scale = max(0.5, min(img.shape[:2]) / 900)
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    top = max(y1 - th - bl - 4, 0)
    cv2.rectangle(img, (x1, top), (x1 + tw + 6, top + th + bl + 4), color, -1)
    cv2.putText(img, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--source", required=True, help="image, folder of images, or video")
    ap.add_argument("--out", default="runs/infer")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--device", default=None)
    ap.add_argument("--hfov", type=float, default=65.0)
    ap.add_argument("--focal-px", type=float, default=None)
    ap.add_argument("--ref-width", type=int, default=1280)
    ap.add_argument("--cam-height", type=float, default=1.4)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--track", action="store_true", help="ByteTrack IDs + temporal smoothing of distance (video)")
    ap.add_argument("--smooth", type=float, default=0.4, help="EMA factor when --track is on")
    args = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(args.source)
    is_video = src.suffix.lower() in VIDEO_EXTS

    model = YOLO(args.weights)
    kw = dict(source=str(src), stream=True, conf=args.conf, imgsz=args.imgsz, verbose=False)
    if args.device is not None:
        kw["device"] = args.device
    results = model.track(persist=True, tracker="bytetrack.yaml", **kw) if args.track else model.predict(**kw)

    estimators, ema = {}, {}
    writer, log = None, open(out / "detections.jsonl", "w")
    for idx, r in enumerate(results):
        img = r.orig_img.copy()
        h, w = img.shape[:2]
        est = estimators.setdefault((w, h), build_estimator(args, w, h))
        frame_name = Path(r.path).name if not is_video else f"frame{idx:06d}"
        for b in r.boxes:
            cls_name = r.names[int(b.cls)]
            box = b.xyxy[0].tolist()
            d = est.estimate(cls_name, box)
            tid = int(b.id) if getattr(b, "id", None) is not None else None
            if tid is not None and d.distance_m:
                prev = ema.get(tid)
                logd = math.log(d.distance_m)
                ema[tid] = logd if prev is None else args.smooth * logd + (1 - args.smooth) * prev
                d.distance_m = math.exp(ema[tid])
            label = format_label(DISPLAY.get(cls_name, cls_name), d) + CONF_MARK[d.confidence]
            draw(img, box, label, COLORS.get(cls_name, (255, 0, 0)))
            log.write(json.dumps({
                "frame": frame_name, "class": cls_name, "conf": round(float(b.conf), 3),
                "box": [round(v, 1) for v in box], "track_id": tid,
                "distance_m": None if d.distance_m is None else round(d.distance_m, 2),
                "z_height": None if d.z_height is None else round(d.z_height, 2),
                "z_ground": None if d.z_ground is None else round(d.z_ground, 2),
                "method": d.method, "confidence": d.confidence, "truncated": d.truncated,
            }) + "\n")
        if is_video:
            if writer is None:
                fps = cv2.VideoCapture(str(src)).get(cv2.CAP_PROP_FPS) or 30
                writer = cv2.VideoWriter(str(out / f"{src.stem}_annotated.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(img)
        else:
            cv2.imwrite(str(out / f"{Path(r.path).stem}_annotated.jpg"), img)
    if writer:
        writer.release()
    log.close()
    print(f"results written to {out}/ (annotated media + detections.jsonl)")


if __name__ == "__main__":
    main()
