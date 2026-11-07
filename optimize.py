#!/usr/bin/env python3
"""Edge-device optimisation + benchmark: 'before vs after' table of FPS (CPU/GPU) and mAP.

Variants (each one is skipped gracefully if it fails or the hardware is missing):
  PyTorch FP32  (CPU, GPU)      baseline
  PyTorch FP16  (GPU)           half precision
  ONNX FP32     (CPU)           graph export, onnxruntime
  OpenVINO INT8 (CPU)           post-training static quantisation (needs `openvino`, `nncf`)
  Pruned FP32   (CPU, GPU)      global unstructured L1 pruning - for the accuracy-vs-sparsity story only;
                                unstructured sparsity gives NO speed-up on dense CPU/GPU kernels.

FPS = end-to-end (pre-process + inference + NMS), batch 1, images pre-loaded in RAM.

Example:
  python -m src.optimize --weights weights/yolo11n_best.pt --data data/merged/data.yaml \
      --images data/merged/images/val --prune 0.3 --out results/benchmark.md
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure_fps(model_path, device, frames, imgsz, half=False, warmup=10):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    kw = dict(imgsz=imgsz, device=device, verbose=False)
    if half:
        kw["half"] = True
    for f in frames[:warmup]:
        m.predict(f, **kw)
    sync()
    t0 = time.perf_counter()
    for f in frames:
        m.predict(f, **kw)
    sync()
    return len(frames) / (time.perf_counter() - t0)


def measure_map(model_path, data, device, imgsz, half=False):
    from ultralytics import YOLO
    m = YOLO(str(model_path), task="detect")
    kw = dict(data=data, imgsz=imgsz, device=device, batch=1 if str(model_path).endswith(".onnx") else 16,
              verbose=False, plots=False)
    if half:
        kw["half"] = True
    r = m.val(**kw)
    return float(r.box.map50), float(r.box.map)


def prune_model(weights: str, amount: float, out_path: str) -> str:
    """Global unstructured L1 pruning of all Conv2d weights except the detection head."""
    import torch
    import torch.nn as nn
    import torch.nn.utils.prune as prune
    from ultralytics import YOLO
    m = YOLO(weights)
    net = m.model.float()
    layers = list(net.model.children())[:-1]  # skip Detect head
    convs = [(mod, "weight") for layer in layers for mod in layer.modules() if isinstance(mod, nn.Conv2d)]
    prune.global_unstructured(convs, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in convs:
        prune.remove(mod, name)
    zeros = sum(int((mod.weight == 0).sum()) for mod, _ in convs)
    total = sum(mod.weight.numel() for mod, _ in convs)
    print(f"pruned {zeros / total:.1%} of conv weights")
    m.save(out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, nargs="+", help="one or more .pt files (e.g. yolo11n and yolo11s)")
    ap.add_argument("--data", required=True)
    ap.add_argument("--images", required=True, help="folder with images for the FPS test")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=100, help="number of images for the FPS test")
    ap.add_argument("--prune", type=float, default=0.0, help="pruning amount, e.g. 0.3 (0 = skip)")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--out", default="results/benchmark.md")
    args = ap.parse_args()

    import cv2
    import torch
    has_gpu = torch.cuda.is_available()
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[: args.n]
    frames = [cv2.imread(str(p)) for p in paths]
    print(f"{len(frames)} frames | GPU: {torch.cuda.get_device_name(0) if has_gpu else 'none'}")

    rows = []  # (model, variant, device, fps, map50, map5095, size_mb)

    def run(model_tag, variant, path, device, half=False):
        try:
            fps = measure_fps(path, device, frames, args.imgsz, half)
            m50, m95 = (float("nan"),) * 2 if args.no_map else measure_map(path, args.data, device, args.imgsz, half)
            p = Path(path)
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size) / 1e6
            rows.append((model_tag, variant, device, fps, m50, m95, size))
            print(f"  {model_tag:12s} {variant:14s} {device:5s} {fps:7.1f} FPS  mAP50-95={m95:.3f}")
        except Exception as e:  # keep going - report what we could measure
            print(f"  [skip] {model_tag} {variant} {device}: {type(e).__name__}: {str(e)[:120]}")

    from ultralytics import YOLO
    for w in args.weights:
        tag = Path(w).stem.replace("_best", "")
        run(tag, "PyTorch FP32", w, "cpu")
        if has_gpu:
            run(tag, "PyTorch FP32", w, "0")
            run(tag, "PyTorch FP16", w, "0", half=True)
        try:
            onnx = YOLO(w).export(format="onnx", imgsz=args.imgsz, simplify=True)
            run(tag, "ONNX FP32", onnx, "cpu")
        except Exception as e:
            print(f"  [skip] ONNX export: {e}")
        try:
            ov = YOLO(w).export(format="openvino", imgsz=args.imgsz, int8=True, data=args.data)
            run(tag, "OpenVINO INT8", ov, "cpu")
        except Exception as e:
            print(f"  [skip] OpenVINO INT8: {type(e).__name__}: {str(e)[:120]}")
        if args.prune > 0:
            try:
                pruned = prune_model(w, args.prune, str(Path(w).with_name(f"{Path(w).stem}_pruned{int(args.prune * 100)}.pt")))
                run(tag, f"Pruned {int(args.prune * 100)}%", pruned, "cpu")
                if has_gpu:
                    run(tag, f"Pruned {int(args.prune * 100)}%", pruned, "0")
            except Exception as e:
                print(f"  [skip] pruning: {type(e).__name__}: {e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Model | Variant | Device | FPS | mAP50 | mAP50-95 | Size (MB) |", "|---|---|---|---|---|---|---|"]
    for m, v, d, fps, m50, m95, sz in rows:
        dev = "GPU" if d != "cpu" else "CPU"
        lines.append(f"| {m} | {v} | {dev} | {fps:.1f} | {m50:.3f} | {m95:.3f} | {sz:.1f} |")
    out.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines) + f"\n\nsaved to {out}")


if __name__ == "__main__":
    main()
