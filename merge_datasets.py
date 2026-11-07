#!/usr/bin/env python3
"""Merge heterogeneous detection datasets into ONE YOLO dataset with unified classes.

Unified classes: 0=cone, 1=barrier, 2=stop_sign

Supported source types (set `type:` in the config):
  yolo : a YOLO-format export (e.g. Roboflow Universe). Finds every <...>/images + sibling <...>/labels.
         Polygon (segmentation) labels are converted to boxes automatically.
  coco : a COCO-format json (e.g. COCO 2017 for "stop sign"). Missing images can be downloaded
         from the `coco_url` stored in the json (only the images we keep, not the whole 18 GB).
  bdd  : a BDD100K-style json (list of frames with `labels[].box2d`), optionally filtered on an
         attribute such as `sign_subtype == stop_sign` (BDD100K-MUTCD extension).

Usage:
  python -m src.merge_datasets --config configs/sources.yaml --out data/merged
  python -m src.merge_datasets --inspect-bdd path/to/labels.json sign_subtype   # list attribute values
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from PIL import Image

UNIFIED_CLASSES = ["cone", "barrier", "stop_sign"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_PX = 4        # drop boxes smaller than this (px, original resolution)
MIN_BOX_NORM = 0.002  # same idea for sources where we only know normalised coords


@dataclass
class Sample:
    img_path: Path
    boxes: list            # [(cls_id, xc, yc, w, h), ...] all normalised to [0, 1]
    source: str
    url: Optional[str] = None   # used to download the image if it is missing locally


# --------------------------------------------------------------------------- helpers
def norm(name) -> str:
    return str(name).strip().lower().replace("_", " ").replace("-", " ")


def build_class_map(class_map: dict) -> dict:
    """{'Traffic-Cone': 'cone'} -> {'traffic cone': 0}"""
    out = {}
    for src_name, uni in class_map.items():
        if uni not in UNIFIED_CLASSES:
            raise ValueError(f"'{uni}' is not a unified class. Choose from {UNIFIED_CLASSES}")
        out[norm(src_name)] = UNIFIED_CLASSES.index(uni)
    return out


def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    """Pixel xyxy -> normalised (xc, yc, w, h). Returns None for degenerate boxes."""
    x1, x2 = sorted((min(max(x1, 0.0), W), min(max(x2, 0.0), W)))
    y1, y2 = sorted((min(max(y1, 0.0), H), min(max(y2, 0.0), H)))
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_PX or h < MIN_BOX_PX:
        return None
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, w / W, h / H)


def clip_norm(xc, yc, w, h):
    x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
    y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_NORM or h < MIN_BOX_NORM:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, w, h)


# --------------------------------------------------------------------------- loaders
def load_yolo_source(cfg: dict) -> list[Sample]:
    root = Path(cfg["root"])
    names = cfg.get("names")
    if names is None:
        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            raise FileNotFoundError(f"{data_yaml} not found; set `names:` in the config instead")
        names = yaml.safe_load(data_yaml.read_text())["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    cmap = build_class_map(cfg["class_map"])
    id2uni = {i: cmap[norm(n)] for i, n in enumerate(names) if norm(n) in cmap}
    if not id2uni:
        print(f"  [warn] {cfg['name']}: none of {names} matched class_map {list(cmap)}")
    samples = []
    for img_dir in sorted(p for p in root.rglob("images") if p.is_dir()):
        lbl_dir = img_dir.parent / "labels"
        if not lbl_dir.is_dir():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            boxes = []
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                if cid not in id2uni:
                    continue
                vals = [float(v) for v in parts[1:]]
                if len(vals) > 4:  # polygon -> bbox
                    xs, ys = vals[0::2], vals[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    xc, yc, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                else:
                    xc, yc, w, h = vals
                b = clip_norm(xc, yc, w, h)
                if b:
                    boxes.append((id2uni[cid], *b))
            if boxes:
                samples.append(Sample(img, boxes, cfg["name"]))
    return samples


def load_coco_source(cfg: dict) -> list[Sample]:
    img_root = Path(cfg["images"])
    data = json.loads(Path(cfg["annotations"]).read_text())
    cmap = build_class_map(cfg["class_map"])
    id2uni = {c["id"]: cmap[norm(c["name"])] for c in data["categories"] if norm(c["name"]) in cmap}
    images = {im["id"]: im for im in data["images"]}
    per_img = defaultdict(list)
    for a in data["annotations"]:
        if a.get("iscrowd", 0) or a["category_id"] not in id2uni:
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        b = xyxy_to_yolo(x, y, x + w, y + h, im["width"], im["height"])
        if b:
            per_img[a["image_id"]].append((id2uni[a["category_id"]], *b))
    samples = []
    for iid, boxes in per_img.items():
        im = images[iid]
        samples.append(Sample(img_root / im["file_name"], boxes, cfg["name"], url=im.get("coco_url")))
    return samples


def _bdd_frames(path: Path):
    data = json.loads(path.read_text())
    return data["frames"] if isinstance(data, dict) and "frames" in data else data


def load_bdd_source(cfg: dict) -> list[Sample]:
    img_root = Path(cfg["images"])
    target = UNIFIED_CLASSES.index(cfg["target_class"])
    want_cat = norm(cfg.get("category", "traffic sign"))
    field = cfg.get("attr_field")
    values = {norm(v) for v in cfg.get("attr_values", [])}
    samples = []
    for fr in _bdd_frames(Path(cfg["annotations"])):
        img_path = img_root / fr["name"]
        if not img_path.exists():
            continue
        boxes, size = [], None
        for lab in fr.get("labels", []) or []:
            if norm(lab.get("category", "")) != want_cat or "box2d" not in lab:
                continue
            if field:
                v = lab.get(field, (lab.get("attributes") or {}).get(field))
                if v is None or norm(v) not in values:
                    continue
            if size is None:
                with Image.open(img_path) as im:
                    size = im.size
            bb = lab["box2d"]
            b = xyxy_to_yolo(bb["x1"], bb["y1"], bb["x2"], bb["y2"], *size)
            if b:
                boxes.append((target, *b))
        if boxes:
            samples.append(Sample(img_path, boxes, cfg["name"]))
    return samples


LOADERS = {"yolo": load_yolo_source, "coco": load_coco_source, "bdd": load_bdd_source}


# --------------------------------------------------------------------------- output
def prefetch(samples: list[Sample], workers: int = 16) -> list[Sample]:
    """Download missing images (COCO urls). Samples that cannot be fetched are dropped."""
    todo = [s for s in samples if not s.img_path.exists() and s.url]
    if todo:
        print(f"  downloading {len(todo)} missing images ...")

        def dl(s: Sample):
            try:
                s.img_path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(s.url, s.img_path)
            except Exception:
                s.img_path.unlink(missing_ok=True)

        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(dl, todo))
    kept = [s for s in samples if s.img_path.exists()]
    if len(kept) < len(samples):
        print(f"  [warn] dropped {len(samples) - len(kept)} samples with missing images")
    return kept


def write_dataset(samples_by_split: dict, out: Path, symlink: bool):
    stats = {sp: Counter() for sp in samples_by_split}
    for split, samples in samples_by_split.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(samples):
            stem = f"{s.source}_{i:06d}"
            dst = out / "images" / split / f"{stem}{s.img_path.suffix.lower()}"
            if symlink:
                dst.unlink(missing_ok=True)
                dst.symlink_to(s.img_path.resolve())
            else:
                shutil.copy2(s.img_path, dst)
            lines = [f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in s.boxes]
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            stats[split].update(UNIFIED_CLASSES[b[0]] for b in s.boxes)
    data_yaml = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {i: n for i, n in enumerate(UNIFIED_CLASSES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML listing the sources (see configs/sources.example.yaml)")
    ap.add_argument("--out", default="data/merged")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--symlink", action="store_true", help="symlink images instead of copying")
    ap.add_argument("--inspect-bdd", nargs=2, metavar=("JSON", "FIELD"),
                    help="print the distinct values of FIELD on traffic-sign labels, then exit")
    args = ap.parse_args()

    if args.inspect_bdd:
        path, field = args.inspect_bdd
        vals = Counter()
        for fr in _bdd_frames(Path(path)):
            for lab in fr.get("labels", []) or []:
                vals[str(lab.get(field, (lab.get("attributes") or {}).get(field)))] += 1
        for v, n in vals.most_common():
            print(f"{n:8d}  {v}")
        return
    if not args.config:
        ap.error("--config is required")

    cfg = yaml.safe_load(Path(args.config).read_text())
    rng = random.Random(args.seed)
    by_source = {}
    for src in cfg["sources"]:
        print(f"[{src['name']}] type={src['type']}")
        samples = LOADERS[src["type"]](src)
        rng.shuffle(samples)
        if src.get("max_images"):
            samples = samples[: src["max_images"]]
        samples = prefetch(samples)
        n_inst = Counter(UNIFIED_CLASSES[b[0]] for s in samples for b in s.boxes)
        print(f"  kept {len(samples)} images, instances: {dict(n_inst)}")
        by_source[src["name"]] = samples

    # stratified split: every source contributes to both train and val
    split = {"train": [], "val": []}
    for samples in by_source.values():
        n_val = int(len(samples) * args.val_frac) if len(samples) > 1 else 0
        split["val"] += samples[:n_val]
        split["train"] += samples[n_val:]
    for sp in split:
        rng.shuffle(split[sp])

    out = Path(args.out)
    stats = write_dataset(split, out, args.symlink)
    print("\n=== merged dataset ===")
    for sp, c in stats.items():
        print(f"{sp:5s}: {len(split[sp]):6d} images | " + ", ".join(f"{k}={c.get(k, 0)}" for k in UNIFIED_CLASSES))
    for k in UNIFIED_CLASSES:
        if sum(stats[sp].get(k, 0) for sp in stats) == 0:
            print(f"[WARNING] class '{k}' has ZERO instances - check your sources/class_map")
    print(f"data.yaml written to {out / 'data.yaml'}")


if __name__ == "__main__":
    main()
