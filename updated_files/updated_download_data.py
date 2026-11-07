#!/usr/bin/env python3
"""Download raw datasets into data/raw/ so that `merge_datasets` can combine them.

What is fully automatic (no login):
  --coco-stop     COCO 2017 annotations json (~240 MB). The stop-sign IMAGES themselves are fetched
                  later by merge_datasets (only the ones it keeps) from the coco_url in the json.

What needs a free Roboflow account + API key (https://app.roboflow.com -> Settings -> API Keys):
  --rf NAME=WORKSPACE/PROJECT/VERSION     e.g.  --rf cones_rf=some-team/traffic-cones-abc/3
  (find WORKSPACE/PROJECT/VERSION in the URL of a dataset page on universe.roboflow.com, e.g.
   universe.roboflow.com/<workspace>/<project>/dataset/<version>)

Manual only (registration required, optional): BDD100K images and the BDD100K-MUTCD labels.

Examples:
  python -m src.download_data --coco-stop
  python -m src.download_data --rf cones_rf=ws/cones/2 --rf barriers_rf=ws2/barriers/1 --rf-key $ROBOFLOW_API_KEY
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def _progress(block, size, total):
    if total > 0:
        pct = min(100, block * size * 100 // total)
        print(f"\r  {pct:3d}%", end="", flush=True)


def coco_annotations(raw: Path):
    target = raw / "coco" / "annotations" / "instances_train2017.json"
    if target.exists():
        print(f"[coco] already present: {target}")
        return
    (raw / "coco").mkdir(parents=True, exist_ok=True)
    zpath = raw / "coco" / "annotations_trainval2017.zip"
    print(f"[coco] downloading {COCO_ANN_URL}")
    urllib.request.urlretrieve(COCO_ANN_URL, zpath, _progress)
    print()
    with zipfile.ZipFile(zpath) as z:
        z.extract("annotations/instances_train2017.json", raw / "coco")
    zpath.unlink()
    print(f"[coco] ready: {target}")


def roboflow_dataset(name: str, spec: str, key: str, raw: Path):
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit("pip install roboflow   (needed for --rf)")
    try:
        workspace, project, version = spec.strip("/").split("/")
    except ValueError:
        sys.exit(f"--rf {name}={spec}: expected WORKSPACE/PROJECT/VERSION")
    out = raw / name
    print(f"[roboflow] {spec} -> {out}")
    Roboflow(api_key=key).workspace(workspace).project(project).version(int(version)).download(
        "yolov8", location=str(out))
    print(f"[roboflow] ready: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--coco-stop", action="store_true")
    ap.add_argument("--rf", action="append", default=[], metavar="NAME=WS/PROJECT/VER")
    ap.add_argument("--rf-key", default=os.environ.get("ROBOFLOW_API_KEY"))
    args = ap.parse_args()
    raw = Path(args.raw)
    raw.mkdir(parents=True, exist_ok=True)

    if args.coco_stop:
        coco_annotations(raw)
    if args.rf:
        if not args.rf_key:
            sys.exit("Roboflow needs an API key: pass --rf-key or set ROBOFLOW_API_KEY")
        for item in args.rf:
            name, _, spec = item.partition("=")
            roboflow_dataset(name, spec, args.rf_key, raw)
    if not args.coco_stop and not args.rf:
        ap.print_help()


if __name__ == "__main__":
    main()
