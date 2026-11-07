#!/usr/bin/env bash
# End-to-end: download -> merge -> train -> demo inference -> edge benchmark.
#
# Required env vars (get the two dataset specs from universe.roboflow.com URLs):
#   ROBOFLOW_API_KEY   free key from app.roboflow.com -> Settings -> API Keys
#   CONES_RF           workspace/project/version of a traffic-cone dataset
#   BARRIERS_RF        workspace/project/version of a road-barrier dataset
# Optional: MODEL (default yolo11n.pt)  HOURS (default 2.0)  BATCH (default 32)  DEVICE (default auto)
#
# Usage:  ROBOFLOW_API_KEY=xxx CONES_RF=ws/cones/2 BARRIERS_RF=ws2/barriers/1 bash run_pipeline.sh
set -euo pipefail

MODEL="${MODEL:-yolo11n.pt}"
HOURS="${HOURS:-2.0}"
BATCH="${BATCH:-32}"
NAME="$(basename "$MODEL" .pt)"
DEV=()
[ -n "${DEVICE:-}" ] && DEV=(--device "$DEVICE")

: "${ROBOFLOW_API_KEY:?set ROBOFLOW_API_KEY}"
: "${CONES_RF:?set CONES_RF=workspace/project/version}"
: "${BARRIERS_RF:?set BARRIERS_RF=workspace/project/version}"

pip install -q -r requirements.txt roboflow

echo "== 1/6 self-test of the distance module =="
python -m src.distance

echo "== 2/6 download =="
python -m src.download_data --coco-stop --rf "cones_rf=$CONES_RF" --rf "barriers_rf=$BARRIERS_RF"

echo "== 3/6 merge =="
python -m src.merge_datasets --config configs/sources.yaml --out data/merged | tee results_merge.log
if grep -q "ZERO instances" results_merge.log; then
  echo "A class is empty - fix class_map in configs/sources.yaml (see 'source classes' lines above) and rerun step 3."
  exit 1
fi

echo "== 4/6 train ($MODEL, ${HOURS}h) =="
python -m src.train --data data/merged/data.yaml --model "$MODEL" --hours "$HOURS" --batch "$BATCH" "${DEV[@]}" | tee results_train.log

echo "== 5/6 demo inference =="
mkdir -p samples
ls data/merged/images/val | head -24 | while read -r f; do cp "data/merged/images/val/$f" samples/; done
python -m src.infer --weights "weights/${NAME}_best.pt" --source samples --out runs/infer "${DEV[@]}"

echo "== 6/6 edge benchmark =="
python -m src.optimize --weights "weights/${NAME}_best.pt" --data data/merged/data.yaml \
  --images data/merged/images/val --prune 0.3 --out results/benchmark.md

echo "Done. See runs/infer/, results/benchmark.md, results_train.log"
