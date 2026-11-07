# Instructions for the coding agent

## Goal
Deliver the assignment "Object Detection + Distance Estimation for Robotics Navigation":
detect **cones, barriers, stop signs**, annotate each box as `Cone, 1.5m` / `Stop Sign, 3.2m`
using **geometry** (not a learned depth head), and show **edge-optimisation before/after results**
with FPS on CPU and GPU.

**HARD CONSTRAINT: the whole job must be finished and pushed today.** Prefer a working,
honest, documented result over an ambitious one. Never invent numbers: every figure in the README
must come from a command you actually ran.

## What is automatic and what is not (read first)
| Step | Automatic? | Notes |
|---|---|---|
| COCO stop-sign annotations + images | YES, no login | `python -m src.download_data --coco-stop`; images are fetched by the merge step |
| Cone and barrier datasets | YES, but needs a free **Roboflow API key** and the dataset's `workspace/project/version` | `--rf NAME=ws/project/version --rf-key KEY` |
| BDD100K / BDD100K-MUTCD | NO (registration) | optional; skip unless the user already has the files |
| Merge + label remap + YOLO conversion | YES | `src/merge_datasets.py`, config in `configs/sources.yaml` |
| Train, infer, benchmark | YES | `run_pipeline.sh` runs everything |

**Ask the user ONCE, at the start, for:** (1) `ROBOFLOW_API_KEY`, (2) the `workspace/project/version` of
one traffic-cone and one road-barrier dataset from universe.roboflow.com (if you have web access, search
Universe yourself: pick datasets with road/vehicle-viewpoint images, >1,500 images, a permissive licence,
and give the user the two choices to confirm), (3) camera height/field of view if known.
If the user cannot provide a Roboflow key, say so and stop on that step; do NOT invent datasets or
fabricate labels. A fallback is Open Images V7 via `fiftyone`
(`fiftyone.utils.openimages.get_classes()` lists the available class names; only use classes that
actually exist there).

One-command run once those are known:
```
ROBOFLOW_API_KEY=... CONES_RF=ws/cones/2 BARRIERS_RF=ws2/barriers/1 HOURS=2.0 bash run_pipeline.sh
```
The script stops if any class is empty after merging. Then read the printed `source classes:` lines, add the
real class names to `class_map` in `configs/sources.yaml`, and rerun from the merge step.

## Fixed technical decisions (do not re-debate)
- Detector: **Ultralytics YOLO11** (`pip install ultralytics`), transfer learning from COCO-pretrained
  `yolo11n.pt` (primary, edge-friendly). `yolo11s.pt` only if time remains (used as the
  "heavier vs lighter backbone" comparison).
- Classes (unified): `0 cone`, `1 barrier`, `2 stop_sign`.
- Distance = fusion of two geometric methods in `src/distance.py` (already written):
  - M1 known-size pinhole: `Z = fy * H_real / h_px`
  - M2 ground-plane: `D = h_cam / tan(atan((y_bottom - cy)/fy) - pitch)` (cones and barriers only;
    a stop sign's box bottom is on a pole, so it uses M1 only)
  - inverse-variance fusion in log-space, disagreement fallback, truncation handling.
- BDD100K's detection labels have only 10 classes: **no cone, barrier or stop sign**. So we merge
  datasets (see below). Mention this in the README.

## Repo layout (already created)
```
src/merge_datasets.py   label remap + YOLO conversion of multiple sources
src/distance.py         CameraModel, DistanceEstimator (M1 + M2 fusion)
src/train.py            YOLO11 fine-tuning (supports --hours time budget)
src/infer.py            detect + distance + annotate images/video (+ --track smoothing)
src/optimize.py         ONNX / OpenVINO INT8 / FP16 / pruning + FPS & mAP table
src/eval_distance.py    MAE/MAPE of M1, M2, fused vs tape-measured ground truth
configs/sources.example.yaml
```
Keep modules small and single-purpose. Do not put logic in notebooks. Add type hints and docstrings.

## Step-by-step plan (time budget)

### 1. Environment (10 min)
```
pip install -r requirements.txt
python -m src.distance            # self-test, must print ~10 m for the synthetic cone
```
Use a GPU runtime (Colab/Kaggle T4 or local CUDA). Check with `nvidia-smi`.

### 2. Data (45-60 min) - the most error-prone step
Use `src/download_data.py` (see the automation table above); do not guess paths, and `ls` each folder
after downloading.
1. `python -m src.download_data --coco-stop --rf cones_rf=<ws/project/ver> --rf barriers_rf=<ws/project/ver>`
   (needs `ROBOFLOW_API_KEY`). Record each dataset's name, URL and licence for the README.
2. Optional: BDD100K images + BDD100K-MUTCD labels (manual, registration). First run
   `python -m src.merge_datasets --inspect-bdd <labels.json> sign_subtype`, then uncomment the `bdd_stop`
   source in `configs/sources.yaml` with the exact printed stop-sign value. COCO alone is enough.
3. `configs/sources.yaml` already points at the downloader's folders. Fix `class_map` keys only if the merge
   prints that a source's classes did not match.
4. `python -m src.merge_datasets --config configs/sources.yaml --out data/merged`
5. Read the printed table. **All three classes must have instances.** If one class is <10% of the others,
   raise its `max_images` or lower the others. Open ~10 random label files and overlay them on the images
   (quick matplotlib/cv2 check) to confirm boxes are correct before training.

### 3. Training (1.5-2.5 h wall clock, run in background)
```
python -m src.train --data data/merged/data.yaml --model yolo11n.pt --hours 2.0 --batch 32
```
- If GPU memory is low, use `--batch 16` or `--batch -1`. Use `--cache` if RAM allows.
- If time is very tight use `--freeze 10` (backbone frozen) and `--hours 1.0`.
- Save the log. Record final mAP50 and mAP50-95 overall and per class.
- **While training runs**, do steps 4-5 on CPU with COCO-pretrained weights as a stand-in.

### 4. Distance calibration and validation (30-40 min, parallel with training)
Assumptions in `src/distance.py` (65 deg HFOV, 1.4 m camera height, 0 deg pitch, cone 0.70 m,
barrier 0.90 m, stop sign 0.75 m) are defaults, not facts.
- If the user supplies camera intrinsics or a phone photo set, calibrate:
  `calibrate_focal_px(box_h_px, real_h_m, true_dist_m)` from one tape-measured photo and pass
  `--focal-px/--ref-width` to `infer.py` and `eval_distance.py`.
- Build `data/dist_gt.csv` from the user's measured photos if they exist
  (`image,class,x1,y1,x2,y2,true_distance_m`) and run `python -m src.eval_distance ...`.
  If no measured data exists, say so in the README and instead report a sanity check on synthetic
  geometry and on a few images with visually plausible distances. **Do not fabricate ground truth.**

### 5. Inference demo (15 min)
```
python -m src.infer --weights weights/yolo11n_best.pt --source <val images dir> --out runs/infer
python -m src.infer --weights weights/yolo11n_best.pt --source <video.mp4> --track --out runs/infer
```
Save 6-10 good annotated examples (including a failure case) under `docs/examples/`.

### 6. Edge optimisation (30-45 min, after training)
```
python -m src.optimize --weights weights/yolo11n_best.pt --data data/merged/data.yaml \
    --images data/merged/images/val --prune 0.3 --out results/benchmark.md
```
(add `weights/yolo11s_best.pt` to `--weights` if you trained it).
- It produces FPS on CPU and GPU and mAP for FP32, FP16, ONNX, OpenVINO INT8 and pruned models.
  Variants that fail are skipped with a message - report only what ran, and say why anything is missing.
- Note in the README: unstructured pruning reduces weights but gives no real speed-up on dense kernels;
  INT8 and a smaller backbone are the effective levers.
- State the hardware used (GPU model, CPU model/cores).

### 7. README and push (30 min)
README.md must contain:
1. Problem + approach overview (one diagram or short paragraph).
2. Datasets actually used (names, URLs, licences, counts per class and split) and why BDD100K alone
   was insufficient.
3. Training setup and results (mAP overall and per class).
4. Distance method: formulas, the assumptions table (HFOV, camera height, pitch, class heights),
   the fusion/uncertainty logic, validation numbers (MAE/MAPE for M1, M2, fused) or an explicit statement
   that validation was limited, plus known failure modes (truncation, slopes, size variation, occlusion).
5. Optimisation: the before/after benchmark table from `results/benchmark.md`, plus a 2-3 sentence
   discussion of the accuracy/speed trade-off.
6. How to reproduce (exact commands) and the example images.
7. Limitations and next steps (nuScenes validation, homography BEV, optical flow).

Commit to the **designated assignment repository** (not a personal one), include `requirements.txt`,
`configs/sources.example.yaml` and the final `weights/*.pt` if size allows (otherwise a download link).
Do not commit raw datasets.

## Rules
- Run each script at least once before reporting it done. Fix errors instead of hand-waving.
- If something in a script contradicts a dataset's real format, adapt the script and tell the user.
- Keep the fused distance as the headline output; also log `z_height` and `z_ground` (already in
  `detections.jsonl`) for the report.
- If a step threatens the deadline, cut scope in this order: yolo11s comparison, BDD source, pruning,
  video tracking. Never cut: training, annotated outputs, FPS table, README.

## Definition of done
- [ ] `data/merged/data.yaml` with all 3 classes populated
- [ ] `weights/yolo11n_best.pt` + mAP table
- [ ] annotated examples labelled like `Cone, 1.5m`
- [ ] `results/benchmark.md` with CPU and GPU FPS and a before/after comparison
- [ ] README with assumptions, validation status and limitations
- [ ] everything pushed to the designated repo
