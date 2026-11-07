# YOLO11 Cone, Barrier, and Stop-Sign Detection

This project fine-tunes YOLO11n to detect cones, barriers, and stop signs. The
merged dataset contains 4,770 images and 6,398 instances: 4,293 training
images with 5,759 instances and 477 validation images with 639 instances.

## Edge benchmark

The PyTorch FP32 row is the baseline. Measurements are end-to-end
preprocess + inference + NMS, batch 1, image size 640, 100 validation images
preloaded in RAM, with 10 warm-up frames and CUDA synchronization before
timing.

| Model | Variant | Device | FPS | mAP50 | mAP50-95 | Size (MB) |
|---|---|---|---:|---:|---:|---:|
| yolo11n | PyTorch FP32 (baseline) | CPU | 11.3 | 0.938 | 0.789 | 5.5 |
| yolo11n | PyTorch FP32 (baseline) | GPU | 98.7 | 0.938 | 0.789 | 5.5 |
| yolo11n | PyTorch FP16 | GPU | 75.8 | 0.937 | 0.788 | 5.5 |
| yolo11n | ONNX FP32 | CPU | 15.8 | 0.939 | 0.787 | 10.7 |
| yolo11n | OpenVINO INT8 | CPU | 23.4 | 0.937 | 0.780 | 3.4 |
| yolo11n | Pruned 30% | CPU | 12.7 | 0.882 | 0.722 | 5.5 |
| yolo11n | Pruned 30% | GPU | 92.2 | 0.882 | 0.722 | 5.5 |

OpenVINO INT8 is the best CPU edge option measured here: 23.4 FPS versus
11.3 FPS for the baseline, with a 0.009 mAP50-95 decrease. The GPU baseline
was fastest at 98.7 FPS. FP16 was slower than FP32 on this run (75.8 versus
98.7 FPS), despite nearly identical accuracy, so that result should be
rechecked on the target deployment stack rather than assumed to be faster.
Unstructured pruning did not speed up dense kernels and reduced mAP50-95 by
0.067, so it is not recommended as a speed optimization.

### Hardware and software

See [results/hardware.txt](results/hardware.txt). This run used two Tesla T4
GPUs, an Intel Xeon CPU with 4 logical CPUs, PyTorch 2.11.0+cu130, Ultralytics
8.4.204, and ONNX Runtime 1.31.1. The runtime did not include the
`nvidia-smi` executable; GPU visibility and model names were confirmed through
PyTorch and `/proc/driver/nvidia`.

FPS measured on a T4 is not representative of a Jetson, Raspberry Pi, or
other edge board. Repeat the benchmark on the target device before selecting
the deployment variant.

## Reproduce the benchmark

```bash
python optimize.py \
  --weights weights/yolo11n_best.pt \
  --data data/merged/data.yaml \
  --images data/merged/images/val \
  --n 100 --prune 0.3 \
  --out results/benchmark.md
```
