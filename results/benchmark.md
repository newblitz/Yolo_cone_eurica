| Model | Variant | Device | FPS | mAP50 | mAP50-95 | File size (MB) |
|---|---|---|---|---|---|---|
| yolo11n | PyTorch FP32 | CPU | 11.3 | 0.938 | 0.789 | 5.5 |
| yolo11n | PyTorch FP32 | GPU | 103.2 | 0.938 | 0.789 | 5.5 |
| yolo11n | PyTorch FP16 | GPU | 79.6 | 0.937 | 0.788 | 5.5 |
| yolo11n | ONNX FP32 | CPU | 15.8 | 0.939 | 0.787 | 10.7 |
| yolo11n | OpenVINO INT8 | CPU | 23.4 | 0.937 | 0.780 | 3.4 |
| yolo11n | Pruned 30% | CPU | 12.7 | 0.882 | 0.722 | 5.5 |
| yolo11n | Pruned 30% | GPU | 92.2 | 0.882 | 0.722 | 5.5 |
