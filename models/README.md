# models/

Drop trained YOLO weights here. `service/inference_service.py` loads `models/best.pt`.

After training (Phase B / `notebooks/03-train-yolo11.ipynb`) you should see:

- `best.pt` — primary; what the demo loads
- (optional) `best.onnx` — ONNX export, runs on AMD/Intel/NVIDIA via ORT
- (optional) `best.engine` — TensorRT FP16, for NVIDIA edge (Jetson)

This directory's contents are gitignored except for this README.
