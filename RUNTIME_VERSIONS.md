# Tested Runtime Versions

## Inference Runtime

Used for local ONNX/TensorRT prediction:

- Python: `3.10.9`
- inference: `1.2.13`
- inference-models: `0.28.7`
- onnxruntime-gpu: `1.20.2` for the best local UI-raw match found in experiments
- TensorRT: `10.12.0.36`
- TensorRT cu12 libs: `10.12.0.36`
- Torch: `2.11.0+cu126`
- TorchVision: `0.26.0+cu126`
- OpenCV: `4.10.0.84`
- Pillow: `12.2.0`
- NumPy: `2.2.6` in the inference environment used for raw prediction tests

## OCR / CSV Runtime

Used for PaddleOCR and CSV extraction:

- Python: `3.10.x`
- paddleocr: `3.4.0`
- paddlepaddle-gpu: `3.0.0`
- paddlex: `3.4.2`
- OpenCV: `4.10.0.84`
- Pillow: `12.2.0`
- NumPy: `1.26.4`
- python-dotenv: `1.2.2`
- requests: `2.33.1`

## Best Candidate Summary

- ONNX artifacts: local `model_artifacts/tabler/45` and `model_artifacts/tabler/46`
- Provider stack: `TensorrtExecutionProvider`, then `CUDAExecutionProvider`, then `CPUExecutionProvider`
- TensorRT FP16: enabled
- TensorRT workspace: `1 GiB`
- Input re-encode: JPEG `quality=92`, `subsampling=2`
- OCR: full preprocessed image
- Confidence: `0.01`
