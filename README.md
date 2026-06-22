# Tabler Local 45/46 Production Clean

Standalone local CSV pipeline for:

- General/table-structure model: `tabler/45`
- Column/split-column model: `tabler/46`

This folder is meant to be pushed to GitHub as a clean, production-style local runner. It does not depend on the original experiment folders.

## What Is Included

- `main.py`: runs local inference and then CSV extraction.
- `local_inference.py`: runs local ONNX/TensorRT inference from files inside `model_artifacts/`.
- `build_csv.py`: builds CSV files from raw predictions.
- `config.py`, `csv_extractor.py`, `image_utils.py`, `ocr_reader.py`: CSV extraction pipeline files required to run independently.
- `model_artifacts/tabler/45`: ONNX artifact and metadata for `tabler/45`.
- `model_artifacts/tabler/46`: ONNX artifact and metadata for `tabler/46`.
- `input/`: put source images here.
- `output/`: generated predictions, OCR, preprocessed images, and CSV files are written here.

## Important Git Note

The ONNX files are larger than 100 MB, so GitHub requires Git LFS.

Before committing:

```powershell
git lfs install
git lfs track "*.onnx"
```

This repository already includes `.gitattributes` for ONNX files. Do not remove it if you want the model artifacts to push correctly.

## Local Model Settings

- Pipeline profile: `local_45_46`
- General model: `tabler/45`
- Column model: `tabler/46`
- Inference artifact type: local ONNX files
- Preferred runtime: ONNX Runtime TensorRT provider, then CUDA provider, then CPU provider
- Input image variant for inference: JPEG quality `92`, subsampling `2`
- Inference confidence: `0.01`
- Normal local run requires no Roboflow API key because model artifacts are already stored locally.

## Class Thresholds

`tabler/45`:

- `column`: `0.51`
- `column_header`: `0.39`
- `row`: `0.41`
- `row_header`: `0.48`
- `spanning_column_cell`: `0.20`
- `spanning_column_header`: `0.39`
- `spanning_row_cell`: `0.20`
- `splitted_column`: `0.15`
- `table_header`: `0.29`
- `text`: `0.28`

`tabler/46`:

- `column`: `0.31`
- `splitted_column`: `0.37`

## How To Run

Put images into:

```powershell
C:\Users\ahirw\Downloads\python\tabler_local_45_46_production_clean\input
```

Run all images:

```powershell
cd "C:\Users\ahirw\Downloads\python\tabler_local_45_46_production_clean"
python main.py
```

Run one image:

```powershell
python main.py --single-image test5_54
```

Rebuild CSV from existing raw predictions without rerunning inference:

```powershell
python main.py --skip-inference
```

## Outputs

Generated files are written under:

```powershell
C:\Users\ahirw\Downloads\python\tabler_local_45_46_production_clean\output
```

Important generated folders:

- `output/raw_predictions/<image_stem>.json`
- `output/image_variants/<image_stem>_jpeg_q92_sub2.jpg`
- `output/preprocessed/pre_<filename>`
- `output/predictions_preprocessed/<image_stem>_predictions_preprocessed.json`
- `output/ocr/<image_stem>_ocr.json`
- `output/csv/<image_stem>.csv`

Generated output and input images are ignored by Git by default.

## Optional Environment Variables

Copy `.env.example` to `.env` only if you need to override defaults.

- `PIPELINE_PROFILE=local_45_46`
- `MODEL_CONFIDENCE=0.01`
- `REQUIRE_TENSORRT=1`
- `INFERENCE_PYTHON=<path-to-tested-python.exe>`
- `INFERENCE_OVERLAY_PATH=<optional-pythonpath-overlay>`
- `OCR_CUDA_VISIBLE_DEVICES=0`

## Offline Behavior

The required model files are already inside:

- `model_artifacts/tabler/45`
- `model_artifacts/tabler/46`

If these files are deleted or you change model versions, artifacts must be downloaded again separately before this local-only runner can work.
