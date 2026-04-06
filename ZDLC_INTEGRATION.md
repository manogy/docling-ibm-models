# ZDLC Integration Guide for Docling IBM Models

This document explains how to use the ZDLC (IBM Z Deep Learning Compiler) versions of the docling_ibm_models predictors.

## Overview

ZDLC allows you to run ONNX models compiled into optimized `.so` (shared object) files for improved performance on IBM Z systems. This integration provides ZDLC-compatible predictor classes that maintain the same output datatypes as the original PyTorch-based predictors.

## Prerequisites

1. **Install zdlc_pyrt package**:
   ```bash
   pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git
   ```

2. **Compile ONNX models to .so files**:
   - Download the [IBM zDLC image](https://ibm.github.io/ibm-z-oss-hub/containers/zdlc.html)
   - Follow the [IBM zDLC instructions](https://github.com/IBM/zDLC/tree/main) to compile your ONNX models

## Available ZDLC Predictors

### 1. Code Formula Predictor (ZDLC)

**Location**: `docling_ibm_models/code_formula_model/code_formula_predictor_zdlc.py`

**Usage**:
```python
from docling_ibm_models.code_formula_model.code_formula_predictor_zdlc import (
    CodeFormulaPredictorZDLC
)

predictor = CodeFormulaPredictorZDLC(
    artifacts_path="/path/to/model/artifacts",  # Tokenizer & image processor
    zdlc_model_path="/path/to/compiled/model.so",
    num_threads=4
)

# Same interface as original predictor
results = predictor.predict(
    images=[image1, image2],
    labels=["code", "formula"],
    temperature=0.0
)
```

**Output**: `List[str]` - Same as original predictor

### 2. Document Figure Classifier (ZDLC)

**Location**: `docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py`

**Usage**:
```python
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC
)

predictor = DocumentFigureClassifierPredictorZDLC(
    artifacts_path="/path/to/config",
    zdlc_model_path="/path/to/compiled/model.so",
    num_threads=4
)

# Same interface as original predictor
results = predictor.predict(images=[image1, image2])
```

**Output**: `List[List[Tuple[str, float]]]` - Same as original predictor
- Each image gets a list of (class_name, confidence_score) tuples
- Sorted by confidence in descending order

### 3. Layout Predictor (ZDLC)

**Location**: `docling_ibm_models/layoutmodel/layout_predictor_zdlc.py`

**Usage**:
```python
from docling_ibm_models.layoutmodel.layout_predictor_zdlc import (
    LayoutPredictorZDLC
)

predictor = LayoutPredictorZDLC(
    artifact_path="/path/to/config",
    zdlc_model_path="/path/to/compiled/model.so",
    num_threads=4,
    base_threshold=0.3,
    blacklist_classes=set()
)

# Single image prediction
for bbox in predictor.predict(image):
    print(bbox)  # Dict with keys: "label", "confidence", "l", "t", "r", "b"

# Batch prediction
results = predictor.predict_batch([image1, image2])
```

**Output**: 
- `predict()`: `Iterable[dict]` - Generator of bounding box dicts
- `predict_batch()`: `List[List[dict]]` - List of bbox lists per image
- Same format as original predictor

### 4. TableFormer Predictor (ZDLC)

**Location**: `docling_ibm_models/tableformer/data_management/tf_predictor_zdlc.py`

**Usage**:
```python
from docling_ibm_models.tableformer.data_management.tf_predictor_zdlc import (
    TFPredictorZDLC
)

predictor = TFPredictorZDLC(
    config=config_dict,
    zdlc_model_path="/path/to/compiled/model.so",
    num_threads=4
)

# Same interface as original predictor
tf_output, matching_details = predictor.predict(
    iocr_page=iocr_page,
    table_bbox=table_bbox,
    table_image=table_image,
    scale_factor=scale_factor,
    correct_overlapping_cells=False
)
```

**Output**: `(tf_output, matching_details)` - Same tuple format as original predictor

## Key Differences from Original Predictors

### 1. Initialization Parameters

**Original**:
```python
predictor = OriginalPredictor(
    artifacts_path="/path/to/artifacts",
    device="cpu",  # or "cuda"
    num_threads=4
)
```

**ZDLC**:
```python
predictor = PredictorZDLC(
    artifacts_path="/path/to/artifacts",
    zdlc_model_path="/path/to/model.so",  # NEW: Path to compiled .so file
    num_threads=4
    # No 'device' parameter - ZDLC handles device management
)
```

### 2. Backend Information

```python
# Original
info = predictor.info()
# {'device': 'cpu', 'num_threads': 4, ...}

# ZDLC
info = predictor.info()
# {'backend': 'ZDLC', 'num_threads': 4, ...}
```

### 3. Internal Processing

- **Original**: Uses PyTorch tensors and GPU/CPU device management
- **ZDLC**: Uses NumPy arrays and ZDLC inference session
- **Output**: Both return the same Python datatypes (lists, dicts, tuples, etc.)

## Output Datatype Guarantees

All ZDLC predictors maintain the exact same output datatypes as the original predictors:

| Predictor | Method | Output Type |
|-----------|--------|-------------|
| CodeFormulaPredictor | `predict()` | `List[str]` |
| DocumentFigureClassifier | `predict()` | `List[List[Tuple[str, float]]]` |
| LayoutPredictor | `predict()` | `Iterable[dict]` |
| LayoutPredictor | `predict_batch()` | `List[List[dict]]` |
| TFPredictor | `predict()` | `Tuple[list, dict]` |

## Model Compilation Notes

When compiling your ONNX models with zDLC:

1. **Export your PyTorch models to ONNX format** first
2. **Ensure input/output names match** what the ZDLC predictors expect
3. **Test with sample data** to verify the compiled model works correctly

### Expected Input/Output Formats

The ZDLC predictors assume the following:

- **Code Formula Model**: 
  - Inputs: `[input_ids, attention_mask, images]`
  - Outputs: `[generated_token_ids]`

- **Document Figure Classifier**:
  - Inputs: `[pixel_values]` (batch_size, 3, 224, 224)
  - Outputs: `[logits]` (batch_size, num_classes)

- **Layout Model**:
  - Inputs: `[pixel_values]`
  - Outputs: `[logits, pred_boxes]`

- **TableFormer**:
  - Inputs: `[image_batch]`
  - Outputs: `[tag_seq, class_logits, coord_predictions]` (optional)

## Migration Example

### Before (Original PyTorch):
```python
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

predictor = LayoutPredictor(
    artifact_path="/models/layout",
    device="cpu",
    num_threads=4
)

results = list(predictor.predict(image))
```

### After (ZDLC):
```python
from docling_ibm_models.layoutmodel.layout_predictor_zdlc import (
    LayoutPredictorZDLC
)

predictor = LayoutPredictorZDLC(
    artifact_path="/models/layout",
    zdlc_model_path="/models/layout/compiled_model.so",  # Add this
    num_threads=4
    # Remove 'device' parameter
)

results = list(predictor.predict(image))  # Same output format!
```

## Performance Considerations

1. **First inference may be slower** due to ZDLC initialization
2. **Subsequent inferences should be faster** on IBM Z systems
3. **Batch processing is recommended** when possible for better throughput
4. **Thread count** can be tuned based on your system

## Troubleshooting

### Import Errors
If you see `Import "zdlc_pyrt" could not be resolved`:
- Ensure zdlc_pyrt is installed: `pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git`
- Check your Python environment

### Model Loading Errors
If the ZDLC session fails to initialize:
- Verify the `.so` file path is correct
- Ensure the model was compiled for your system architecture
- Check that the ONNX model export was successful

### Output Mismatch
If outputs don't match the original predictor:
- Verify the ONNX export preserved model behavior
- Check input preprocessing is identical
- Ensure output post-processing matches the original

## Support

For issues related to:
- **ZDLC compilation**: See [IBM zDLC documentation](https://github.com/IBM/zDLC)
- **zdlc_pyrt package**: Contact the zdlc_pyrt maintainers
- **Predictor integration**: Check this repository's issues

## Summary

The ZDLC predictors provide a drop-in replacement for the original PyTorch predictors with:
- ✅ Same output datatypes
- ✅ Same method signatures
- ✅ Compatible with existing code
- ✅ Optimized for IBM Z systems
- ✅ No changes to downstream processing required

Simply replace the import and add the `zdlc_model_path` parameter to start using ZDLC!