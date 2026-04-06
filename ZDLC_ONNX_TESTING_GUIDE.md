# ZDLC ONNX Model Testing and Compilation Guide

## Overview

You have the ONNX model at: `/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx`

This guide will help you:
1. Test the ONNX model to verify it works correctly
2. Identify the correct preprocessing format
3. Compile it with ZDLC
4. Test the compiled model

## Step 1: Check ONNX Model File

First, verify the model file is complete:

```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0
ls -lh model.onnx
```

**Expected:** File should be **>10MB** (typically 80-400MB for vision models)

**If file is <1MB:** The model.onnx is likely a Git LFS pointer, not the actual model. You need to:

```bash
# Install Git LFS if not already installed
git lfs install

# Pull the actual model file
git lfs pull
```

## Step 2: Test ONNX Model Directly

Run the test script to find the correct preprocessing:

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Install onnxruntime if needed
pip install onnxruntime

# Run the test
python demo/test_onnx_model.py
```

This script will:
- Load the ONNX model
- Test 5 different preprocessing approaches
- Show which one gives good predictions (>50% confidence)

**Expected Output:**
```
Test 5: ImageNet normalized, NCHW:
  Shape: (1, 3, 224, 224)
  Top probability: 0.8523 (85.23%)
  ✓ Good confidence!
```

## Step 3: Compile with ZDLC

Once you've verified the ONNX model works, compile it:

```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0

# Compile with ZDLC (adjust optimization level as needed)
zdlc -O3 model.onnx -o model.so

# Or with more verbose output
zdlc -O3 -v model.onnx -o model.so
```

**ZDLC Compilation Options:**
- `-O0`: No optimization (fastest compilation, slowest inference)
- `-O1`: Basic optimization
- `-O2`: Standard optimization (recommended)
- `-O3`: Aggressive optimization (slowest compilation, fastest inference)
- `-v`: Verbose output

## Step 4: Update ZDLC Predictor Path

Update the model path in the ZDLC predictor:

```python
# In docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py

# Change this line:
model_path = artifacts_path / "model.so"

# To point to your compiled model:
model_path = Path("/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.so")
```

## Step 5: Test ZDLC Predictor

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Test with the ZDLC predictor
python demo/test_zdlc_classifier.py
```

**Expected Output:**
```
Predictions for bar_chart.jpg:
  1. bar_chart (85.23%)
  2. line_chart (8.45%)
  3. table (3.21%)
```

## Troubleshooting

### Issue 1: ONNX Model File Too Small

**Symptom:** `model.onnx` is only 133 bytes

**Cause:** Git LFS pointer file instead of actual model

**Solution:**
```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0
git lfs install
git lfs pull
ls -lh model.onnx  # Should now be >10MB
```

### Issue 2: Low Confidence Predictions

**Symptom:** All predictions around 5-10%

**Cause:** Wrong preprocessing format

**Solution:**
1. Run `python demo/test_onnx_model.py` to find correct format
2. Update the preprocessing in `document_figure_classifier_predictor_zdlc.py`
3. The correct format is likely: NCHW, ImageNet normalized

### Issue 3: ZDLC Compilation Fails

**Symptom:** `zdlc: command not found` or compilation errors

**Solution:**
```bash
# Check ZDLC installation
which zdlc
zdlc --version

# If not installed, install from your zdlc_pyrt directory
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .
```

### Issue 4: Shape Mismatch

**Symptom:** `RuntimeError: Input shape mismatch`

**Cause:** Model expects different input shape

**Solution:**
1. Check model input shape: `python demo/test_onnx_model.py`
2. Update image resize in predictor to match expected shape
3. Common shapes: (224, 224), (384, 384), (512, 512)

## Preprocessing Reference

The correct preprocessing for most vision models:

```python
import numpy as np
from PIL import Image

# 1. Resize image
img = Image.open(image_path).convert("RGB")
img = img.resize((224, 224))

# 2. Convert to numpy array
img_array = np.array(img).astype(np.float32)

# 3. Normalize to [0, 1]
img_array = img_array / 255.0

# 4. Transpose to NCHW format
img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW

# 5. Add batch dimension
img_array = np.expand_dims(img_array, 0)  # CHW -> NCHW

# 6. Apply ImageNet normalization
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.47853944, 0.4732864, 0.47434163]).reshape(1, 3, 1, 1)
img_array = (img_array - mean) / std
```

## Quick Reference Commands

```bash
# 1. Check model file size
ls -lh /root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx

# 2. Pull LFS files if needed
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0 && git lfs pull

# 3. Test ONNX model
cd /Users/manogya/forkedrepos/docling-ibm-models && python demo/test_onnx_model.py

# 4. Compile with ZDLC
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0 && zdlc -O3 model.onnx -o model.so

# 5. Test ZDLC predictor
cd /Users/manogya/forkedrepos/docling-ibm-models && python demo/test_zdlc_classifier.py
```

## Next Steps After Success

Once you have working predictions:

1. **Test other models:** Apply same process to layout, code_formula, and tableformer models
2. **Optimize compilation:** Try different ZDLC optimization levels
3. **Benchmark performance:** Compare ZDLC vs PyTorch inference speed
4. **Integration:** Use ZDLC predictors in your production code

## Support

If you encounter issues:
1. Check the ONNX model file size first
2. Run the test script to verify preprocessing
3. Check ZDLC compilation output for errors
4. Verify zdlc_pyrt is installed correctly