# Step-by-Step ZDLC Setup Guide

## Current Situation

You have:
- ✓ ONNX model at: `/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx`
- ✓ ZDLC predictor code ready in: `docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py`
- ✓ Test scripts ready

**Issue:** Model giving uniform low predictions (~5%) - likely preprocessing mismatch

## Step 1: Check ONNX File (CRITICAL)

The `model.onnx` file you showed is only **133 bytes** - this is a Git LFS pointer, not the actual model!

```bash
# Check the file
python demo/check_onnx_file.py
```

**If it's a Git LFS pointer, download the actual model:**

```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0

# Install Git LFS (if not already installed)
git lfs install

# Download the actual model file
git lfs pull

# Verify the file is now large
ls -lh model.onnx
# Should show: model.onnx is 80-400MB (not 133 bytes!)
```

## Step 2: Test ONNX Model

Once you have the actual model file (>10MB), test it:

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Install onnxruntime if needed
pip install onnxruntime

# Test the ONNX model with different preprocessing
python demo/test_onnx_model.py
```

**What this does:**
- Tests 5 different preprocessing formats
- Shows which one gives good predictions (>50% confidence)
- Identifies the correct input format for ZDLC

**Expected output:**
```
Test 5: ImageNet normalized, NCHW:
  Shape: (1, 3, 224, 224)
  Top probability: 0.8523 (85.23%)
  ✓ Good confidence!
```

## Step 3: Compile with ZDLC

Once ONNX model works correctly:

```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0

# Compile ONNX to .so file
zdlc -O3 model.onnx -o model.so

# This will create model.so (the compiled model)
ls -lh model.so
```

**Compilation options:**
- `-O0`: Fast compile, slow inference
- `-O2`: Balanced (recommended for testing)
- `-O3`: Slow compile, fast inference (recommended for production)

## Step 4: Update ZDLC Predictor

Edit the predictor to use your compiled model:

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models
```

Open: `docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py`

Find this line (around line 45):
```python
model_path = artifacts_path / "model.so"
```

Change to:
```python
model_path = Path("/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.so")
```

## Step 5: Test ZDLC Predictor

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Test the ZDLC predictor
python demo/test_zdlc_classifier.py
```

**Expected output:**
```
Testing ZDLC Document Figure Classifier

Test Image: tests/test_data/figure_classifier/images/bar_chart.jpg

Predictions for bar_chart.jpg:
  1. bar_chart (85.23%)
  2. line_chart (8.45%)
  3. table (3.21%)
  ...

✓ Predictions look good!
```

## Troubleshooting

### Problem 1: Still Getting Low Predictions

**If predictions are still ~5% after following all steps:**

1. The ONNX model might not include preprocessing
2. You need to match the preprocessing exactly

**Solution:** Check the `preprocessor_config.json` file:

```bash
cat /root/manogya/manogya/DocumentFigureClassifier-v2.0/preprocessor_config.json
```

This will show the exact preprocessing parameters. Update the ZDLC predictor to match.

### Problem 2: ZDLC Command Not Found

```bash
# Check if ZDLC is installed
which zdlc

# If not found, install from zdlc_pyrt
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .
```

### Problem 3: Shape Mismatch Error

The model expects a specific input shape. Check with:

```bash
python demo/test_onnx_model.py
```

Look for the "Model Information" section showing input shape.

## Quick Command Reference

```bash
# 1. Check ONNX file
python demo/check_onnx_file.py

# 2. Download actual model (if needed)
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0 && git lfs pull

# 3. Test ONNX model
cd /Users/manogya/forkedrepos/docling-ibm-models && python demo/test_onnx_model.py

# 4. Compile with ZDLC
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0 && zdlc -O3 model.onnx -o model.so

# 5. Test ZDLC predictor
cd /Users/manogya/forkedrepos/docling-ibm-models && python demo/test_zdlc_classifier.py
```

## Understanding the Issue

The 133-byte `model.onnx` file is a **Git LFS pointer**, not the actual model. It looks like:

```
version https://git-lfs.github.com/spec/v1
oid sha256:abc123...
size 123456789
```

This is why you're getting low predictions - you're trying to run inference on a text file, not a neural network!

After `git lfs pull`, the file will be replaced with the actual ONNX model (80-400MB).

## Next Steps After Success

Once you have working predictions for the document figure classifier:

1. **Apply to other models:**
   - Layout model
   - Code/Formula model
   - TableFormer model

2. **Performance testing:**
   - Compare ZDLC vs PyTorch speed
   - Test with different optimization levels

3. **Production deployment:**
   - Use ZDLC predictors in your application
   - Enjoy faster inference!

## Files Created

All these files are ready in your workspace:

- `demo/check_onnx_file.py` - Check if ONNX is LFS pointer
- `demo/test_onnx_model.py` - Test ONNX with different preprocessing
- `demo/test_zdlc_classifier.py` - Test ZDLC predictor
- `ZDLC_ONNX_TESTING_GUIDE.md` - Detailed testing guide
- `STEP_BY_STEP_ZDLC_SETUP.md` - This file

## Summary

**The key issue:** Your `model.onnx` is a 133-byte Git LFS pointer, not the actual model.

**The solution:** Run `git lfs pull` to download the actual model file.

**Then:** Follow steps 2-5 above to test and compile.