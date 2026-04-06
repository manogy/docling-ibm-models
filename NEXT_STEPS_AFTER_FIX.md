# Next Steps After Preprocessing Fix

## What Was Fixed

Based on the ONNX testing results, we identified that the model expects:
- **Format:** NCHW (batch, channels, height, width)
- **Preprocessing:** Normalized to [0-1] range only
- **NO ImageNet normalization** (mean/std normalization)

The ZDLC predictor has been updated to remove the ImageNet normalization that was causing low confidence predictions.

## Step 1: Compile ONNX to ZDLC

```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0

# Compile with ZDLC (this creates model.so)
zdlc -O3 model.onnx -o model.so

# Verify the compiled model exists
ls -lh model.so
```

**Expected:** `model.so` file should be created (size varies, typically 10-50MB)

## Step 2: Test the Fixed ZDLC Predictor

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Run the test with fixed preprocessing
python demo/test_zdlc_after_fix.py
```

**Expected Output:**
```
Top 5 Predictions for bar_chart.jpg:
  1. bar_chart                      (99.99%)
  2. line_chart                     ( 0.01%)
  ...

✓ EXCELLENT! Top prediction has 99.99% confidence
  The preprocessing fix worked!
```

## Step 3: Compare with Original Test

You can also run the original test to see the improvement:

```bash
# Old test (for comparison)
python demo/test_zdlc_classifier.py
```

## Understanding the Fix

### Before (Wrong Preprocessing)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0-1] normalization
    transforms.Normalize(   # ImageNet normalization (WRONG!)
        mean=[0.485, 0.456, 0.406],
        std=[0.479, 0.473, 0.474],
    ),
])
```

This produced values in range `[-2.5, 2.5]` which the model didn't expect.

### After (Correct Preprocessing)
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0-1] normalization (CORRECT!)
])
```

This produces values in range `[0, 1]` which matches what the ONNX model expects.

## Why This Matters

The ONNX model was exported **without** the ImageNet normalization layer. This means:
1. The normalization was part of the PyTorch training pipeline
2. But it was NOT included in the ONNX export
3. So the ONNX/ZDLC model expects raw [0-1] normalized pixels

## Verification Steps

After running the test, verify:

1. **High Confidence:** Top prediction should be >90%
2. **Correct Class:** Should predict "bar_chart" for bar_chart.jpg
3. **Reasonable Distribution:** Other classes should have much lower probabilities

## Troubleshooting

### Issue: model.so not found

**Solution:**
```bash
cd /root/manogya/manogya/DocumentFigureClassifier-v2.0
zdlc -O3 model.onnx -o model.so
```

### Issue: Still getting low confidence

**Possible causes:**
1. ZDLC compilation failed - check for errors during compilation
2. Wrong model.so file - verify it's from the correct ONNX model
3. Image loading issue - verify test image exists

**Debug:**
```bash
# Check ONNX model works
python demo/test_onnx_model.py

# Check ZDLC compilation
zdlc -v -O3 model.onnx -o model.so
```

### Issue: Import errors

**Solution:**
```bash
# Install required packages
pip install torchvision transformers pillow

# Install zdlc_pyrt
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .
```

## Next Steps After Success

Once you have working predictions:

### 1. Apply Fix to Other Models

The same preprocessing issue likely affects other models. Update:
- `code_formula_model/code_formula_predictor_zdlc.py`
- `layoutmodel/layout_predictor_zdlc.py`
- `tableformer/data_management/tf_predictor_zdlc.py`

For each model:
1. Test the ONNX model first: `python demo/test_onnx_model.py`
2. Identify correct preprocessing
3. Update the ZDLC predictor
4. Compile and test

### 2. Performance Benchmarking

Compare ZDLC vs PyTorch:

```python
import time
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import DocumentFigureClassifierPredictor
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import DocumentFigureClassifierPredictorZDLC

# Test both predictors with same images
# Measure inference time
# Compare accuracy
```

### 3. Production Deployment

Once verified:
1. Use ZDLC predictors in your application
2. Enjoy faster inference (typically 2-5x speedup)
3. Lower memory usage
4. Better CPU utilization

## Summary

**Problem:** ImageNet normalization was applied but model didn't expect it
**Solution:** Remove ImageNet normalization, use only [0-1] normalization
**Result:** Should see 90%+ confidence predictions

**Key Learning:** Always test ONNX models directly before compiling to ZDLC to verify preprocessing requirements!