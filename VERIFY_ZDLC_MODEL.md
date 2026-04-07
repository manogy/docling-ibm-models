# How to Verify Your ZDLC Model is Correct

This guide explains how to verify that your ZDLC compiled model (`.so` file) produces the same results as the original ONNX model.

## Overview

The [`compare_onnx_vs_zdlc.py`](compare_onnx_vs_zdlc.py:1) script compares:
- **ONNX Model** (original PyTorch model exported to ONNX format)
- **ZDLC Model** (ONNX model compiled to `.so` file for IBM Z)

It runs the same input through both models and compares:
1. Raw logits (model outputs before softmax)
2. Probabilities (after softmax)
3. Top predictions
4. Statistical metrics (mean, std, min, max, correlation)

## Prerequisites

### On IBM Z System

```bash
# Install onnxruntime
pip install onnxruntime

# Verify zdlc_pyrt is available
python -c "import zdlc_pyrt; print('✅ zdlc_pyrt available')"
```

## Usage

### Step 1: Copy Script to IBM Z

```bash
cd /root/manogya/manogya/docling-ibm-models
git pull origin main
```

The script is now available at: [`compare_onnx_vs_zdlc.py`](compare_onnx_vs_zdlc.py:1)

### Step 2: Run the Comparison

```bash
python compare_onnx_vs_zdlc.py
```

### Step 3: Choose Model to Test

The script will prompt you:

```
Which ZDLC model do you want to test?
1. CPU model
2. NNPA model

Enter choice (1 or 2):
```

- Enter `1` to test CPU model
- Enter `2` to test NNPA model

### Step 4: Review Results

The script will output detailed comparisons for each test image.

## Understanding the Output

### 1. Raw Logits Statistics

```
1. RAW LOGITS STATISTICS
--------------------------------------------------------------------------------
Metric               ONNX                 ZDLC                 Difference
--------------------------------------------------------------------------------
Mean                 -2.345678            -2.345680            0.000002
Std Dev              4.567890             4.567888             0.000002
Min                  -11.234567           -11.234565           0.000002
Max                  15.678901            15.678899            0.000002
Range                26.913468            26.913464            0.000004
```

**What to look for:**
- ✅ **Good**: Differences < 0.01 (numerical precision)
- ⚠️ **Concerning**: Differences > 0.1 (minor compilation issues)
- ❌ **Critical**: Differences > 1.0 (major compilation problems)

### 2. Logits Difference Analysis

```
2. LOGITS DIFFERENCE ANALYSIS
--------------------------------------------------------------------------------
Mean Absolute Error:     0.000123
Max Absolute Error:      0.000456
Root Mean Square Error:  0.000234
Correlation:             0.999999
```

**What to look for:**
- ✅ **Excellent**: MAE < 0.01, Correlation > 0.99
- ✅ **Good**: MAE < 0.1, Correlation > 0.95
- ⚠️ **Acceptable**: MAE < 1.0, Correlation > 0.90
- ❌ **Critical**: MAE > 1.0 or Correlation < 0.90

### 3. Top 5 Logits Comparison

```
3. TOP 5 LOGITS (Raw Values)
--------------------------------------------------------------------------------
Rank   Class                          ONNX Logit      ZDLC Logit      Diff      
--------------------------------------------------------------------------------
1      bar_chart                      15.678901       15.678899       0.000002  ✅
2      line_chart                     -2.345678       -2.345680       0.000002  ✅
3      pie_chart                      -5.678901       -5.678903       0.000002  ✅
```

**What to look for:**
- ✅ All top 5 classes match between ONNX and ZDLC
- ❌ Different classes in top 5 = compilation problem

### 4. Top 5 Probabilities

```
4. TOP 5 PROBABILITIES (After Softmax)
--------------------------------------------------------------------------------
Rank   Class                          ONNX Prob       ZDLC Prob       Diff      
--------------------------------------------------------------------------------
1      bar_chart                      0.999900        0.999899        0.000001
2      line_chart                     0.000050        0.000051        0.000001
```

**What to look for:**
- ✅ Probability differences < 0.001
- ⚠️ Probability differences > 0.01
- ❌ Probability differences > 0.1

### 5. Prediction Agreement

```
5. PREDICTION AGREEMENT
--------------------------------------------------------------------------------
ONNX Prediction:  bar_chart (0.9999)
ZDLC Prediction:  bar_chart (0.9999)
✅ PREDICTIONS MATCH!
```

**What to look for:**
- ✅ **Critical**: Predictions must match!
- ❌ If predictions don't match, model is broken

### 6. Verdict

```
6. VERDICT
--------------------------------------------------------------------------------
✅ EXCELLENT: Models are nearly identical (MAE < 0.01)
✅ EXCELLENT: Very high correlation (> 0.99)
```

**Possible verdicts:**
- ✅ **EXCELLENT**: MAE < 0.01, Correlation > 0.99
- ✅ **GOOD**: MAE < 0.1, Correlation > 0.95
- ⚠️ **ACCEPTABLE**: MAE < 1.0, Correlation > 0.90
- ⚠️ **CONCERNING**: MAE < 5.0
- ❌ **CRITICAL**: MAE >= 5.0 or Correlation < 0.90

## Final Summary

```
OVERALL VERDICT
================================================================================
Model Type:           CPU
All Predictions Match: ✅ YES
Average MAE:          0.000123
Average Correlation:  0.999999

✅ VERDICT: ZDLC model is CORRECT and matches ONNX model!
```

## Interpreting Results

### ✅ Model is CORRECT if:
1. All predictions match between ONNX and ZDLC
2. MAE < 0.1
3. Correlation > 0.99
4. Top 5 classes are the same

### ⚠️ Model is ACCEPTABLE if:
1. All predictions match
2. MAE < 1.0
3. Correlation > 0.95
4. Minor numerical differences due to quantization

### ❌ Model is BROKEN if:
1. Predictions don't match
2. MAE > 5.0
3. Correlation < 0.90
4. Different top classes

## Common Issues

### Issue 1: CPU Model Shows Uniform Probabilities

**Symptom:**
```
All classes have ~5% probability (1/20)
Logits are clipped to narrow range [-0.4, 0.2]
```

**Cause:** ZDLC compilation issue with quantization

**Solution:** Use NNPA model instead, or recompile with float32 precision

### Issue 2: Predictions Don't Match

**Symptom:**
```
ONNX Prediction:  bar_chart (0.9999)
ZDLC Prediction:  crossword_puzzle (0.9999)
❌ PREDICTIONS DO NOT MATCH!
```

**Cause:** Class label ordering bug (already fixed in our code)

**Solution:** Ensure you're using the latest code with the fix

### Issue 3: High MAE but Predictions Match

**Symptom:**
```
MAE: 2.5
Correlation: 0.92
✅ PREDICTIONS MATCH!
```

**Cause:** Quantization differences, but model still works

**Solution:** Acceptable for production if predictions are correct

## Example: Good NNPA Model

```
OVERALL VERDICT
================================================================================
Model Type:           NNPA
All Predictions Match: ✅ YES
Average MAE:          0.000234
Average Correlation:  0.999998

✅ VERDICT: ZDLC model is CORRECT and matches ONNX model!
```

## Example: Broken CPU Model

```
OVERALL VERDICT
================================================================================
Model Type:           CPU
All Predictions Match: ❌ NO
Average MAE:          8.456789
Average Correlation:  0.234567

❌ VERDICT: ZDLC model is INCORRECT - predictions do not match ONNX!
```

## Troubleshooting

### Script Fails to Import onnxruntime

```bash
pip install onnxruntime
```

### Script Fails to Import zdlc_pyrt

```bash
# Verify you're on IBM Z system
python -c "import zdlc_pyrt"
```

### Image Files Not Found

Update paths in script:
```python
TEST_IMAGES_DIR = "/your/path/to/test/images"
```

### ONNX Model Not Found

Update path in script:
```python
ONNX_MODEL_PATH = "/your/path/to/model.onnx"
```

## Next Steps

### If Model is CORRECT:
1. ✅ Use it in production
2. ✅ Integrate into your pipeline
3. ✅ Run performance benchmarks

### If Model is BROKEN:
1. ❌ Do NOT use in production
2. 🔧 Check ZDLC compilation settings
3. 🔧 Try different precision (float32 vs int8)
4. 🔧 Contact ZDLC team for support

## Related Files

- [`compare_onnx_vs_zdlc.py`](compare_onnx_vs_zdlc.py:1) - Main comparison script
- [`test_predictor_directly.py`](test_predictor_directly.py:1) - Simple predictor test
- [`demo/diagnose_cpu_model.py`](demo/diagnose_cpu_model.py:1) - CPU vs NNPA comparison
- [`CLASS_LABEL_BUG_FIX_SUMMARY.md`](CLASS_LABEL_BUG_FIX_SUMMARY.md:1) - Bug fix details

## Summary

This script provides a comprehensive way to verify your ZDLC model is correct by:
1. Comparing raw outputs (logits)
2. Comparing probabilities
3. Comparing predictions
4. Computing statistical metrics
5. Providing clear verdicts

**Use this before deploying any ZDLC model to production!**