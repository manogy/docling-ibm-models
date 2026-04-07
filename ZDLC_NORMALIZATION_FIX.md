# ZDLC Document Figure Classifier - Normalization Fix

## 🐛 Problem Identified

The ZDLC predictor was producing **completely wrong predictions**:
- `bar_chart.jpg` → predicted as `crossword_puzzle` (99.99%)
- `map.jpg` → predicted as `logo` (99.97%)

## 🔍 Root Cause

The ZDLC predictor was **missing ImageNet normalization** in the preprocessing pipeline.

### Original PyTorch Predictor (CORRECT)
```python
self._image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])
```

### ZDLC Predictor (INCORRECT - Before Fix)
```python
self._image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # Missing normalization!
])
```

## ✅ Solution Applied

Added ImageNet normalization to match the original predictor:

**File:** `docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py`

**Lines 86-99:** Updated preprocessing pipeline to include normalization

```python
self._image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])
```

## 📋 Next Steps

### 1. Copy Updated File to IBM Z System

```bash
# On your local machine (where you have the updated code)
scp docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/docling_ibm_models/document_figure_classifier_model/
```

### 2. Also Copy Updated Demo File

```bash
scp demo/demo_document_figure_classifier_predictor_zdlc.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/demo/
```

### 3. Re-run the Test

```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_nnpa_results/
```

## 🎯 Expected Results After Fix

With correct normalization, you should see:
- `bar_chart.jpg` → predicted as `bar_chart` (high confidence)
- `map.jpg` → predicted as `map` (high confidence)

## 📊 Why This Matters

**ImageNet Normalization** is critical because:
1. The EfficientNet model was **trained** with ImageNet normalization
2. Input pixel values must be in the **same distribution** as training data
3. Without normalization, the model sees completely different input ranges
4. This causes the model to produce nonsensical predictions

### Technical Details

- **Mean:** `[0.485, 0.456, 0.406]` - ImageNet RGB channel means
- **Std:** `[0.47853944, 0.4732864, 0.47434163]` - ImageNet RGB channel standard deviations

The normalization formula applied per channel:
```
normalized_value = (pixel_value - mean) / std
```

This transforms pixel values from `[0, 1]` range to approximately `[-2, 2]` range, matching the training distribution.

## 🔧 Other ZDLC Predictors

**Good News:** Other ZDLC predictors were implemented correctly:
- ✅ `code_formula_predictor_zdlc.py` - Has correct normalization
- ✅ `layout_predictor_zdlc.py` - Has correct normalization  
- ✅ `tf_predictor_zdlc.py` - Has correct normalization

Only the document figure classifier had this issue.

---

**Status:** ✅ Fixed - Ready for testing on IBM Z system