# Verify Normalization Fix - Step by Step Guide

## 📋 Overview

This guide helps you verify that the normalization fix for the ZDLC Document Figure Classifier is working correctly on your IBM Z system.

## 🔧 Prerequisites

Ensure you have copied the updated files to your IBM Z system:

```bash
# From your local machine, copy updated files
scp docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/docling_ibm_models/document_figure_classifier_model/

scp demo/demo_document_figure_classifier_predictor_zdlc.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/demo/

scp demo/test_normalization_fix.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/demo/
```

## ✅ Test 1: Quick ZDLC Test

Run the ZDLC demo to see if predictions are now correct:

```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_nnpa_results/
```

### Expected Output (After Fix):

```
Image: 'bar_chart.jpg'
----------------------------------------------------------------------
  1. bar_chart                      - 0.XXXX (XX.XX%)  ← Should be bar_chart!
  2. line_chart                     - 0.XXXX (XX.XX%)
  3. pie_chart                      - 0.XXXX (XX.XX%)
  ...

Image: 'map.jpg'
----------------------------------------------------------------------
  1. map                            - 0.XXXX (XX.XX%)  ← Should be map!
  2. remote_sensing                 - 0.XXXX (XX.XX%)
  3. other                          - 0.XXXX (XX.XX%)
  ...
```

### ❌ If Still Wrong:

If you still see wrong predictions like `crossword_puzzle` or `logo`, the updated file wasn't copied correctly. Verify:

```bash
# Check if the file has the normalization code
grep -A 5 "transforms.Normalize" \
    /root/manogya/manogya/docling-ibm-models/docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py
```

You should see:
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.47853944, 0.4732864, 0.47434163],
),
```

## ✅ Test 2: Compare PyTorch vs ZDLC

This test compares the original PyTorch model with the ZDLC model to verify they produce the same results:

```bash
cd /root/manogya/manogya/docling-ibm-models

python demo/test_normalization_fix.py \
    --pytorch-artifacts /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    --zdlc-model /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    --zdlc-artifacts /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    --image tests/test_data/figure_classifier/images/bar_chart.jpg \
    --num-threads 4
```

### Expected Output:

```
======================================================================
PyTorch Predictions:
======================================================================

Image 1:
----------------------------------------------------------------------
  1. bar_chart                      - 0.XXXXXX (XX.XX%)
  2. line_chart                     - 0.XXXXXX (XX.XX%)
  ...

======================================================================
ZDLC Predictions:
======================================================================

Image 1:
----------------------------------------------------------------------
  1. bar_chart                      - 0.XXXXXX (XX.XX%)
  2. line_chart                     - 0.XXXXXX (XX.XX%)
  ...

======================================================================
Comparison Results:
======================================================================

Image 1:
----------------------------------------------------------------------
✅ Top class MATCHES: bar_chart
   PyTorch prob: 0.XXXXXX
   ZDLC prob:    0.XXXXXX
   Difference:   0.00XXXX

======================================================================
✅ ALL PREDICTIONS MATCH! Normalization fix is working correctly.
======================================================================
```

## 🔍 Test 3: Verify Preprocessing Code

Check that the preprocessing pipeline is correct:

```bash
cd /root/manogya/manogya/docling-ibm-models

python3 << 'EOF'
import sys
sys.path.insert(0, '.')

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

# Check the preprocessing pipeline
predictor = DocumentFigureClassifierPredictorZDLC(
    artifacts_path="/root/manogya/manogya/DocumentFigureClassifier-v2.0",
    zdlc_model_path="/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so",
    num_threads=4,
)

print("\n" + "="*70)
print("Preprocessing Pipeline:")
print("="*70)
for i, transform in enumerate(predictor._image_processor.transforms):
    print(f"{i+1}. {transform}")
print("="*70)

# Check for Normalize transform
has_normalize = any(
    'Normalize' in str(type(t).__name__) 
    for t in predictor._image_processor.transforms
)

if has_normalize:
    print("\n✅ Normalize transform is present!")
    for t in predictor._image_processor.transforms:
        if 'Normalize' in str(type(t).__name__):
            print(f"   Mean: {t.mean}")
            print(f"   Std:  {t.std}")
else:
    print("\n❌ ERROR: Normalize transform is MISSING!")
    print("   The file was not updated correctly.")

print()
EOF
```

### Expected Output:

```
======================================================================
Preprocessing Pipeline:
======================================================================
1. Resize(size=(224, 224), interpolation=bilinear, max_size=None, antialias=warn)
2. ToTensor()
3. Normalize(mean=[0.485, 0.456, 0.406], std=[0.47853944, 0.4732864, 0.47434163])
======================================================================

✅ Normalize transform is present!
   Mean: [0.485, 0.456, 0.406]
   Std:  [0.47853944, 0.4732864, 0.47434163]
```

## 🐛 Troubleshooting

### Issue: Still seeing wrong predictions

**Solution:**
1. Verify the file was copied correctly
2. Check Python is not caching the old module:
   ```bash
   # Remove Python cache
   find /root/manogya/manogya/docling-ibm-models -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
   find /root/manogya/manogya/docling-ibm-models -type f -name "*.pyc" -delete
   ```
3. Re-run the test

### Issue: Import errors

**Solution:**
Ensure all dependencies are installed:
```bash
pip install torch torchvision transformers pillow numpy zdlc_pyrt
```

### Issue: ZDLC model not found

**Solution:**
Verify the model path:
```bash
ls -lh /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so
```

### Issue: Config files not found

**Solution:**
Verify the artifacts directory contains config files:
```bash
ls -lh /root/manogya/manogya/DocumentFigureClassifier-v2.0/
# Should show: config.json, preprocessor_config.json
```

## 📊 Performance Testing

After verifying correctness, test performance:

```bash
cd /root/manogya/manogya/docling-ibm-models

# Test with different thread counts
for threads in 1 2 4 8; do
    echo "Testing with $threads threads..."
    python -m demo.demo_document_figure_classifier_predictor_zdlc \
        -i tests/test_data/figure_classifier/images \
        -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
        -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
        -n $threads \
        -v viz_nnpa_results/ 2>&1 | grep "total|avg"
done
```

## ✅ Success Criteria

The fix is working correctly if:

1. ✅ `bar_chart.jpg` is predicted as `bar_chart` (not `crossword_puzzle`)
2. ✅ `map.jpg` is predicted as `map` (not `logo`)
3. ✅ PyTorch and ZDLC predictions match (same top class)
4. ✅ Preprocessing pipeline includes `Normalize` transform
5. ✅ Confidence scores are reasonable (>50% for correct class)

## 📝 Next Steps After Verification

Once verified:
1. Update other ZDLC predictors if needed
2. Run full test suite
3. Integrate into production WDU pipeline
4. Document performance benchmarks

---

**Created:** 2026-04-07  
**Status:** Ready for testing on IBM Z system