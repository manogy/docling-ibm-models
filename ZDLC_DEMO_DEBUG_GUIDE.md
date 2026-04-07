# ZDLC Document Figure Classifier Demo - Debugging Guide

## 📋 Overview

This guide helps debug the ZDLC Document Figure Classifier demo when you have:
- ✅ ONNX model file
- ✅ ZDLC compiled .so model
- ❌ NO PyTorch .pt model

## 🔍 Potential Issues and Solutions

### Issue 1: Missing Config Files

**Problem:** The predictor needs `config.json` to load class labels.

**Location in code:**
- [`document_figure_classifier_predictor_zdlc.py:100`](docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py:100)
  ```python
  config = AutoConfig.from_pretrained(artifacts_path)
  ```

**Solution:** You need either:
1. Local `config.json` file in artifacts directory
2. Download from HuggingFace (demo does this automatically if `-a` not provided)

**Files needed in artifacts directory:**
```
artifacts_directory/
├── config.json              # Required - contains id2label mapping
└── preprocessor_config.json # Optional - not used by ZDLC predictor
```

**Command to download config only:**
```bash
# On IBM Z system
from huggingface_hub import snapshot_download
download_path = snapshot_download(
    repo_id="ds4sd/DocumentFigureClassifier",
    revision="v1.0.0",
    allow_patterns=["config.json", "preprocessor_config.json"]
)
```

---

### Issue 2: ZDLC Model Path

**Problem:** The `.so` file path must be correct and accessible.

**Location in code:**
- [`document_figure_classifier_predictor_zdlc.py:84`](docling_ibm_models/document_figure_classifier_predictor_zdlc.py:84)
  ```python
  self._zdlc_session = zdlc_pyrt.InferenceSession(zdlc_model_path)
  ```

**Verification:**
```bash
# Check if .so file exists
ls -lh /path/to/your/model.so

# Check file permissions
chmod +r /path/to/your/model.so
```

---

### Issue 3: Image Preprocessing

**Problem:** Images must be preprocessed correctly with ImageNet normalization.

**Location in code:**
- [`document_figure_classifier_predictor_zdlc.py:89-98`](docling_ibm_models/document_figure_classifier_predictor_zdlc.py:89-98)
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

**This is CRITICAL:** Without normalization, predictions will be wrong!

---

### Issue 4: ZDLC Input/Output Format

**Problem:** ZDLC expects specific input format and returns specific output format.

**Input format (line 172-174):**
```python
numpy_images = np.stack(
    [img.numpy() for img in processed_images]
).astype(np.float32)
```
- Shape: `(batch_size, 3, 224, 224)`
- Type: `float32`
- Format: NCHW (batch, channels, height, width)
- Values: Normalized with ImageNet mean/std

**Output format (line 177):**
```python
outputs = self._zdlc_session.run([numpy_images])
```
- Returns: List of arrays
- `outputs[0]`: Logits or probabilities with shape `(batch_size, num_classes)`

**Potential issue:** If ZDLC model was compiled incorrectly, output might be wrong shape or type.

---

### Issue 5: Softmax Application

**Problem:** Model might output logits or probabilities depending on compilation.

**Location in code:**
- [`document_figure_classifier_predictor_zdlc.py:183-197`](docling_ibm_models/document_figure_classifier_predictor_zdlc.py:183-197)

**Current logic:**
```python
first_sample_sum = np.sum(np.abs(logits[0]))

if 0.99 < first_sample_sum < 1.01:
    # Already probabilities
    probs_batch = logits.tolist()
else:
    # Apply softmax to logits
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs_batch = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).tolist()
```

**Potential issue:** This heuristic might fail if:
- Logits happen to sum to ~1 (rare but possible)
- Probabilities don't sum to exactly 1 due to numerical precision

**Better approach:**
```python
# Check if values are in [0, 1] range and sum to ~1
is_prob = (
    np.all(logits >= 0) and 
    np.all(logits <= 1) and 
    np.allclose(np.sum(logits, axis=1), 1.0, atol=0.01)
)
```

---

### Issue 6: Class Label Mismatch

**Problem:** Model outputs might not match expected class order.

**Location in code:**
- [`document_figure_classifier_predictor_zdlc.py:102-103`](docling_ibm_models/document_figure_classifier_predictor_zdlc.py:102-103)
  ```python
  self._classes = list(config.id2label.values())
  self._classes.sort()  # ⚠️ This sorts alphabetically!
  ```

**Potential issue:** If model outputs classes in a different order than alphabetical, predictions will be wrong!

**Expected classes (alphabetically sorted):**
1. bar_chart
2. bar_code
3. chemistry_markush_structure
4. chemistry_molecular_structure
5. flow_chart
6. icon
7. line_chart
8. logo
9. map
10. other
11. pie_chart
12. qr_code
13. remote_sensing
14. screenshot
15. signature
16. stamp

**Verification:** Check if model outputs match this order or use `config.id2label` mapping directly.

---

## 🧪 Debugging Steps

### Step 1: Verify Dependencies

```python
import zdlc_pyrt
import torchvision
import transformers
import numpy as np
from PIL import Image

print("✅ All dependencies imported successfully")
```

### Step 2: Test ZDLC Model Loading

```python
import zdlc_pyrt

zdlc_model_path = "/path/to/your/model.so"
session = zdlc_pyrt.InferenceSession(zdlc_model_path)
print(f"✅ ZDLC model loaded: {zdlc_model_path}")
```

### Step 3: Test Config Loading

```python
from transformers import AutoConfig

artifacts_path = "/path/to/artifacts"
config = AutoConfig.from_pretrained(artifacts_path)
print(f"✅ Config loaded")
print(f"Number of classes: {len(config.id2label)}")
print(f"Classes: {list(config.id2label.values())}")
```

### Step 4: Test Image Preprocessing

```python
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

image_processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])

image = Image.open("tests/test_data/figure_classifier/images/bar_chart.jpg")
processed = image_processor(image)
print(f"✅ Image preprocessed")
print(f"Shape: {processed.shape}")  # Should be (3, 224, 224)
print(f"Type: {processed.dtype}")   # Should be torch.float32
print(f"Min: {processed.min():.3f}, Max: {processed.max():.3f}")
```

### Step 5: Test ZDLC Inference

```python
# Prepare batch
numpy_image = processed.numpy().astype(np.float32)
batch = np.expand_dims(numpy_image, axis=0)  # Add batch dimension
print(f"Batch shape: {batch.shape}")  # Should be (1, 3, 224, 224)

# Run inference
outputs = session.run([batch])
print(f"✅ Inference completed")
print(f"Output shape: {outputs[0].shape}")  # Should be (1, 16)
print(f"Output values: {outputs[0][0]}")
print(f"Sum: {np.sum(outputs[0][0]):.4f}")
```

### Step 6: Test Full Pipeline

```python
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

predictor = DocumentFigureClassifierPredictorZDLC(
    artifacts_path="/path/to/artifacts",
    zdlc_model_path="/path/to/model.so",
    num_threads=4,
)

image = Image.open("tests/test_data/figure_classifier/images/bar_chart.jpg")
predictions = predictor.predict([image])

print(f"✅ Full pipeline test")
print(f"Top prediction: {predictions[0][0]}")
```

---

## 🚀 Running the Demo

### Minimal Command (with HuggingFace download)

```bash
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/your/model.so
```

### With Local Artifacts (recommended)

```bash
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/your/model.so \
    -a /path/to/artifacts \
    -n 4 \
    -v viz_results/
```

---

## 🐛 Common Errors and Fixes

### Error: "No module named 'zdlc_pyrt'"

**Fix:** Install ZDLC Python runtime
```bash
pip install zdlc_pyrt
```

### Error: "Cannot load .so file"

**Possible causes:**
1. File doesn't exist - check path
2. Wrong architecture - .so compiled for different system
3. Missing dependencies - check ZDLC runtime installation

### Error: "Config not found"

**Fix:** Either:
1. Provide `-a` parameter with path to config.json
2. Let demo download from HuggingFace (requires internet)

### Error: Wrong predictions (e.g., bar_chart → crossword_puzzle)

**Cause:** Missing ImageNet normalization (FIXED in current version)

**Verify fix is applied:**
```bash
grep -A 5 "transforms.Normalize" \
    docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py
```

Should show:
```python
transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.47853944, 0.4732864, 0.47434163],
),
```

### Error: Shape mismatch

**Check:**
1. Input shape: `(batch_size, 3, 224, 224)`
2. Output shape: `(batch_size, 16)` for 16 classes

---

## 📊 Expected Output

For correct predictions:

```
Image: 'bar_chart.jpg'
----------------------------------------------------------------------
  1. bar_chart                      - 0.XXXX (>50%)
  2. line_chart                     - 0.XXXX
  3. pie_chart                      - 0.XXXX
  ...

Image: 'map.jpg'
----------------------------------------------------------------------
  1. map                            - 0.XXXX (>50%)
  2. remote_sensing                 - 0.XXXX
  3. other                          - 0.XXXX
  ...
```

---

## 📝 Summary

**Key files:**
1. **Predictor:** [`document_figure_classifier_predictor_zdlc.py`](docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py:1)
2. **Demo:** [`demo_document_figure_classifier_predictor_zdlc.py`](demo/demo_document_figure_classifier_predictor_zdlc.py:1)

**Critical requirements:**
- ✅ ZDLC .so model file
- ✅ config.json (from HuggingFace or local)
- ✅ ImageNet normalization (FIXED)
- ✅ Correct input format: `(batch, 3, 224, 224)` float32
- ✅ zdlc_pyrt installed

**Most common issue:** Missing ImageNet normalization → FIXED in current version