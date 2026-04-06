# ZDLC Troubleshooting Guide

This guide helps diagnose and fix issues with ZDLC model predictions.

## Problem: Uniform Low Predictions (~5% for all classes)

### Symptoms
```
1. logo                           0.0530 (5.30%)
2. line_chart                     0.0504 (5.04%)
3. bar_chart                      0.0489 (4.89%)
4. chemistry_markush_structure    0.0460 (4.60%)
5. chemistry_molecular_structure  0.0442 (4.42%)
```

All predictions are similar and low (~5%), indicating the model is essentially guessing randomly.

### Root Causes

#### 1. **Input Preprocessing Mismatch** (Most Common)
The ZDLC model expects inputs in a specific format, but the predictor is sending different data.

**Check:**
- Does the ONNX model include preprocessing?
- Are pixel values in the correct range?
- Is normalization applied correctly?

#### 2. **ONNX Export Issues**
The PyTorch → ONNX conversion may not have preserved the model correctly.

**Check:**
- Was the model exported with correct input/output names?
- Are dynamic axes configured properly?
- Is the opset version compatible?

#### 3. **ZDLC Compilation Problems**
The ONNX → `.so` compilation may have optimization issues.

**Check:**
- Was the correct optimization level used?
- Are there any compilation warnings?
- Is the target architecture correct?

## Diagnostic Steps

### Step 1: Test PyTorch Model First

```bash
# Test original PyTorch model
python -m demo.demo_document_figure_classifier_predictor \
    -i tests/test_data/figure_classifier/images \
    -d cpu
```

**Expected output:**
```
bar_chart.jpg:
  1. bar_chart    0.9856 (98.56%)
  2. line_chart   0.0089 (0.89%)
  ...
```

If PyTorch works correctly, the issue is with ZDLC integration.

### Step 2: Run Diagnostic Script

```bash
python demo/debug_zdlc_vs_pytorch.py
```

This script will:
- Test both PyTorch and ZDLC models
- Compare their outputs
- Show preprocessing details
- Identify specific issues

### Step 3: Check Input Preprocessing

The model expects:
- **Input shape**: `(batch_size, 3, 224, 224)`
- **Pixel range**: Normalized with mean=[0.485, 0.456, 0.406], std=[0.479, 0.473, 0.474]
- **Color space**: RGB
- **Data type**: float32

Verify preprocessing:
```python
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Load image
img = Image.open("test.jpg").convert("RGB")

# Apply preprocessing
processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])

processed = processor(img).numpy()
print(f"Shape: {processed.shape}")  # Should be (3, 224, 224)
print(f"Range: [{processed.min():.3f}, {processed.max():.3f}]")
print(f"Mean: {processed.mean():.3f}")
```

## Solutions

### Solution 1: Re-export ONNX with Preprocessing

Export the model with preprocessing included:

```python
import torch
from transformers import AutoModelForImageClassification
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = AutoModelForImageClassification.from_pretrained(
    "ds4sd/DocumentFigureClassifier"
)
model.eval()

# Create a wrapper that includes preprocessing
class ModelWithPreprocessing(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # Define preprocessing as part of the model
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std',
            torch.tensor([0.47853944, 0.4732864, 0.47434163]).view(1, 3, 1, 1)
        )
    
    def forward(self, x):
        # x is expected to be in range [0, 1]
        x = (x - self.mean) / self.std
        return self.model(x).logits

wrapped_model = ModelWithPreprocessing(model)

# Test input: (batch_size, 3, 224, 224) in range [0, 1]
dummy_input = torch.randn(1, 3, 224, 224).clamp(0, 1)

# Export to ONNX
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "model_with_preprocessing.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=14,
    do_constant_folding=True
)

print("ONNX model exported with preprocessing included")
```

Then compile with ZDLC:
```bash
podman run --rm -v $(pwd):/workspace:z icr.io/ibmz/zdlc:5.0.0 \
    zdlc -O3 /workspace/model_with_preprocessing.onnx \
    -o /workspace/model.so
```

### Solution 2: Adjust Predictor Preprocessing

If the ONNX model expects raw pixels [0-255]:

```python
# In document_figure_classifier_predictor_zdlc.py
# Change preprocessing to match ONNX expectations

# Instead of normalized values, send raw pixels
numpy_images = np.stack([
    np.array(img.resize((224, 224))) / 255.0  # Scale to [0, 1]
    for img in rgb_images
]).transpose(0, 3, 1, 2).astype(np.float32)
```

### Solution 3: Verify ONNX Model

Test the ONNX model before ZDLC compilation:

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# Load ONNX model
session = ort.InferenceSession("model.onnx")

# Prepare input
img = Image.open("test.jpg").convert("RGB")
img = img.resize((224, 224))
img_array = np.array(img).astype(np.float32) / 255.0
img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
img_array = np.expand_dims(img_array, 0)  # Add batch dim

# Run inference
outputs = session.run(None, {"pixel_values": img_array})
logits = outputs[0]

# Check output
print(f"Logits shape: {logits.shape}")
print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

# Apply softmax
import scipy.special
probs = scipy.special.softmax(logits[0])
print(f"Top prediction: {probs.max():.4f}")
```

If ONNX gives good predictions but ZDLC doesn't, the issue is in ZDLC compilation.

### Solution 4: Check ZDLC Compilation

Recompile with different settings:

```bash
# Try different optimization levels
zdlc -O2 model.onnx -o model_O2.so
zdlc -O3 model.onnx -o model_O3.so

# Check for warnings during compilation
zdlc -O3 model.onnx -o model.so 2>&1 | tee compilation.log
```

## Quick Fixes

### Fix 1: Use PyTorch Model Temporarily

While debugging ZDLC, use the PyTorch predictor:

```python
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor
)

predictor = DocumentFigureClassifierPredictor(
    artifacts_path=artifact_path,
    device="cpu",
    num_threads=4
)
```

### Fix 2: Add Debug Logging

Add logging to see what the model receives:

```python
# In document_figure_classifier_predictor_zdlc.py
import logging
logger = logging.getLogger(__name__)

# Before ZDLC inference
logger.debug(f"Input shape: {numpy_images.shape}")
logger.debug(f"Input range: [{numpy_images.min()}, {numpy_images.max()}]")
logger.debug(f"Input mean: {numpy_images.mean()}")

outputs = self._zdlc_session.run([numpy_images])

# After ZDLC inference
logger.debug(f"Output shape: {outputs[0].shape}")
logger.debug(f"Output range: [{outputs[0].min()}, {outputs[0].max()}]")
```

## Expected vs Actual

### Expected Behavior
```
bar_chart.jpg:
  1. bar_chart    0.9856 (98.56%)  ← High confidence
  2. line_chart   0.0089 (0.89%)
  3. pie_chart    0.0032 (0.32%)
```

### Current Behavior (Problem)
```
bar_chart.jpg:
  1. logo         0.0530 (5.30%)   ← Low, uniform
  2. line_chart   0.0504 (5.04%)   ← All similar
  3. bar_chart    0.0489 (4.89%)   ← No clear winner
```

## Getting Help

If issues persist:

1. **Check ZDLC logs**: Look for compilation warnings
2. **Test ONNX directly**: Verify ONNX model works before ZDLC
3. **Compare preprocessing**: Ensure input format matches expectations
4. **Contact support**: Provide diagnostic script output

## Summary

The uniform low predictions indicate:
- ✗ Model is not processing inputs correctly
- ✗ Likely preprocessing mismatch
- ✗ Need to verify ONNX export and ZDLC compilation

Follow the diagnostic steps above to identify and fix the specific issue.