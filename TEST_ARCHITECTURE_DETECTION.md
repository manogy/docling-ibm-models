# Architecture Detection Test Script

This test script demonstrates automatic architecture detection and backend selection (ZDLC vs PyTorch) in the docling-ibm-models library.

## Overview

The script `test_architecture_detection.py` automatically:
- Detects the system architecture (s390x vs others)
- Uses ZDLC backend on s390x systems
- Uses PyTorch backend on x86_64, ARM, and other architectures
- Tests both LayoutPredictor and DocumentFigureClassifierPredictor

## Prerequisites

### For All Architectures
```bash
pip install -r requirements.txt
```

### For s390x (IBM Z) Systems
Additionally install ZDLC:
```bash
pip install zdlc_pyrt
```

## Model Setup

Before running the test, you need to download the models:

### Option 1: Using Hugging Face Hub
```python
from huggingface_hub import snapshot_download

# Download layout model
layout_path = snapshot_download(
    repo_id="ds4sd/docling-models",
    allow_patterns="model_artifacts/layout/*"
)

# Download figure classifier model
figure_path = snapshot_download(
    repo_id="ds4sd/docling-models",
    allow_patterns="model_artifacts/figure_classifier/*"
)
```

### Option 2: Manual Download
Download models from the Hugging Face repository and place them in:
- `models/layout/` - Layout detection model
- `models/figure_classifier/` - Figure classification model

## Running the Test

### Basic Usage
```bash
python test_architecture_detection.py
```

### Customizing Model Paths

Edit the script to point to your model locations:

```python
# In main() function
layout_model_path = "path/to/your/layout/model"
figure_model_path = "path/to/your/figure/classifier/model"

# For s390x systems, also set:
layout_zdlc_path = "path/to/layout_model.so"
```

## Expected Output

The script will:

1. **Display System Information**
   - Platform details
   - Architecture (s390x, x86_64, arm64, etc.)
   - Python version

2. **Show Backend Selection**
   - On s390x: "ZDLC backend will be used"
   - On others: "PyTorch backend will be used"

3. **Test Layout Predictor**
   - Initialize predictor
   - Show configuration (backend, device, model info)
   - Run prediction on test image
   - Display detected layout elements

4. **Test Figure Classifier**
   - Initialize predictor
   - Show configuration
   - Run prediction on test image
   - Display classification results

5. **Show Test Summary**
   - Pass/fail status for each test
   - Overall result

## Example Output

```
================================================================================
SYSTEM INFORMATION
================================================================================
Platform: Linux-5.15.0-s390x-with-glibc2.31
Machine: s390x
Processor: s390x
Python version: 3.10.12
================================================================================

⚠️  Detected s390x architecture - ZDLC backend will be used
⚠️  Make sure to set layout_zdlc_path to your ZDLC .so file

================================================================================
TESTING LAYOUT PREDICTOR
================================================================================
Initializing LayoutPredictor from: models/layout
ZDLC model path: models/layout/model.so

Predictor Configuration:
  backend: ZDLC
  model_name: RTDetrForObjectDetection
  device: cpu
  num_threads: 4
  image_size: {'height': 1025, 'width': 1025}
  threshold: 0.3

Loading image: tests/test_data/samples/empty_iocr.png
Image size: (1024, 768)
Image mode: RGB

Running prediction...

Found 15 predictions:
  1. Label: text, Confidence: 0.952, BBox: (100.0, 50.0, 900.0, 150.0)
  2. Label: title, Confidence: 0.887, BBox: (100.0, 20.0, 900.0, 45.0)
  ...

✅ Layout Predictor test PASSED

================================================================================
TESTING DOCUMENT FIGURE CLASSIFIER
================================================================================
...

✅ Figure Classifier test PASSED

================================================================================
TEST SUMMARY
================================================================================
Layout Predictor: ✅ PASSED
Figure Classifier: ✅ PASSED
================================================================================
🎉 All tests PASSED!
```

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Missing safe tensors file
```
**Solution**: Download the models as described in "Model Setup" section.

### ZDLC Not Available on s390x
```
WARNING: Running on s390x but zdlc_pyrt not available, falling back to PyTorch
```
**Solution**: Install zdlc_pyrt: `pip install zdlc_pyrt`

### Test Image Not Found
```
⚠️  Test image not found: tests/test_data/samples/empty_iocr.png
```
**Solution**: Make sure you're running the script from the repository root, or update the image paths in the script.

## Architecture-Specific Notes

### s390x (IBM Z)
- ZDLC backend is automatically selected
- Requires `zdlc_model_path` parameter for LayoutPredictor
- ZDLC models must be compiled for s390x architecture
- Falls back to PyTorch if ZDLC is not available

### x86_64, ARM, and Others
- PyTorch backend is automatically selected
- No special configuration needed
- Uses standard PyTorch models with safetensors format

## Integration with Your Code

To use automatic architecture detection in your own code:

```python
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# The predictor automatically detects architecture
predictor = LayoutPredictor(
    artifact_path="path/to/model",
    zdlc_model_path="path/to/model.so",  # Only used on s390x
    device="cpu",
)

# Check which backend is being used
info = predictor.info()
print(f"Using backend: {info['backend']}")  # 'ZDLC' or 'PyTorch'

# Use the predictor normally - API is the same regardless of backend
predictions = predictor.predict(image)
```

## Additional Resources

- [Docling Documentation](https://github.com/DS4SD/docling)
- [Model Repository](https://huggingface.co/ds4sd/docling-models)
- [ZDLC Documentation](https://www.ibm.com/docs/en/zdlc)