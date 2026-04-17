# Testing on IBM Z (s390x) System

This guide is specifically for testing the ZDLC integration on your IBM Z system at `/root/manogya/manogya/`.

## Quick Start

### 1. Navigate to the repository
```bash
cd /root/manogya/manogya/docling-ibm-models
```

### 2. Run the test script
```bash
python3 test_ibmz_architecture.py
```

## What the Test Does

The script `test_ibmz_architecture.py` will:

1. **Detect your s390x architecture** automatically
2. **Use ZDLC backend** (not PyTorch) for inference
3. **Test Layout Predictor** using:
   - Model: `/root/manogya/manogya/docling-layout-heron/`
   - ZDLC: `/root/manogya/manogya/docling-layout-heron-NNPA.so`
   - Test image: `tests/test_data/samples/empty_iocr.png`

4. **Test Figure Classifier** using:
   - Model: `/root/manogya/manogya/DocumentFigureClassifier-v2.0/`
   - ZDLC: `/root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so`
   - Test image: `tests/test_data/figure_classifier/images/bar_chart.jpg`

## Expected Output

```
================================================================================
SYSTEM INFORMATION
================================================================================
Platform: Linux-5.15.0-s390x-with-glibc2.31
Machine: s390x
Processor: s390x
Python version: 3.10.x
ZDLC version: Available
ZDLC backend: Will be used automatically on s390x
================================================================================

================================================================================
TESTING LAYOUT PREDICTOR
================================================================================
Model path: /root/manogya/manogya/docling-layout-heron
ZDLC model path: /root/manogya/manogya/docling-layout-heron-NNPA.so
Test image: tests/test_data/samples/empty_iocr.png

Initializing LayoutPredictor...

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

✅ Prediction completed!
Found 15 layout elements:

Layout elements by type:
  text: 10
  title: 3
  figure: 2

First 5 predictions:
  1. text: conf=0.952, bbox=(100,50,900,150)
  2. title: conf=0.887, bbox=(100,20,900,45)
  ...

✅ Layout Predictor test PASSED

================================================================================
TESTING DOCUMENT FIGURE CLASSIFIER
================================================================================
Model path: /root/manogya/manogya/DocumentFigureClassifier-v2.0
ZDLC model path: /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so
Test image: tests/test_data/figure_classifier/images/bar_chart.jpg

Initializing DocumentFigureClassifierPredictor...

Predictor Configuration:
  backend: ZDLC
  device: cpu
  num_threads: 4
  classes: 16 classes

Loading image: tests/test_data/figure_classifier/images/bar_chart.jpg
Image size: (800, 600)
Image mode: RGB

Running prediction...

✅ Prediction completed!

Top 5 predictions:
  1. bar_chart: 0.9876
  2. line_chart: 0.0098
  3. pie_chart: 0.0015
  4. other: 0.0008
  5. flow_chart: 0.0003

✅ Figure Classifier test PASSED

================================================================================
TEST SUMMARY
================================================================================
Layout Predictor: ✅ PASSED
Figure Classifier: ✅ PASSED
================================================================================
🎉 All tests PASSED!

The ZDLC backend is working correctly on your IBM Z system!
```

## Verifying ZDLC is Being Used

The test output will show:
- `backend: ZDLC` in the Predictor Configuration
- This confirms ZDLC is being used, not PyTorch

## Troubleshooting

### Issue: "ZDLC: Not available"
**Solution**: Install zdlc_pyrt
```bash
cd /root/manogya/manogya/zdlc_pyrt
pip install .
```

### Issue: "Image not found"
**Solution**: Make sure you're running from the repository root:
```bash
cd /root/manogya/manogya/docling-ibm-models
python3 test_ibmz_architecture.py
```

### Issue: "Model file not found"
**Solution**: Verify your model paths match the script:
```bash
ls -la /root/manogya/manogya/docling-layout-heron/
ls -la /root/manogya/manogya/docling-layout-heron-NNPA.so
ls -la /root/manogya/manogya/DocumentFigureClassifier-v2.0/
ls -la /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so
```

## Customizing the Test

If your paths are different, edit `test_ibmz_architecture.py`:

### For Layout Predictor:
```python
# Around line 48-50
model_path = "/your/path/to/docling-layout-heron"
zdlc_model_path = "/your/path/to/docling-layout-heron-NNPA.so"
test_image = "path/to/your/test/image.png"
```

### For Figure Classifier:
```python
# Around line 120-125
model_path = "/your/path/to/DocumentFigureClassifier-v2.0"
zdlc_model_path = "/your/path/to/DocumentFigureClassifier-V2-NNPA.so"
test_image = "path/to/your/test/image.jpg"
```

## Using in Your Own Code

Once the test passes, you can use the same pattern in your code:

```python
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from PIL import Image

# Initialize - ZDLC will be used automatically on s390x
predictor = LayoutPredictor(
    artifact_path="/root/manogya/manogya/docling-layout-heron",
    zdlc_model_path="/root/manogya/manogya/docling-layout-heron-NNPA.so",
    device="cpu",
)

# Check backend
info = predictor.info()
print(f"Using backend: {info['backend']}")  # Should print "ZDLC"

# Use predictor
image = Image.open("your_image.png")
predictions = list(predictor.predict(image))

for pred in predictions:
    print(f"{pred['label']}: {pred['confidence']:.3f}")
```

## Performance Notes

- ZDLC backend uses IBM Z Neural Network Processing Assist (NNPA)
- Should be faster than PyTorch on s390x
- Uses the `-NNPA.so` compiled models for hardware acceleration

## Next Steps

After successful testing:
1. The code automatically detects s390x and uses ZDLC
2. No code changes needed in your applications
3. Just ensure `zdlc_model_path` is provided when initializing predictors
4. The same code will work on x86_64/ARM (using PyTorch) without changes