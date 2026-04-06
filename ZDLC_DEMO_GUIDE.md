# End-to-End ZDLC Demo Guide: Document Figure Classifier

This guide provides complete step-by-step instructions to test the ZDLC Document Figure Classifier using freely available test data.

## Prerequisites

### 1. Install Required Packages

```bash
# Install zdlc_pyrt
pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git

# Install docling_ibm_models dependencies
pip install -r requirements.txt

# Install additional demo dependencies
pip install huggingface_hub pillow
```

### 2. Prepare Your ZDLC Compiled Model

You need a `.so` file compiled from the ONNX model. If you don't have one yet:

#### Option A: Use Existing ONNX Model
If you have an ONNX model, compile it using zDLC:

```bash
# Pull the zDLC container
podman pull icr.io/ibmz/zdlc:5.0.0

# Compile your ONNX model
podman run --rm -v $(pwd):/workspace:z icr.io/ibmz/zdlc:5.0.0 \
    zdlc -O3 /workspace/model.onnx -o /workspace/model.so
```

#### Option B: Export PyTorch Model to ONNX First
If you only have the PyTorch model:

```python
import torch
from transformers import AutoModelForImageClassification

# Load the model
model = AutoModelForImageClassification.from_pretrained(
    "ds4sd/DocumentFigureClassifier"
)
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "document_figure_classifier.onnx",
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamic_axes={
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    },
    opset_version=14
)
```

Then compile the ONNX model to `.so` as shown in Option A.

## Test Data

The repository includes test images in:
```
tests/test_data/figure_classifier/images/
├── bar_chart.jpg
└── map.jpg
```

These are freely available test images included in the repository.

## Running the Demo

### Step 1: Verify Test Images Exist

```bash
ls -lh tests/test_data/figure_classifier/images/
```

You should see:
- `bar_chart.jpg` - A bar chart image
- `map.jpg` - A map image

### Step 2: Run the ZDLC Demo

```bash
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/your/compiled/model.so \
    -n 4
```

**Parameters:**
- `-i` or `--image_dir`: Directory containing test images
- `-z` or `--zdlc_model_path`: Path to your ZDLC compiled `.so` file
- `-n` or `--num_threads`: Number of threads (default: 4)
- `-v` or `--viz_dir`: Output directory for visualizations (optional)

### Step 3: Expected Output

```
2026-04-03 13:30:00 DocumentFigureClassifierPredictorZDLC INFO     Downloading model artifacts from HuggingFace...
2026-04-03 13:30:05 DocumentFigureClassifierPredictorZDLC INFO     Downloaded to: /home/user/.cache/huggingface/...
2026-04-03 13:30:05 DocumentFigureClassifierPredictorZDLC INFO     Initializing ZDLC Document Figure Classifier...
2026-04-03 13:30:05 DocumentFigureClassifierPredictorZDLC INFO     Artifacts path: /home/user/.cache/huggingface/...
2026-04-03 13:30:05 DocumentFigureClassifierPredictorZDLC INFO     ZDLC model path: /path/to/model.so
2026-04-03 13:30:06 DocumentFigureClassifierPredictorZDLC INFO     Model info: {'backend': 'ZDLC', 'num_threads': 4, 'classes': [...]}
2026-04-03 13:30:06 DocumentFigureClassifierPredictorZDLC INFO     Loading 2 images from tests/test_data/figure_classifier/images...
2026-04-03 13:30:06 DocumentFigureClassifierPredictorZDLC INFO     Running inference...
2026-04-03 13:30:07 DocumentFigureClassifierPredictorZDLC INFO     For 2 images(ms): [total|avg] = [150.5|75.2]
2026-04-03 13:30:07 DocumentFigureClassifierPredictorZDLC INFO     
======================================================================
PREDICTION RESULTS
======================================================================

Image: 'bar_chart.jpg'
----------------------------------------------------------------------
  1. bar_chart                      - 0.9856 (98.56%)
  2. line_chart                     - 0.0089 (0.89%)
  3. pie_chart                      - 0.0032 (0.32%)
  4. other                          - 0.0015 (0.15%)
  5. flow_chart                     - 0.0008 (0.08%)

Image: 'map.jpg'
----------------------------------------------------------------------
  1. map                            - 0.9923 (99.23%)
  2. remote_sensing                 - 0.0045 (0.45%)
  3. other                          - 0.0018 (0.18%)
  4. screenshot                     - 0.0009 (0.09%)
  5. flow_chart                     - 0.0005 (0.05%)

======================================================================
```

## Using Your Own Images

### Option 1: Add Images to Test Directory

```bash
# Copy your images
cp /path/to/your/images/*.jpg tests/test_data/figure_classifier/images/

# Run the demo
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/model.so
```

### Option 2: Use a Different Directory

```bash
# Create a directory with your images
mkdir -p my_test_images
cp /path/to/your/images/*.jpg my_test_images/

# Run the demo
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i my_test_images \
    -z /path/to/model.so
```

## Supported Image Formats

The demo supports:
- JPEG/JPG
- PNG
- Any format supported by PIL/Pillow

## Understanding the Output

### Classification Results

The model classifies images into 16 categories:
1. **bar_chart** - Bar charts and histograms
2. **bar_code** - Barcodes
3. **chemistry_markush_structure** - Chemical structure diagrams
4. **chemistry_molecular_structure** - Molecular structures
5. **flow_chart** - Flowcharts and diagrams
6. **icon** - Icons and symbols
7. **line_chart** - Line graphs
8. **logo** - Company logos
9. **map** - Geographic maps
10. **other** - Other figure types
11. **pie_chart** - Pie charts
12. **qr_code** - QR codes
13. **remote_sensing** - Satellite/aerial imagery
14. **screenshot** - Screenshots
15. **signature** - Signatures
16. **stamp** - Stamps and seals

### Output Format

For each image, you get:
- **List of tuples**: `[(class_name, probability), ...]`
- **Sorted by probability**: Highest confidence first
- **All 16 classes**: Complete probability distribution

## Comparing with Original PyTorch Predictor

To verify ZDLC output matches the original:

### Run Original Predictor
```bash
python -m demo.demo_document_figure_classifier_predictor \
    -i tests/test_data/figure_classifier/images \
    -d cpu
```

### Run ZDLC Predictor
```bash
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/model.so
```

### Compare Results
The class predictions and probabilities should be nearly identical (within floating-point precision).

## Programmatic Usage

You can also use the ZDLC predictor in your own Python code:

```python
from PIL import Image
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC
)

# Initialize predictor
predictor = DocumentFigureClassifierPredictorZDLC(
    artifacts_path="/path/to/huggingface/cache",
    zdlc_model_path="/path/to/model.so",
    num_threads=4
)

# Load images
images = [
    Image.open("tests/test_data/figure_classifier/images/bar_chart.jpg"),
    Image.open("tests/test_data/figure_classifier/images/map.jpg")
]

# Run inference
results = predictor.predict(images)

# Process results
for i, predictions in enumerate(results):
    print(f"\nImage {i+1}:")
    for class_name, probability in predictions[:3]:  # Top 3
        print(f"  {class_name}: {probability:.4f}")
```

## Troubleshooting

### Issue: "Import zdlc_pyrt could not be resolved"
**Solution**: Install zdlc_pyrt package
```bash
pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git
```

### Issue: "FileNotFoundError: model.so not found"
**Solution**: Verify the path to your compiled `.so` file
```bash
ls -lh /path/to/model.so
```

### Issue: "ZDLC inference failed"
**Solution**: Ensure the model was compiled for your system architecture
- Check zDLC compilation logs
- Verify ONNX export was successful
- Test with a simple ONNX model first

### Issue: "Different results from PyTorch version"
**Solution**: 
- Verify ONNX export preserved model behavior
- Check input preprocessing is identical
- Compare intermediate outputs if possible

### Issue: "Out of memory"
**Solution**: 
- Reduce batch size (process fewer images at once)
- Reduce number of threads
- Check system resources

## Performance Tips

1. **Batch Processing**: Process multiple images together for better throughput
2. **Thread Tuning**: Adjust `-n` parameter based on your CPU cores
3. **Image Preprocessing**: Resize large images before inference
4. **Warm-up**: First inference may be slower due to initialization

## Next Steps

After successfully running the demo:

1. **Test with your own images**: Use real-world data
2. **Integrate into your pipeline**: Use the predictor in your application
3. **Benchmark performance**: Compare ZDLC vs PyTorch speed
4. **Try other models**: Apply the same approach to other predictors

## Additional Resources

- [ZDLC Documentation](https://github.com/IBM/zDLC)
- [zdlc_pyrt Repository](https://github.ibm.com/zosdev/zdlc_pyrt)
- [ZDLC Integration Guide](ZDLC_INTEGRATION.md)
- [HuggingFace Model](https://huggingface.co/ds4sd/DocumentFigureClassifier)

## Summary

This demo shows:
- ✅ How to use ZDLC compiled models
- ✅ End-to-end inference pipeline
- ✅ Using freely available test data
- ✅ Same output format as PyTorch version
- ✅ Easy integration into existing code

The ZDLC predictor is a drop-in replacement that provides optimized inference on IBM Z systems while maintaining full compatibility with the original PyTorch implementation.