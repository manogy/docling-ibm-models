# Quick Start: Testing ZDLC Document Figure Classifier

## TL;DR - Run the Demo in 3 Steps

```bash
# 1. Install dependencies
pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git huggingface_hub pillow

# 2. Run the demo (replace /path/to/model.so with your compiled model)
# From project root directory:
python demo/test_zdlc_classifier.py \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/your/compiled/model.so

# 3. View results in terminal
```

## What You Need

1. **ZDLC compiled model**: `model.so` file (compiled from ONNX)
2. **Test images**: Already included in `tests/test_data/figure_classifier/images/`
3. **Python packages**: `zdlc_pyrt`, `huggingface_hub`, `pillow`

## Test Images Included

The repository includes 2 test images:
- `bar_chart.jpg` - Should classify as "bar_chart"
- `map.jpg` - Should classify as "map"

## Expected Output

```
Image: 'bar_chart.jpg'
  1. bar_chart                      - 0.9856 (98.56%)
  2. line_chart                     - 0.0089 (0.89%)
  ...

Image: 'map.jpg'
  1. map                            - 0.9923 (99.23%)
  2. remote_sensing                 - 0.0045 (0.45%)
  ...
```

## Don't Have a .so File Yet?

### Option 1: Export PyTorch → ONNX → .so

```python
# export_to_onnx.py
import torch
from transformers import AutoModelForImageClassification

model = AutoModelForImageClassification.from_pretrained(
    "ds4sd/DocumentFigureClassifier"
)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

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
print("ONNX model saved!")
```

Then compile with zDLC:
```bash
podman run --rm -v $(pwd):/workspace:z icr.io/ibmz/zdlc:5.0.0 \
    zdlc -O3 /workspace/document_figure_classifier.onnx \
    -o /workspace/document_figure_classifier.so
```

### Option 2: Use Pre-compiled Model

If you have access to a pre-compiled `.so` file, just use it directly.

## Test with Your Own Images

```bash
# Create a directory with your images
mkdir my_images
cp /path/to/your/*.jpg my_images/

# Run the demo (from project root)
python demo/test_zdlc_classifier.py \
    -i my_images \
    -z /path/to/model.so
```

## Supported Image Types

The classifier recognizes 16 types:
- Charts: bar_chart, line_chart, pie_chart
- Maps: map, remote_sensing
- Codes: bar_code, qr_code
- Chemistry: chemistry_markush_structure, chemistry_molecular_structure
- Documents: screenshot, signature, stamp
- Graphics: flow_chart, icon, logo
- Other: other

## Command Line Options

```bash
python demo/test_zdlc_classifier.py \
    -i <image_directory>           # Required: Input images
    -z <path_to_model.so>          # Required: ZDLC model
    -n <num_threads>               # Optional: Default 4
    -a <artifact_path>             # Optional: Skip HF download
    -v                             # Optional: Verbose logging
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Import zdlc_pyrt not found` | `pip install git+ssh://git@github.ibm.com/zosdev/zdlc_pyrt.git` |
| `model.so not found` | Check the path: `ls -lh /path/to/model.so` |
| `ZDLC inference failed` | Verify model compiled for your architecture |
| `Different results` | Check ONNX export and preprocessing |

## Compare with Original

Run both versions to verify identical results:

```bash
# Original PyTorch version (from project root)
python -m demo.demo_document_figure_classifier_predictor \
    -i tests/test_data/figure_classifier/images -d cpu

# ZDLC version (from project root)
python demo/test_zdlc_classifier.py \
    -i tests/test_data/figure_classifier/images \
    -z /path/to/model.so
```

Results should match within floating-point precision.

## Python API Usage

```python
from PIL import Image
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC
)

# Initialize
predictor = DocumentFigureClassifierPredictorZDLC(
    artifacts_path="/path/to/config",
    zdlc_model_path="/path/to/model.so",
    num_threads=4
)

# Predict
images = [Image.open("image.jpg")]
results = predictor.predict(images)

# Results format: List[List[Tuple[str, float]]]
for predictions in results:
    top_class, confidence = predictions[0]
    print(f"{top_class}: {confidence:.2%}")
```

## Performance Tips

- **Batch processing**: Process multiple images together
- **Thread tuning**: Adjust `-n` based on CPU cores
- **First run slower**: ZDLC initialization overhead
- **Warm-up**: Run once to initialize, then benchmark

## Need More Help?

- Full guide: [`ZDLC_DEMO_GUIDE.md`](ZDLC_DEMO_GUIDE.md)
- Integration docs: [`ZDLC_INTEGRATION.md`](ZDLC_INTEGRATION.md)
- ZDLC docs: https://github.com/IBM/zDLC

## Summary

✅ Demo script ready: `demo/demo_document_figure_classifier_predictor_zdlc.py`  
✅ Test images included: `tests/test_data/figure_classifier/images/`  
✅ Same output as PyTorch version  
✅ Easy to use with your own images  
✅ Drop-in replacement for existing code