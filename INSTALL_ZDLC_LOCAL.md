# Installing zdlc_pyrt from Local Directory

You have `zdlc_pyrt` in `/root/manogya/manogya/zdlc_pyrt` but it's not installed in your Python environment.

## Quick Fix - Install from Local Directory

```bash
# Navigate to the zdlc_pyrt directory
cd /root/manogya/manogya/zdlc_pyrt

# Install in development mode (recommended)
pip install -e .

# OR install normally
pip install .
```

## Verify Installation

```bash
# Check if zdlc_pyrt is installed
python -c "import zdlc_pyrt; print('zdlc_pyrt installed successfully!')"
```

## Now Run the Demo

```bash
# Go back to docling-ibm-models
cd /root/manogya/manogya/docling-ibm-models

# Run the test
python demo/test_zdlc_classifier.py \
    -i tests/test_data/figure_classifier/images \
    -z ../DocumentFigureClassifier-V2.so \
    -n 8
```

## Alternative: Add to PYTHONPATH (Temporary)

If you don't want to install, you can add to PYTHONPATH:

```bash
export PYTHONPATH="/root/manogya/manogya/zdlc_pyrt:$PYTHONPATH"

# Then run the demo
python demo/test_zdlc_classifier.py \
    -i tests/test_data/figure_classifier/images \
    -z ../DocumentFigureClassifier-V2.so \
    -n 8
```

## Complete Installation Steps

```bash
# 1. Install zdlc_pyrt
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .

# 2. Verify installation
python -c "import zdlc_pyrt; print('Success!')"

# 3. Go to docling-ibm-models
cd /root/manogya/manogya/docling-ibm-models

# 4. Install docling-ibm-models dependencies (if not done)
pip install -r requirements.txt

# 5. Run the ZDLC demo
python demo/test_zdlc_classifier.py \
    -i tests/test_data/figure_classifier/images \
    -z ../DocumentFigureClassifier-V2.so \
    -n 8
```

## Expected Output After Installation

```
2026-04-06 15:00:00 [INFO] Downloading model artifacts from HuggingFace...
2026-04-06 15:00:05 [INFO] ✓ Downloaded to: /root/.cache/huggingface/...
======================================================================
ZDLC Document Figure Classifier Demo
======================================================================
Artifacts path: /root/.cache/huggingface/...
ZDLC model path: ../DocumentFigureClassifier-V2.so
Image directory: tests/test_data/figure_classifier/images
Number of threads: 8
======================================================================

Initializing ZDLC predictor...
✓ Predictor initialized successfully

Model Info:
  backend: ZDLC
  num_threads: 8
  classes: 16 classes

Loading 2 images...
  ✓ bar_chart.jpg - Size: (800, 600)
  ✓ map.jpg - Size: (1024, 768)

Running inference on 2 images...
✓ Inference complete!
  Total time: 150.5 ms
  Average per image: 75.2 ms

======================================================================
PREDICTION RESULTS
======================================================================

[1/2] bar_chart.jpg
----------------------------------------------------------------------
  1. bar_chart                      ████████████████████████████████████████ 0.9856 (98.56%)
  2. line_chart                     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0089 (0.89%)
  3. pie_chart                      ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0032 (0.32%)
  4. other                          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0015 (0.15%)
  5. flow_chart                     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0008 (0.08%)

[2/2] map.jpg
----------------------------------------------------------------------
  1. map                            ████████████████████████████████████████ 0.9923 (99.23%)
  2. remote_sensing                 ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0045 (0.45%)
  3. other                          ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0018 (0.18%)
  4. screenshot                     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0009 (0.09%)
  5. flow_chart                     ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ 0.0005 (0.05%)

======================================================================
Demo completed successfully!
======================================================================
```

## Troubleshooting

### Issue: "No module named 'zdlc_pyrt'"
**Solution**: Install zdlc_pyrt first
```bash
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .
```

### Issue: "pip: command not found"
**Solution**: Use pip3 or python -m pip
```bash
python -m pip install -e .
```

### Issue: "Permission denied"
**Solution**: You're running as root, so this shouldn't happen. But if it does:
```bash
pip install --user -e .
```

### Issue: zdlc_pyrt has dependencies
**Solution**: Install its requirements first
```bash
cd /root/manogya/manogya/zdlc_pyrt
# If it has a requirements.txt:
pip install -r requirements.txt
# Then install zdlc_pyrt:
pip install -e .
```

## Verify Your Setup

```bash
# Check Python version
python --version  # Should be 3.11.2

# Check if zdlc_pyrt is installed
python -c "import zdlc_pyrt; print(zdlc_pyrt.__file__)"

# Check if docling_ibm_models can import ZDLC predictor
python -c "from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import DocumentFigureClassifierPredictorZDLC; print('Success!')"
```

## Summary

The error occurs because `zdlc_pyrt` is in your filesystem but not installed in your Python environment. Simply install it with:

```bash
cd /root/manogya/manogya/zdlc_pyrt && pip install -e .
```

Then you can run the demo successfully!