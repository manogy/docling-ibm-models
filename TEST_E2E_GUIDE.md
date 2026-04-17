# End-to-End Testing Guide for ZDLC Integration

This guide explains how to test the complete ZDLC integration flow across all three repositories: `watson_doc_understanding` → `wdu` → `docling-ibm-models`.

## Overview

The ZDLC integration has three layers:

1. **Layer 1: docling-ibm-models** - Core predictors with ZDLC backend
2. **Layer 2: wdu** - Model provider and configuration
3. **Layer 3: watson_doc_understanding** - Full service integration

## Prerequisites

### On s390x (IBM Z) Systems:
- ZDLC runtime installed (`zdlc_pyrt` package)
- Compiled ZDLC models (`.so` files)
- Model artifacts (config files)

### On Other Systems:
- PyTorch installed
- Model artifacts only (no ZDLC models needed)

## Test Scripts

### 1. Quick Layer Test (docling-ibm-models only)

Use the existing test script:

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# On s390x with ZDLC
python test_ibmz_architecture.py

# On other systems (uses PyTorch)
python test_architecture_detection.py
```

### 2. Comprehensive End-to-End Test

Use the new comprehensive test script:

```bash
cd /Users/manogya/forkedrepos/docling-ibm-models

# Run with default paths
python test_e2e_zdlc_flow.py

# Or with custom paths via environment variables
export LAYOUT_ARTIFACTS_PATH="/path/to/layout/artifacts"
export LAYOUT_ZDLC_PATH="/path/to/layout.so"
export CLASSIFIER_ARTIFACTS_PATH="/path/to/classifier/artifacts"
export CLASSIFIER_ZDLC_PATH="/path/to/classifier.so"
export TEST_IMAGE_PATH="tests/test_data/samples/empty_iocr.png"
export WDU_SERVICE_URL="http://localhost:8080"

python test_e2e_zdlc_flow.py
```

## Layer-by-Layer Testing

### Layer 1: docling-ibm-models Direct Test

Tests the core predictors directly:

```python
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor
from PIL import Image

# Initialize - auto-detects s390x and uses ZDLC if available
predictor = LayoutPredictor(
    artifact_path="/path/to/artifacts",
    zdlc_model_path="/path/to/model.so",  # Only needed on s390x
    device="cpu",
)

# Check backend
info = predictor.info()
print(f"Backend: {info['backend']}")  # "ZDLC" or "PyTorch"

# Run prediction
image = Image.open("test.png")
predictions = list(predictor.predict(image))
print(f"Found {len(predictions)} elements")
```

**Expected Output on s390x:**
```
Backend: ZDLC
Device: cpu
Found X layout elements
```

**Expected Output on other systems:**
```
Backend: PyTorch
Device: cpu
Found X layout elements
```

### Layer 2: wdu ModelsProvider Test

Tests model loading through wdu:

```python
from wdu.models.models_provider import ModelsProvider
from wdu.config.wdu_config import WduConfig

# Setup configuration
config = WduConfig()
# Or load from file: config = WduConfig.from_yaml("config.yaml")

# Verify paths are set
print(f"Layout ZDLC: {config.models.layout_model.zdlc_model_path}")
print(f"Classifier ZDLC: {config.models.document_figure_classifier.zdlc_model_path}")

# Setup and load models
ModelsProvider.setup(config)

# Load models - will auto-detect s390x and use ZDLC
layout_model = ModelsProvider.get_layout_model()
classifier_model = ModelsProvider.get_document_figure_classifier_model()

print("✅ Models loaded successfully")
```

**Configuration Check:**

Verify your `wdu_config.py` has correct paths:

```python
@dataclass
class LayoutModel(DataModelBase):
    weights_path: str = "./models/layout_model/artifacts"
    zdlc_model_path: str = "./models/layout_model/docling-layout-heron-NNPA.so"

@dataclass
class DocumentFigureClassifierModel(DataModelBase):
    weights_path: str = "./models/document_figure_classifier/artifacts"
    zdlc_model_path: str = "./models/document_figure_classifier/DocumentFigureClassifier-V2-NNPA.so"
```

### Layer 3: Full Service Test

Test the complete watson_doc_understanding service:

```bash
# Start the service (if not already running)
cd /path/to/watson_doc_understanding
# Follow service startup instructions

# Submit a test document
python examples/process_file.py \
    --server-url http://localhost:8080 \
    --input-file test.pdf \
    --output-folder ./output \
    --mode high_quality \
    --sync
```

Or use curl:

```bash
curl -X POST "http://localhost:8080/api/v2/task/process" \
  -F "file=@test.pdf" \
  -F 'parameters={"mode":"high_quality","requested_outputs":["wdu_json"]}'
```

## Verification Checklist

### ✅ Architecture Detection
- [ ] System correctly identifies s390x architecture
- [ ] ZDLC availability is properly detected
- [ ] Fallback to PyTorch works when ZDLC unavailable

### ✅ Model Loading
- [ ] Layout model loads with correct backend
- [ ] Document figure classifier loads with correct backend
- [ ] ZDLC paths are passed correctly from config
- [ ] Models initialize without errors

### ✅ Inference
- [ ] Layout prediction produces results
- [ ] Figure classification produces results
- [ ] Results are consistent across backends
- [ ] Performance is acceptable

### ✅ Configuration
- [ ] `zdlc_model_path` is set for each model
- [ ] Paths point to correct `.so` files on s390x
- [ ] Artifact paths are correct
- [ ] Config is loaded properly by service

## Troubleshooting

### Issue: "zdlc_model_path is required when running on s390x"

**Cause:** Running on s390x with ZDLC available but path not provided.

**Solution:** Set the path in config:
```python
config.models.layout_model.zdlc_model_path = "/path/to/model.so"
```

### Issue: "ZDLC not available, falling back to PyTorch"

**Cause:** Running on s390x but `zdlc_pyrt` not installed.

**Solution:** Install ZDLC runtime or accept PyTorch fallback.

### Issue: Models load but predictions fail

**Cause:** Mismatch between artifacts and ZDLC model versions.

**Solution:** Ensure artifacts and compiled models are from same version.

### Issue: Wrong backend selected

**Cause:** Architecture detection issue.

**Solution:** Check `platform.machine()` output and verify it returns 's390x'.

## Expected Log Output

### On s390x with ZDLC:
```
INFO - Running on s390x architecture - ZDLC backend will be used
INFO - LayoutPredictor initialized with ZDLC backend
INFO - DocumentFigureClassifierPredictor initialized with ZDLC backend
```

### On other systems:
```
INFO - Running on x86_64 architecture - PyTorch backend will be used
INFO - LayoutPredictor initialized with PyTorch backend
INFO - DocumentFigureClassifierPredictor initialized with PyTorch backend
```

## Performance Benchmarking

To compare ZDLC vs PyTorch performance:

```python
import time
from PIL import Image

# Load test image
image = Image.open("test.png")

# Time prediction
start = time.time()
predictions = list(predictor.predict(image))
elapsed = time.time() - start

print(f"Backend: {predictor.info()['backend']}")
print(f"Time: {elapsed:.3f}s")
print(f"Elements: {len(predictions)}")
```

## Continuous Integration

For CI/CD pipelines:

```bash
#!/bin/bash
# Run tests based on architecture

if [ "$(uname -m)" = "s390x" ]; then
    echo "Running s390x tests with ZDLC"
    python test_e2e_zdlc_flow.py
else
    echo "Running x86_64 tests with PyTorch"
    python test_architecture_detection.py
fi
```

## Summary

The ZDLC integration is **fully automatic**:

1. ✅ **Auto-detects** s390x architecture at module import
2. ✅ **Auto-selects** ZDLC backend when available
3. ✅ **Auto-falls back** to PyTorch when ZDLC unavailable
4. ✅ **Requires only** correct `zdlc_model_path` in configuration

No code changes needed - just provide the paths and the system handles the rest!