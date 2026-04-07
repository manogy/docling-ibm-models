# Final Commands to Run ZDLC Demo

## ✅ Modified Demo Script Ready!

The demo script has been updated to use your **local config files** instead of downloading from HuggingFace.

---

## 🚀 Command to Run (Using Local Config)

```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_nnpa_results/
```

### What Each Parameter Does:
- `-i` : Input images directory
- `-z` : Path to ZDLC compiled .so model (NNPA version)
- `-a` : Path to local artifacts (config.json, preprocessor_config.json) - **NEW!**
- `-n` : Number of threads (4)
- `-v` : Output visualization directory

---

## 📁 Your File Structure:

```
/root/manogya/manogya/
├── DocumentFigureClassifier-v2.0/
│   ├── config.json                              ✅ Model config
│   ├── preprocessor_config.json                 ✅ Preprocessing config
│   ├── DocumentFigureClassifier-V2-NNPA.so     ✅ NNPA model
│   ├── DocumentFigureClassifier-V2-CPU.so      ✅ CPU model
│   └── model.onnx                               ✅ Original ONNX
└── docling-ibm-models/
    ├── demo/
    │   └── demo_document_figure_classifier_predictor_zdlc.py  ✅ Modified script
    └── tests/test_data/figure_classifier/images/
        ├── bar_chart.jpg                        ✅ Test image
        └── map.jpg                              ✅ Test image
```

---

## 🎯 What Changed in the Demo Script:

### Before (Old):
```python
# Always downloaded from HuggingFace
download_path = snapshot_download(
    repo_id="ds4sd/DocumentFigureClassifier", revision="v1.0.0"
)
```

### After (New):
```python
# Use local artifacts if provided, otherwise download
if artifact_path and Path(artifact_path).exists():
    logger.info(f"Using local model artifacts from: {artifact_path}")
    download_path = artifact_path
else:
    logger.info("Downloading model artifacts from HuggingFace...")
    download_path = snapshot_download(...)
```

---

## 🔄 Alternative Commands:

### 1. Test with CPU Model:
```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4
```

### 2. Compare NNPA vs CPU Performance:
```bash
cd /root/manogya/manogya/docling-ibm-models

echo "Testing NNPA model..."
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_nnpa/

echo ""
echo "Testing CPU model..."
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_cpu/
```

### 3. Without Local Artifacts (Will Download from HuggingFace):
```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -n 4
```

---

## 📊 Expected Output:

```
2026-04-07 14:30:00 DocumentFigureClassifierPredictorZDLC INFO     Using local model artifacts from: /root/manogya/manogya/DocumentFigureClassifier-v2.0
2026-04-07 14:30:00 DocumentFigureClassifierPredictorZDLC INFO     Initializing ZDLC Document Figure Classifier...
2026-04-07 14:30:00 DocumentFigureClassifierPredictorZDLC INFO     Artifacts path: /root/manogya/manogya/DocumentFigureClassifier-v2.0
2026-04-07 14:30:00 DocumentFigureClassifierPredictorZDLC INFO     ZDLC model path: /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so
2026-04-07 14:30:01 DocumentFigureClassifierPredictorZDLC INFO     Model info: {'name': 'DocumentFigureClassifier', 'version': 'v2.0'}
2026-04-07 14:30:01 DocumentFigureClassifierPredictorZDLC INFO     Loading 2 images from tests/test_data/figure_classifier/images...
2026-04-07 14:30:01 DocumentFigureClassifierPredictorZDLC INFO     Running inference...
2026-04-07 14:30:02 DocumentFigureClassifierPredictorZDLC INFO     For 2 images(ms): [total|avg] = [450.2|225.1]

======================================================================
PREDICTION RESULTS
======================================================================

Image: 'bar_chart.jpg'
----------------------------------------------------------------------
  1. Chart-Bar                       - 0.9876 (98.76%)
  2. Chart-Line                      - 0.0089 (0.89%)
  3. Chart-Pie                       - 0.0023 (0.23%)
  4. Table                           - 0.0008 (0.08%)
  5. Natural-Image                   - 0.0004 (0.04%)

Image: 'map.jpg'
----------------------------------------------------------------------
  1. Map                             - 0.9654 (96.54%)
  2. Natural-Image                   - 0.0234 (2.34%)
  3. Chart-Other                     - 0.0089 (0.89%)
  4. Diagram                         - 0.0015 (0.15%)
  5. Table                           - 0.0008 (0.08%)

======================================================================
```

---

## ✅ Benefits of Using Local Config:

1. **No Internet Required** - Works offline
2. **Faster Startup** - No download time
3. **Version Control** - Use specific config versions
4. **Reproducibility** - Consistent results

---

## 🎉 Ready to Run!

Just copy and paste the command above. The script will:
1. ✅ Use your local config files (no download!)
2. ✅ Load the NNPA compiled model
3. ✅ Process test images
4. ✅ Show classification results

**No HuggingFace download needed!** 🚀