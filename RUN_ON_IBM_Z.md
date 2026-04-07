# Running ZDLC Demo on IBM Z System

## 📋 Quick Start

### Option 1: Use the IBM Z Specific Script (Easiest)

The script [`demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py`](demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py:1) has hardcoded paths for your IBM Z system.

**Copy to IBM Z:**
```bash
scp demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py \
    root@b39-cp4d4:/root/manogya/manogya/docling-ibm-models/demo/
```

**Run with defaults (NNPA model, test images):**
```bash
cd /root/manogya/manogya/docling-ibm-models
python demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py
```

**Run with CPU model:**
```bash
python demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so
```

**Run with more threads:**
```bash
python demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py -n 8
```

### Option 2: Use the Generic Script

**Run with explicit paths:**
```bash
cd /root/manogya/manogya/docling-ibm-models

python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -a /root/manogya/manogya/DocumentFigureClassifier-v2.0 \
    -n 4 \
    -v viz_nnpa_results/
```

## 🎯 Hardcoded Paths in IBM Z Script

The IBM Z specific script uses these paths by default:

```python
IBM_Z_BASE_PATH = "/root/manogya/manogya"
IBM_Z_MODEL_DIR = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
IBM_Z_NNPA_MODEL = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
IBM_Z_CPU_MODEL = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so"
IBM_Z_ARTIFACTS = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
IBM_Z_TEST_IMAGES = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images"
```

## ✅ Expected Output

```
======================================================================
ZDLC Document Figure Classifier - IBM Z Demo
======================================================================
✅ ZDLC model: /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so
✅ Artifacts: /root/manogya/manogya/DocumentFigureClassifier-v2.0
✅ Images: /root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images
✅ Threads: 4
======================================================================

Initializing ZDLC Document Figure Classifier...
✅ Predictor initialized successfully

Model info:
  Backend: ZDLC
  Threads: 4
  Classes: 16

Loading 2 images from /root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images...
✅ Loaded 2 images

Running inference...
✅ Inference completed
Performance: 2 images in XXX.Xms (avg: XX.Xms per image)

======================================================================
PREDICTION RESULTS
======================================================================

Image: 'bar_chart.jpg'
----------------------------------------------------------------------
✅ 1. bar_chart                      - 0.XXXX (XX.XX%)
   2. line_chart                     - 0.XXXX (XX.XX%)
   3. pie_chart                      - 0.XXXX (XX.XX%)
   4. other                          - 0.XXXX (XX.XX%)
   5. flow_chart                     - 0.XXXX (XX.XX%)

Image: 'map.jpg'
----------------------------------------------------------------------
✅ 1. map                            - 0.XXXX (XX.XX%)
   2. remote_sensing                 - 0.XXXX (XX.XX%)
   3. other                          - 0.XXXX (XX.XX%)
   4. screenshot                     - 0.XXXX (XX.XX%)
   5. logo                           - 0.XXXX (XX.XX%)

======================================================================
✅ Results saved to: viz_nnpa_results/predictions_summary.txt
```

## 🔧 Your IBM Z System Structure

```
/root/manogya/manogya/
├── DocumentFigureClassifier-v2.0/
│   ├── config.json                              ← Required
│   ├── preprocessor_config.json                 ← Required
│   ├── DocumentFigureClassifier-V2-NNPA.so     ← NNPA model
│   ├── DocumentFigureClassifier-V2-CPU.so      ← CPU model
│   └── model.onnx                               ← Original ONNX
└── docling-ibm-models/
    ├── demo/
    │   └── demo_document_figure_classifier_predictor_zdlc_ibmz.py
    ├── docling_ibm_models/
    │   └── document_figure_classifier_model/
    │       └── document_figure_classifier_predictor_zdlc.py
    └── tests/test_data/figure_classifier/images/
        ├── bar_chart.jpg
        └── map.jpg
```

## 🐛 Troubleshooting

### Error: "ZDLC model not found"
**Check:**
```bash
ls -lh /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so
```

### Error: "config.json not found"
**Check:**
```bash
ls -lh /root/manogya/manogya/DocumentFigureClassifier-v2.0/config.json
```

### Error: "Image directory not found"
**Check:**
```bash
ls -lh /root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images/
```

### Wrong predictions (e.g., bar_chart → crossword_puzzle)
**Cause:** Missing ImageNet normalization

**Fix:** Ensure you have the latest version of `document_figure_classifier_predictor_zdlc.py` with normalization:
```bash
cd /root/manogya/manogya/docling-ibm-models
git pull origin main
```

## 📊 Performance Comparison

Run both NNPA and CPU models to compare:

```bash
# NNPA model
python demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so \
    -v viz_nnpa_results/

# CPU model
python demo/demo_document_figure_classifier_predictor_zdlc_ibmz.py \
    -z /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so \
    -v viz_cpu_results/
```

Compare the performance metrics in the output.

---

**Ready to run on your IBM Z system!** 🚀