# Running ZDLC Document Figure Classifier Demo on IBM Z

## Your Current Setup

Based on your directory listing on the IBM Z system:
```bash
/root/manogya/manogya/
├── demo_document_figure_classifier_predictor_zdlc.py
├── docling-ibm-models/
├── DocumentFigureClassifier-v2.0/
├── DocumentFigureClassifier-V2-CPU.so          # CPU-only compiled model
├── DocumentFigureClassifier-V2-NNPA.so         # NNPA accelerated model ⚡
├── zdlc_pyrt/
└── omconfig_cpu.json
```

---

## Prerequisites Check

### 1. Verify Python Environment
```bash
# Check Python version (should be 3.11)
python --version

# Verify you're in the correct environment
which python
```

### 2. Verify ZDLC Installation
```bash
# Check if zdlc_pyrt is importable
python -c "import zdlc_pyrt; print('ZDLC version:', zdlc_pyrt.__version__)"
```

### 3. Check Test Images
```bash
# Navigate to your docling-ibm-models directory
cd /root/manogya/manogya/docling-ibm-models

# Check if test images exist
ls -la tests/test_data/figure_classifier/images/
```

Expected output:
```
bar_chart.jpg
map.jpg
```

---

## Running the Demo

### Option 1: Test with NNPA Accelerated Model (Recommended) ⚡

```bash
# Navigate to docling-ibm-models directory
cd /root/manogya/manogya/docling-ibm-models

# Run with NNPA model
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so \
    -n 4 \
    -v viz_nnpa/
```

### Option 2: Test with CPU-Only Model (Fallback)

```bash
# Run with CPU model
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so \
    -n 4 \
    -v viz_cpu/
```

### Option 3: Compare NNPA vs CPU Performance

```bash
# Test NNPA
echo "Testing NNPA model..."
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so \
    -n 4 \
    -v viz_nnpa/

echo ""
echo "Testing CPU model..."
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so \
    -n 4 \
    -v viz_cpu/
```

---

## Command Parameters Explained

| Parameter | Description | Example |
|-----------|-------------|---------|
| `-i` / `--image_dir` | Directory with test images | `tests/test_data/figure_classifier/images` |
| `-z` / `--zdlc_model_path` | Path to compiled .so file | `/root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so` |
| `-n` / `--num_threads` | Number of inference threads | `4` (default) |
| `-v` / `--viz_dir` | Output directory for results | `viz_nnpa/` |

---

## Expected Output

### Successful Run Example:
```
2026-04-07 14:15:00 DocumentFigureClassifierPredictorZDLC INFO     Downloading model artifacts from HuggingFace...
2026-04-07 14:15:05 DocumentFigureClassifierPredictorZDLC INFO     Downloaded to: /root/.cache/huggingface/hub/...
2026-04-07 14:15:05 DocumentFigureClassifierPredictorZDLC INFO     Initializing ZDLC Document Figure Classifier...
2026-04-07 14:15:05 DocumentFigureClassifierPredictorZDLC INFO     Artifacts path: /root/.cache/huggingface/hub/...
2026-04-07 14:15:05 DocumentFigureClassifierPredictorZDLC INFO     ZDLC model path: /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so
2026-04-07 14:15:06 DocumentFigureClassifierPredictorZDLC INFO     Model info: {'name': 'DocumentFigureClassifier', 'version': 'v2.0'}
2026-04-07 14:15:06 DocumentFigureClassifierPredictorZDLC INFO     Loading 2 images from tests/test_data/figure_classifier/images...
2026-04-07 14:15:06 DocumentFigureClassifierPredictorZDLC INFO     Running inference...
2026-04-07 14:15:07 DocumentFigureClassifierPredictorZDLC INFO     For 2 images(ms): [total|avg] = [450.2|225.1]

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

## Troubleshooting

### Issue 1: Module Not Found Error
```bash
# Error: ModuleNotFoundError: No module named 'docling_ibm_models'

# Solution: Install the package in development mode
cd /root/manogya/manogya/docling-ibm-models
pip install -e .
```

### Issue 2: ZDLC Import Error
```bash
# Error: ModuleNotFoundError: No module named 'zdlc_pyrt'

# Solution: Install zdlc_pyrt
cd /root/manogya/manogya/zdlc_pyrt
pip install -e .
```

### Issue 3: HuggingFace Download Fails
```bash
# Error: Cannot download from HuggingFace

# Solution: Set HF token or use offline mode
export HF_TOKEN="your_token_here"

# Or download manually:
huggingface-cli download ds4sd/DocumentFigureClassifier --revision v1.0.0
```

### Issue 4: NNPA Model Fails
```bash
# Error: NNPA acceleration not available

# Solution: Fall back to CPU model
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so \
    -n 4
```

### Issue 5: Permission Denied
```bash
# Error: Permission denied on .so file

# Solution: Check file permissions
chmod +x /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so
```

---

## Testing with Your Own Images

### 1. Create a Test Directory
```bash
mkdir -p /root/manogya/manogya/my_test_images
```

### 2. Copy Your Images
```bash
# Copy images (PNG, JPG, JPEG supported)
cp /path/to/your/images/*.jpg /root/manogya/manogya/my_test_images/
```

### 3. Run Inference
```bash
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i /root/manogya/manogya/my_test_images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so \
    -n 4 \
    -v my_results/
```

---

## Performance Benchmarking

### Benchmark Script
```bash
#!/bin/bash
# benchmark_zdlc.sh

echo "=== ZDLC Document Figure Classifier Benchmark ==="
echo ""

# Test with different thread counts
for threads in 1 2 4 8; do
    echo "Testing with $threads threads (NNPA)..."
    time python -m demo.demo_document_figure_classifier_predictor_zdlc \
        -i tests/test_data/figure_classifier/images \
        -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so \
        -n $threads \
        -v viz_nnpa_${threads}/ 2>&1 | grep "For.*images"
    echo ""
done

echo "Testing CPU model for comparison..."
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so \
    -n 4 \
    -v viz_cpu/ 2>&1 | grep "For.*images"
```

### Run Benchmark
```bash
chmod +x benchmark_zdlc.sh
./benchmark_zdlc.sh
```

---

## Quick Start Commands (Copy-Paste Ready)

### Minimal Test (NNPA):
```bash
cd /root/manogya/manogya/docling-ibm-models && \
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so
```

### Minimal Test (CPU):
```bash
cd /root/manogya/manogya/docling-ibm-models && \
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so
```

### Full Test with Timing:
```bash
cd /root/manogya/manogya/docling-ibm-models && \
time python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i tests/test_data/figure_classifier/images \
    -z /root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so \
    -n 4 \
    -v viz_results/
```

---

## Expected Classification Categories

The model can classify figures into these categories:
- **Chart-Bar**: Bar charts
- **Chart-Line**: Line charts  
- **Chart-Pie**: Pie charts
- **Chart-Other**: Other chart types
- **Map**: Geographic maps
- **Natural-Image**: Photographs
- **Diagram**: Technical diagrams
- **Table**: Tables (though primarily handled by TableFormer)
- **Flowchart**: Process flowcharts
- **Screenshot**: UI screenshots

---

## Next Steps

After successful testing:

1. **Integrate into WDU Pipeline**
   - Update WDU config to use ZDLC models
   - Test with full document processing

2. **Performance Optimization**
   - Tune thread count for your workload
   - Monitor NNPA utilization
   - Compare NNPA vs CPU performance

3. **Production Deployment**
   - Set up model versioning
   - Configure monitoring
   - Implement fallback to CPU if NNPA fails

---

## Support Files

- **Demo Script**: `/root/manogya/manogya/docling-ibm-models/demo/demo_document_figure_classifier_predictor_zdlc.py`
- **ZDLC Predictor**: `/root/manogya/manogya/docling-ibm-models/docling_ibm_models/document_figure_classifier_model/document_figure_classifier_predictor_zdlc.py`
- **Test Images**: `/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images/`

---

## Questions?

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Ensure file paths are correct
4. Check system logs for NNPA-related errors