#!/bin/bash
# Commands to run ZDLC Document Figure Classifier Demo
# Location: /root/manogya/manogya/docling-ibm-models/

echo "========================================================================"
echo "ZDLC Document Figure Classifier Demo - IBM Z"
echo "========================================================================"
echo ""

# Set paths based on your directory structure
DOCLING_DIR="/root/manogya/manogya/docling-ibm-models"
NNPA_MODEL="/root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so"
CPU_MODEL="/root/manogya/manogya/DocumentFigureClassifier-V2-CPU.so"
TEST_IMAGES="$DOCLING_DIR/tests/test_data/figure_classifier/images"

# Change to docling directory
cd "$DOCLING_DIR" || exit 1

echo "Current directory: $(pwd)"
echo "NNPA Model: $NNPA_MODEL"
echo "Test Images: $TEST_IMAGES"
echo ""

# Check if model exists
if [ ! -f "$NNPA_MODEL" ]; then
    echo "ERROR: NNPA model not found at $NNPA_MODEL"
    exit 1
fi

# Check if test images exist
if [ ! -d "$TEST_IMAGES" ]; then
    echo "ERROR: Test images directory not found at $TEST_IMAGES"
    exit 1
fi

echo "========================================================================"
echo "Running ZDLC Demo with NNPA Model..."
echo "========================================================================"
echo ""

# Run the demo
python -m demo.demo_document_figure_classifier_predictor_zdlc \
    -i "$TEST_IMAGES" \
    -z "$NNPA_MODEL" \
    -n 4 \
    -v viz_nnpa_results/

echo ""
echo "========================================================================"
echo "Demo Complete!"
echo "========================================================================"

# Made with Bob
