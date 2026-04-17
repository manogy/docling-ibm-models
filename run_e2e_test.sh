#!/bin/bash
# Quick test runner for E2E ZDLC integration
# This script sets up environment variables and runs the test

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment
VENV_DIR="$SCRIPT_DIR/venv_e2e_test"
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found. Please run ./setup_e2e_test_env.sh first"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Set model paths based on your directory structure
export LAYOUT_ARTIFACTS_PATH="$PARENT_DIR/docling-layout-heron"
export LAYOUT_ZDLC_PATH="$PARENT_DIR/docling-layout-heron/docling-layout-heron-NNPA.so"
export CLASSIFIER_ARTIFACTS_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0"
export CLASSIFIER_ZDLC_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
export TEST_IMAGE_PATH="$SCRIPT_DIR/tests/test_data/samples/empty_iocr.png"

echo "=========================================="
echo "Running E2E ZDLC Integration Test"
echo "=========================================="
echo ""
echo "Model Configuration:"
echo "  Layout artifacts: $LAYOUT_ARTIFACTS_PATH"
echo "  Layout ZDLC: $LAYOUT_ZDLC_PATH"
echo "  Classifier artifacts: $CLASSIFIER_ARTIFACTS_PATH"
echo "  Classifier ZDLC: $CLASSIFIER_ZDLC_PATH"
echo "  Test image: $TEST_IMAGE_PATH"
echo ""

# Run the test
cd "$SCRIPT_DIR"
python test_e2e_zdlc_flow.py

echo ""
echo "Test complete!"

# Made with Bob
