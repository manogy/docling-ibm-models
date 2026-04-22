#!/bin/bash
# Run unit tests for docling-ibm-models with ZDLC support
# This script runs the existing unit tests with proper configuration

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate virtual environment if it exists
VENV_DIR="$SCRIPT_DIR/venv_e2e_test"
if [ -d "$VENV_DIR" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "⚠️  Virtual environment not found at $VENV_DIR"
    echo "Please run ./setup_e2e_test_env.sh first"
    exit 1
fi

# Set model paths for tests
export LAYOUT_ARTIFACTS_PATH="$PARENT_DIR/docling-layout-heron"
export LAYOUT_ZDLC_PATH="$PARENT_DIR/docling-layout-heron/docling-layout-heron-NNPA.so"
export CLASSIFIER_ARTIFACTS_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0"
export CLASSIFIER_ZDLC_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"

echo "=========================================="
echo "Running docling-ibm-models Unit Tests"
echo "=========================================="
echo ""
echo "Model Configuration:"
echo "  Layout artifacts: $LAYOUT_ARTIFACTS_PATH"
echo "  Layout ZDLC: $LAYOUT_ZDLC_PATH"
echo "  Classifier artifacts: $CLASSIFIER_ARTIFACTS_PATH"
echo "  Classifier ZDLC: $CLASSIFIER_ZDLC_PATH"
echo ""

# Change to project directory
cd "$SCRIPT_DIR"

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Installing pytest..."
    pip install pytest pytest-cov --only-binary :all:
fi

# Run tests
echo "Running unit tests..."
echo ""

# Option 1: Run all tests
if [ "$1" == "all" ]; then
    echo "Running ALL unit tests..."
    pytest tests/ -v --tb=short
    
# Option 2: Run specific test
elif [ -n "$1" ]; then
    echo "Running specific test: $1"
    pytest "tests/$1" -v --tb=short
    
# Option 3: Run layout and classifier tests (most relevant for ZDLC)
else
    echo "Running Layout and Classifier tests (ZDLC-relevant)..."
    pytest tests/test_layout_predictor.py tests/test_document_figure_classifier.py -v --tb=short
fi

echo ""
echo "=========================================="
echo "Unit Tests Complete!"
echo "=========================================="
echo ""
echo "To run specific tests:"
echo "  ./run_unit_tests.sh test_layout_predictor.py"
echo "  ./run_unit_tests.sh test_document_figure_classifier.py"
echo ""
echo "To run all tests:"
echo "  ./run_unit_tests.sh all"
echo ""

# Made with Bob
