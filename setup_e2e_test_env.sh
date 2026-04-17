#!/bin/bash
# Setup script for E2E ZDLC testing environment
# This script installs wdu and watson_doc_understanding from local branches

set -e  # Exit on error

echo "=========================================="
echo "E2E ZDLC Test Environment Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Define paths to the three repositories
DOCLING_IBM_MODELS_DIR="$SCRIPT_DIR"
WDU_DIR="$(dirname "$SCRIPT_DIR")/wdu"
WATSON_DOC_DIR="$(dirname "$SCRIPT_DIR")/watson_doc_understanding"

echo "Repository paths:"
echo "  docling-ibm-models: $DOCLING_IBM_MODELS_DIR"
echo "  wdu: $WDU_DIR"
echo "  watson_doc_understanding: $WATSON_DOC_DIR"
echo ""

# Check if directories exist
if [ ! -d "$WDU_DIR" ]; then
    echo "❌ Error: wdu directory not found at $WDU_DIR"
    exit 1
fi

if [ ! -d "$WATSON_DOC_DIR" ]; then
    echo "❌ Error: watson_doc_understanding directory not found at $WATSON_DOC_DIR"
    exit 1
fi

echo "✅ All repository directories found"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if Python 3.11 or 3.12
if [[ ! "$PYTHON_VERSION" =~ ^3\.(11|12) ]]; then
    echo "⚠️  Warning: Python 3.11 or 3.12 recommended, you have $PYTHON_VERSION"
fi
echo ""

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/venv_e2e_test"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtual environment created at $VENV_DIR"
else
    echo "✅ Virtual environment already exists at $VENV_DIR"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
echo "✅ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo ""

# Install docling-ibm-models (current directory)
echo "=========================================="
echo "Installing docling-ibm-models (Layer 1)"
echo "=========================================="
cd "$DOCLING_IBM_MODELS_DIR"
echo "Current branch: $(git branch --show-current)"
pip install -e .
echo "✅ docling-ibm-models installed"
echo ""

# Install wdu
echo "=========================================="
echo "Installing wdu (Layer 2)"
echo "=========================================="
cd "$WDU_DIR"
echo "Current branch: $(git branch --show-current)"
pip install -e .
echo "✅ wdu installed"
echo ""

# Install watson_doc_understanding
echo "=========================================="
echo "Installing watson_doc_understanding (Layer 3)"
echo "=========================================="
cd "$WATSON_DOC_DIR"
echo "Current branch: $(git branch --show-current)"
pip install -e .
echo "✅ watson_doc_understanding installed"
echo ""

# Install test dependencies
echo "=========================================="
echo "Installing test dependencies"
echo "=========================================="
cd "$DOCLING_IBM_MODELS_DIR"
pip install pytest requests
echo "✅ Test dependencies installed"
echo ""

# Verify installations
echo "=========================================="
echo "Verifying installations"
echo "=========================================="
echo ""

echo "Checking docling-ibm-models..."
python3 -c "from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor; print('✅ docling-ibm-models import successful')" || echo "❌ docling-ibm-models import failed"

echo "Checking wdu..."
python3 -c "from wdu.models.models_provider import ModelsProvider; print('✅ wdu import successful')" || echo "❌ wdu import failed"

echo "Checking watson_doc_understanding..."
python3 -c "import watson_doc_understanding; print('✅ watson_doc_understanding import successful')" || echo "❌ watson_doc_understanding import failed"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the E2E test:"
echo "  cd $DOCLING_IBM_MODELS_DIR"
echo "  python test_e2e_zdlc_flow.py"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""

# Made with Bob
