#!/bin/bash
# Setup script for E2E ZDLC testing environment
# This script clones/uses wdu and watson_doc_understanding repos and configures them

set -e  # Exit on error

echo "=========================================="
echo "E2E ZDLC Test Environment Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Current directory: $SCRIPT_DIR"
echo "Parent directory: $PARENT_DIR"
echo ""

# Define paths
DOCLING_IBM_MODELS_DIR="$SCRIPT_DIR"
WDU_DIR="$PARENT_DIR/wdu"
WATSON_DOC_DIR="$PARENT_DIR/watson_doc_understanding"

# Model paths (update these to match your setup)
LAYOUT_ARTIFACTS_PATH="$PARENT_DIR/docling-layout-heron"
LAYOUT_ZDLC_PATH="$PARENT_DIR/docling-layout-heron/docling-layout-heron-NNPA.so"
CLASSIFIER_ARTIFACTS_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0"
CLASSIFIER_ZDLC_PATH="$PARENT_DIR/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"

echo "Repository paths:"
echo "  docling-ibm-models: $DOCLING_IBM_MODELS_DIR"
echo "  wdu: $WDU_DIR"
echo "  watson_doc_understanding: $WATSON_DOC_DIR"
echo ""
echo "Model paths:"
echo "  Layout artifacts: $LAYOUT_ARTIFACTS_PATH"
echo "  Layout ZDLC: $LAYOUT_ZDLC_PATH"
echo "  Classifier artifacts: $CLASSIFIER_ARTIFACTS_PATH"
echo "  Classifier ZDLC: $CLASSIFIER_ZDLC_PATH"
echo ""

# Clone wdu if it doesn't exist
if [ ! -d "$WDU_DIR" ]; then
    echo "Cloning wdu repository..."
    cd "$PARENT_DIR"
    git clone https://github.ibm.com/ai-foundation/wdu.git
    echo "✅ wdu cloned"
else
    echo "✅ wdu directory already exists"
fi

# Clone watson_doc_understanding if it doesn't exist
if [ ! -d "$WATSON_DOC_DIR" ]; then
    echo "Cloning watson_doc_understanding repository..."
    cd "$PARENT_DIR"
    git clone https://github.ibm.com/ai-foundation/watson_doc_understanding.git
    echo "✅ watson_doc_understanding cloned"
else
    echo "✅ watson_doc_understanding directory already exists"
fi
echo ""

# Set Python path - use custom Python 3.11.2 if available, otherwise system python3
PYTHON_BIN="/root/python-3.11.2/bin/python3.11"

if [ -f "$PYTHON_BIN" ]; then
    echo "Using custom Python: $PYTHON_BIN"
    PYTHON_CMD="$PYTHON_BIN"
else
    echo "Custom Python not found at $PYTHON_BIN"
    echo "Using system python3"
    PYTHON_CMD="python3"
fi

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

if [[ ! "$PYTHON_VERSION" =~ ^3\.(11|12) ]]; then
    echo "⚠️  Warning: Python 3.11 or 3.12 recommended, you have $PYTHON_VERSION"
fi
echo ""

# Create virtual environment if it doesn't exist
VENV_DIR="$SCRIPT_DIR/venv_e2e_test"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with $PYTHON_CMD..."
    $PYTHON_CMD -m venv "$VENV_DIR"
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
pip install --upgrade pip setuptools wheel --only-binary :all:
echo ""

# Configure pip to prefer binary packages
echo "Configuring pip to use only binary packages..."
pip config set global.only-binary ":all:"
echo "✅ Pip configured to use only binary packages"
echo ""

# Install docling-ibm-models (current directory)
echo "=========================================="
echo "Installing docling-ibm-models (Layer 1)"
echo "=========================================="
cd "$DOCLING_IBM_MODELS_DIR"
echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
pip install -e . --only-binary :all: || pip install -e .
echo "✅ docling-ibm-models installed"
echo ""

# Install wdu
echo "=========================================="
echo "Installing wdu (Layer 2)"
echo "=========================================="
cd "$WDU_DIR"
echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
pip install -e . --only-binary :all: || pip install -e .
echo "✅ wdu installed"
echo ""

# Create wdu config override file
echo "=========================================="
echo "Creating WDU configuration override"
echo "=========================================="
WDU_CONFIG_OVERRIDE="$PARENT_DIR/wdu_config_override.py"
cat > "$WDU_CONFIG_OVERRIDE" << EOF
"""
WDU Configuration Override for E2E Testing
This file overrides the model paths without modifying the wdu repository.
"""
from wdu.config.wdu_config import WduConfig

# Create config with custom model paths
config = WduConfig()

# Override layout model paths
config.models.layout_model.weights_path = "$LAYOUT_ARTIFACTS_PATH"
config.models.layout_model.zdlc_model_path = "$LAYOUT_ZDLC_PATH"

# Override document figure classifier paths
config.models.document_figure_classifier.weights_path = "$CLASSIFIER_ARTIFACTS_PATH"
config.models.document_figure_classifier.zdlc_model_path = "$CLASSIFIER_ZDLC_PATH"

print("✅ WDU config override loaded")
print(f"   Layout artifacts: {config.models.layout_model.weights_path}")
print(f"   Layout ZDLC: {config.models.layout_model.zdlc_model_path}")
print(f"   Classifier artifacts: {config.models.document_figure_classifier.weights_path}")
print(f"   Classifier ZDLC: {config.models.document_figure_classifier.zdlc_model_path}")
EOF

echo "✅ WDU config override created at: $WDU_CONFIG_OVERRIDE"
echo ""

# Install watson_doc_understanding
echo "=========================================="
echo "Installing watson_doc_understanding (Layer 3)"
echo "=========================================="
cd "$WATSON_DOC_DIR"
echo "Current branch: $(git branch --show-current 2>/dev/null || echo 'detached HEAD')"
pip install -e . --only-binary :all: || pip install -e .
echo "✅ watson_doc_understanding installed"
echo ""

# Install test dependencies
echo "=========================================="
echo "Installing test dependencies"
echo "=========================================="
cd "$DOCLING_IBM_MODELS_DIR"
pip install pytest requests --only-binary :all:
echo "✅ Test dependencies installed"
echo ""

# Verify model files exist
echo "=========================================="
echo "Verifying model files"
echo "=========================================="
echo ""

if [ -f "$LAYOUT_ARTIFACTS_PATH/config.json" ]; then
    echo "✅ Layout model artifacts found"
else
    echo "❌ Layout model artifacts NOT found at: $LAYOUT_ARTIFACTS_PATH"
fi

if [ -f "$LAYOUT_ZDLC_PATH" ]; then
    echo "✅ Layout ZDLC model found"
else
    echo "⚠️  Layout ZDLC model NOT found at: $LAYOUT_ZDLC_PATH"
    echo "   (This is OK if not running on s390x)"
fi

if [ -f "$CLASSIFIER_ARTIFACTS_PATH/config.json" ]; then
    echo "✅ Classifier model artifacts found"
else
    echo "❌ Classifier model artifacts NOT found at: $CLASSIFIER_ARTIFACTS_PATH"
fi

if [ -f "$CLASSIFIER_ZDLC_PATH" ]; then
    echo "✅ Classifier ZDLC model found"
else
    echo "⚠️  Classifier ZDLC model NOT found at: $CLASSIFIER_ZDLC_PATH"
    echo "   (This is OK if not running on s390x)"
fi
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
echo "Environment is ready for testing!"
echo ""
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run the E2E test with custom config:"
echo "  cd $DOCLING_IBM_MODELS_DIR"
echo "  export LAYOUT_ARTIFACTS_PATH='$LAYOUT_ARTIFACTS_PATH'"
echo "  export LAYOUT_ZDLC_PATH='$LAYOUT_ZDLC_PATH'"
echo "  export CLASSIFIER_ARTIFACTS_PATH='$CLASSIFIER_ARTIFACTS_PATH'"
echo "  export CLASSIFIER_ZDLC_PATH='$CLASSIFIER_ZDLC_PATH'"
echo "  python test_e2e_zdlc_flow.py"
echo ""
echo "Or use the quick test script:"
echo "  ./run_e2e_test.sh"
echo ""
echo "To deactivate the environment:"
echo "  deactivate"
echo ""

# Made with Bob
