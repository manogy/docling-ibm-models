#!/usr/bin/env python3
"""
Test ONNX model directly before ZDLC compilation.
This verifies the ONNX model works correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnxruntime as ort
from PIL import Image

print("=" * 70)
print("TESTING ONNX MODEL DIRECTLY")
print("=" * 70)

# Path to ONNX model
onnx_model_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx"

print(f"\nONNX Model: {onnx_model_path}")

# Load ONNX model
try:
    session = ort.InferenceSession(onnx_model_path)
    print("✓ ONNX model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load ONNX model: {e}")
    sys.exit(1)

# Get model info
print("\nModel Information:")
print(f"  Inputs:")
for inp in session.get_inputs():
    print(f"    - {inp.name}: {inp.shape} ({inp.type})")
print(f"  Outputs:")
for out in session.get_outputs():
    print(f"    - {out.name}: {out.shape} ({out.type})")

# Load test image
test_image_path = "tests/test_data/figure_classifier/images/bar_chart.jpg"
print(f"\nTest Image: {test_image_path}")

try:
    img = Image.open(test_image_path).convert("RGB")
    print(f"✓ Image loaded: {img.size}")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    sys.exit(1)

# Prepare input - try different preprocessing approaches
print("\n" + "=" * 70)
print("TESTING DIFFERENT INPUT FORMATS")
print("=" * 70)

def test_preprocessing(name, img_array, session):
    """Test a specific preprocessing approach"""
    print(f"\n{name}:")
    print(f"  Shape: {img_array.shape}")
    print(f"  Dtype: {img_array.dtype}")
    print(f"  Range: [{img_array.min():.3f}, {img_array.max():.3f}]")
    print(f"  Mean: {img_array.mean():.3f}")
    print(f"  Std: {img_array.std():.3f}")
    
    try:
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: img_array})
        logits = outputs[0]
        
        print(f"  Output shape: {logits.shape}")
        print(f"  Output range: [{logits.min():.3f}, {logits.max():.3f}]")
        
        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Get top prediction
        top_idx = np.argmax(probs[0])
        top_prob = probs[0][top_idx]
        
        print(f"  Top class index: {top_idx}")
        print(f"  Top probability: {top_prob:.4f} ({top_prob*100:.2f}%)")
        
        # Check if predictions look reasonable
        if top_prob > 0.5:
            print(f"  ✓ Good confidence!")
            return True
        elif top_prob > 0.1:
            print(f"  ⚠ Low confidence")
            return False
        else:
            print(f"  ✗ Very low confidence - likely wrong preprocessing")
            return False
            
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

# Test 1: Raw pixels [0-255], HWC format
img_resized = img.resize((224, 224))
img_array_1 = np.array(img_resized).astype(np.float32)
img_array_1 = np.expand_dims(img_array_1, 0)  # Add batch dim
test_preprocessing("Test 1: Raw pixels [0-255], NHWC", img_array_1, session)

# Test 2: Normalized [0-1], HWC format
img_array_2 = img_array_1 / 255.0
test_preprocessing("Test 2: Normalized [0-1], NHWC", img_array_2, session)

# Test 3: Raw pixels [0-255], CHW format
img_array_3 = np.array(img_resized).astype(np.float32)
img_array_3 = img_array_3.transpose(2, 0, 1)  # HWC -> CHW
img_array_3 = np.expand_dims(img_array_3, 0)  # Add batch dim
test_preprocessing("Test 3: Raw pixels [0-255], NCHW", img_array_3, session)

# Test 4: Normalized [0-1], CHW format
img_array_4 = img_array_3 / 255.0
test_preprocessing("Test 4: Normalized [0-1], NCHW", img_array_4, session)

# Test 5: ImageNet normalized, CHW format
img_array_5 = img_array_4.copy()
mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
std = np.array([0.47853944, 0.4732864, 0.47434163]).reshape(1, 3, 1, 1)
img_array_5 = (img_array_5 - mean) / std
result = test_preprocessing(
    "Test 5: ImageNet normalized, NCHW", 
    img_array_5, 
    session
)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if result:
    print("\n✓ Found working preprocessing!")
    print("\nNext steps:")
    print("1. Use this preprocessing in the ZDLC predictor")
    print("2. Recompile ONNX to .so with ZDLC")
    print("3. Test with ZDLC predictor")
else:
    print("\n⚠ No preprocessing worked well")
    print("\nPossible issues:")
    print("1. ONNX model may be incomplete")
    print("2. Model may need different input format")
    print("3. Check model.onnx file size - it should be large (>10MB)")
    
    import os
    if os.path.exists(onnx_model_path):
        size_mb = os.path.getsize(onnx_model_path) / (1024 * 1024)
        print(f"\nONNX file size: {size_mb:.2f} MB")
        if size_mb < 1:
            print("✗ File is too small - model weights may be missing!")
            print("   The model.onnx should contain the full model weights")

print("\n" + "=" * 70)

# Made with Bob
