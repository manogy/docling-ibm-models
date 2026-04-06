#!/usr/bin/env python3
"""Compare ONNX vs ZDLC outputs to debug the issue."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import onnxruntime as ort
import zdlc_pyrt
from PIL import Image

print("=" * 70)
print("COMPARING ONNX vs ZDLC OUTPUTS")
print("=" * 70)

# Paths
onnx_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx"
zdlc_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.so"
test_image = "tests/test_data/figure_classifier/images/bar_chart.jpg"

# Load image and preprocess (Test 4 format that worked)
img = Image.open(test_image).convert("RGB")
img = img.resize((224, 224))
img_array = np.array(img).astype(np.float32)
img_array = img_array / 255.0  # Normalize to [0-1]
img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
img_array = np.expand_dims(img_array, 0)  # Add batch dim

print(f"\nInput shape: {img_array.shape}")
print(f"Input range: [{img_array.min():.3f}, {img_array.max():.3f}]")

# Test ONNX
print("\n" + "=" * 70)
print("ONNX INFERENCE")
print("=" * 70)
onnx_session = ort.InferenceSession(onnx_path)
onnx_outputs = onnx_session.run(None, {onnx_session.get_inputs()[0].name: img_array})
onnx_logits = onnx_outputs[0]

print(f"Output shape: {onnx_logits.shape}")
print(f"Output range: [{onnx_logits.min():.3f}, {onnx_logits.max():.3f}]")

# Apply softmax
exp_logits = np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True))
onnx_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
onnx_top_idx = np.argmax(onnx_probs[0])
onnx_top_prob = onnx_probs[0][onnx_top_idx]

print(f"Top class: {onnx_top_idx}, Confidence: {onnx_top_prob*100:.2f}%")

# Test ZDLC
print("\n" + "=" * 70)
print("ZDLC INFERENCE")
print("=" * 70)
zdlc_session = zdlc_pyrt.InferenceSession(zdlc_path)
zdlc_outputs = zdlc_session.run([img_array])
zdlc_logits = zdlc_outputs[0]

print(f"Output shape: {zdlc_logits.shape}")
print(f"Output range: [{zdlc_logits.min():.3f}, {zdlc_logits.max():.3f}]")

# Apply softmax
exp_logits = np.exp(zdlc_logits - np.max(zdlc_logits, axis=1, keepdims=True))
zdlc_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
zdlc_top_idx = np.argmax(zdlc_probs[0])
zdlc_top_prob = zdlc_probs[0][zdlc_top_idx]

print(f"Top class: {zdlc_top_idx}, Confidence: {zdlc_top_prob*100:.2f}%")

# Compare
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)

print(f"\nONNX top class: {onnx_top_idx} ({onnx_top_prob*100:.2f}%)")
print(f"ZDLC top class: {zdlc_top_idx} ({zdlc_top_prob*100:.2f}%)")

# Check if outputs match
max_diff = np.abs(onnx_logits - zdlc_logits).max()
print(f"\nMax difference in logits: {max_diff:.6f}")

if max_diff < 0.01:
    print("✓ Outputs match! ZDLC compilation is correct.")
else:
    print("✗ Outputs differ! ZDLC compilation may have issues.")
    print("\nFirst 5 logits comparison:")
    print("ONNX:", onnx_logits[0][:5])
    print("ZDLC:", zdlc_logits[0][:5])

# Made with Bob
