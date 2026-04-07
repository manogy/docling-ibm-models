#!/usr/bin/env python3
"""
Diagnostic script to identify why CPU model gives uniform probabilities.
"""

import sys
from pathlib import Path

# Add parent directory to Python path
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

import numpy as np
import torchvision.transforms as transforms
import zdlc_pyrt
from PIL import Image

# Hardcoded paths for IBM Z
CPU_MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so"
NNPA_MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
TEST_IMAGE = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images/bar_chart.jpg"

print("="*70)
print("CPU vs NNPA Model Diagnostic")
print("="*70)

# Load image and preprocess WITH normalization
image = Image.open(TEST_IMAGE).convert("RGB")
print(f"\nImage: {TEST_IMAGE}")
print(f"Size: {image.size}")

processor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])

processed = processor(image)
numpy_image = processed.numpy().astype(np.float32)
batch = np.expand_dims(numpy_image, axis=0)

print(f"\nInput shape: {batch.shape}")
print(f"Input range: [{batch.min():.3f}, {batch.max():.3f}]")
print(f"Input mean: {batch.mean():.3f}")
print(f"Input std: {batch.std():.3f}")

# Test CPU model
print(f"\n{'='*70}")
print("Testing CPU Model")
print(f"{'='*70}")
print(f"Model: {CPU_MODEL_PATH}")

cpu_session = zdlc_pyrt.InferenceSession(CPU_MODEL_PATH)
cpu_outputs = cpu_session.run([batch])
cpu_logits = cpu_outputs[0][0]

print(f"\nOutput shape: {cpu_outputs[0].shape}")
print(f"Output range: [{cpu_logits.min():.3f}, {cpu_logits.max():.3f}]")
print(f"Output mean: {cpu_logits.mean():.3f}")
print(f"Output std: {cpu_logits.std():.3f}")
print(f"Output sum: {cpu_logits.sum():.3f}")

# Check if outputs are all similar (uniform)
output_variance = np.var(cpu_logits)
print(f"Output variance: {output_variance:.6f}")

if output_variance < 0.01:
    print("⚠️  WARNING: Very low variance - outputs are nearly uniform!")
    print("   This suggests the model is not working correctly.")

# Apply softmax
exp_logits = np.exp(cpu_logits - np.max(cpu_logits))
cpu_probs = exp_logits / np.sum(exp_logits)

print(f"\nTop 5 probabilities:")
top_5_idx = np.argsort(cpu_probs)[::-1][:5]
for i, idx in enumerate(top_5_idx):
    print(f"  {i+1}. Index {idx}: {cpu_probs[idx]:.6f} ({cpu_probs[idx]*100:.2f}%)")

# Test NNPA model for comparison
print(f"\n{'='*70}")
print("Testing NNPA Model (for comparison)")
print(f"{'='*70}")
print(f"Model: {NNPA_MODEL_PATH}")

nnpa_session = zdlc_pyrt.InferenceSession(NNPA_MODEL_PATH)
nnpa_outputs = nnpa_session.run([batch])
nnpa_logits = nnpa_outputs[0][0]

print(f"\nOutput shape: {nnpa_outputs[0].shape}")
print(f"Output range: [{nnpa_logits.min():.3f}, {nnpa_logits.max():.3f}]")
print(f"Output mean: {nnpa_logits.mean():.3f}")
print(f"Output std: {nnpa_logits.std():.3f}")
print(f"Output sum: {nnpa_logits.sum():.3f}")

output_variance = np.var(nnpa_logits)
print(f"Output variance: {output_variance:.6f}")

# Apply softmax
exp_logits = np.exp(nnpa_logits - np.max(nnpa_logits))
nnpa_probs = exp_logits / np.sum(exp_logits)

print(f"\nTop 5 probabilities:")
top_5_idx = np.argsort(nnpa_probs)[::-1][:5]
for i, idx in enumerate(top_5_idx):
    print(f"  {i+1}. Index {idx}: {nnpa_probs[idx]:.6f} ({nnpa_probs[idx]*100:.2f}%)")

# Compare outputs
print(f"\n{'='*70}")
print("Comparison")
print(f"{'='*70}")

logits_diff = np.abs(cpu_logits - nnpa_logits)
print(f"\nLogits difference:")
print(f"  Max diff: {logits_diff.max():.6f}")
print(f"  Mean diff: {logits_diff.mean():.6f}")
print(f"  Std diff: {logits_diff.std():.6f}")

if logits_diff.max() > 1.0:
    print("\n❌ CPU and NNPA models produce DIFFERENT outputs!")
    print("   The CPU model is likely compiled incorrectly.")
    print("\n   Possible issues:")
    print("   1. CPU model compiled with wrong settings")
    print("   2. CPU backend missing operations")
    print("   3. Quantization issue in CPU model")
    print("   4. CPU model is from different source/version")
else:
    print("\n✅ CPU and NNPA models produce SIMILAR outputs")
    print("   The issue might be in post-processing")

print(f"\n{'='*70}")

# Made with Bob
