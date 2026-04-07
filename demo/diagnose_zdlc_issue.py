#!/usr/bin/env python3
"""
Diagnostic script to identify why ZDLC predictions are wrong.

This script checks:
1. If normalization is being applied
2. What the raw model outputs look like
3. If class labels are in the correct order
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
from transformers import AutoConfig

# Hardcoded paths for IBM Z
MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
ARTIFACTS_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
TEST_IMAGE = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images/bar_chart.jpg"

print("="*70)
print("ZDLC Diagnostic Script")
print("="*70)

# Load config
print("\n1. Loading config...")
config = AutoConfig.from_pretrained(ARTIFACTS_PATH)
print(f"   Config loaded: {len(config.id2label)} classes")
print(f"   id2label mapping:")
for id, label in sorted(config.id2label.items()):
    print(f"      {id}: {label}")

# Create class list (sorted alphabetically like in predictor)
classes = list(config.id2label.values())
classes.sort()
print(f"\n   Alphabetically sorted classes:")
for i, cls in enumerate(classes):
    print(f"      {i}: {cls}")

# Load ZDLC model
print(f"\n2. Loading ZDLC model...")
print(f"   Model path: {MODEL_PATH}")
session = zdlc_pyrt.InferenceSession(MODEL_PATH)
print(f"   ✅ Model loaded")

# Load and preprocess image WITHOUT normalization
print(f"\n3. Testing WITHOUT normalization...")
image = Image.open(TEST_IMAGE).convert("RGB")
print(f"   Image: {TEST_IMAGE}")
print(f"   Size: {image.size}")

processor_no_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

processed_no_norm = processor_no_norm(image)
numpy_no_norm = processed_no_norm.numpy().astype(np.float32)
batch_no_norm = np.expand_dims(numpy_no_norm, axis=0)

print(f"   Input shape: {batch_no_norm.shape}")
print(f"   Input range: [{batch_no_norm.min():.3f}, {batch_no_norm.max():.3f}]")

outputs_no_norm = session.run([batch_no_norm])
logits_no_norm = outputs_no_norm[0][0]

print(f"   Output shape: {outputs_no_norm[0].shape}")
print(f"   Output range: [{logits_no_norm.min():.3f}, {logits_no_norm.max():.3f}]")
print(f"   Output sum: {np.sum(logits_no_norm):.3f}")

# Apply softmax
exp_logits = np.exp(logits_no_norm - np.max(logits_no_norm))
probs_no_norm = exp_logits / np.sum(exp_logits)

top_5_no_norm = np.argsort(probs_no_norm)[::-1][:5]
print(f"\n   Top 5 predictions WITHOUT normalization:")
for i, idx in enumerate(top_5_no_norm):
    print(f"      {i+1}. {classes[idx]:30s} - {probs_no_norm[idx]:.4f} ({probs_no_norm[idx]*100:.2f}%)")

# Load and preprocess image WITH normalization
print(f"\n4. Testing WITH normalization...")

processor_with_norm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.47853944, 0.4732864, 0.47434163],
    ),
])

processed_with_norm = processor_with_norm(image)
numpy_with_norm = processed_with_norm.numpy().astype(np.float32)
batch_with_norm = np.expand_dims(numpy_with_norm, axis=0)

print(f"   Input shape: {batch_with_norm.shape}")
print(f"   Input range: [{batch_with_norm.min():.3f}, {batch_with_norm.max():.3f}]")

outputs_with_norm = session.run([batch_with_norm])
logits_with_norm = outputs_with_norm[0][0]

print(f"   Output shape: {outputs_with_norm[0].shape}")
print(f"   Output range: [{logits_with_norm.min():.3f}, {logits_with_norm.max():.3f}]")
print(f"   Output sum: {np.sum(logits_with_norm):.3f}")

# Apply softmax
exp_logits = np.exp(logits_with_norm - np.max(logits_with_norm))
probs_with_norm = exp_logits / np.sum(exp_logits)

top_5_with_norm = np.argsort(probs_with_norm)[::-1][:5]
print(f"\n   Top 5 predictions WITH normalization:")
for i, idx in enumerate(top_5_with_norm):
    print(f"      {i+1}. {classes[idx]:30s} - {probs_with_norm[idx]:.4f} ({probs_with_norm[idx]*100:.2f}%)")

# Compare
print(f"\n5. Analysis:")
print(f"   Expected: bar_chart should be top prediction")
print(f"   Without norm: {classes[top_5_no_norm[0]]}")
print(f"   With norm: {classes[top_5_with_norm[0]]}")

if classes[top_5_with_norm[0]] == "bar_chart":
    print(f"\n   ✅ WITH normalization gives CORRECT prediction!")
    print(f"   ❌ The predictor code is using normalization but still getting wrong results")
    print(f"   🔍 Possible issues:")
    print(f"      1. Python cache - old .pyc files being used")
    print(f"      2. ZDLC model compiled incorrectly")
    print(f"      3. Class label order mismatch")
elif classes[top_5_no_norm[0]] == "bar_chart":
    print(f"\n   ✅ WITHOUT normalization gives CORRECT prediction!")
    print(f"   ❌ The model was compiled WITHOUT normalization in mind")
    print(f"   🔍 Solution: Remove normalization from predictor")
else:
    print(f"\n   ❌ NEITHER gives correct prediction!")
    print(f"   🔍 Possible issues:")
    print(f"      1. ZDLC model compiled incorrectly")
    print(f"      2. Class label order is wrong")
    print(f"      3. Model expects different preprocessing")

print("\n" + "="*70)

# Made with Bob
