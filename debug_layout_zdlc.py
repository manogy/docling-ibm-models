#!/usr/bin/env python3
"""
Debug script to check ZDLC layout predictor outputs
"""
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, '/root/manogya/manogya/docling-ibm-models')

from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

# Initialize predictor
predictor = LayoutPredictor(
    "/root/manogya/manogya/docling-layout-heron",
    zdlc_model_path="/root/manogya/manogya/docling-layout-heron-NNPA.so",
    num_threads=4,
    base_threshold=0.01  # Very low threshold to see all outputs
)

print(f"Backend: {predictor._backend}")
print(f"Threshold: {predictor._threshold}")

# Load test image
test_img = "/root/manogya/manogya/docling-ibm-models/tests/test_data/samples/ADS.2007.page_123.png"
img = Image.open(test_img)
print(f"Image size: {img.size}")

# Prepare inputs manually to debug
page_img = img.convert("RGB")
target_sizes = np.array([page_img.size[::-1]], dtype=np.int64)
inputs = predictor._image_processor(images=[page_img], return_tensors="np")
pixel_values = inputs["pixel_values"].astype(np.float32)

print(f"\nInput shapes:")
print(f"  pixel_values: {pixel_values.shape}")
print(f"  target_sizes: {target_sizes.shape}")
print(f"  target_sizes values: {target_sizes}")

# Run ZDLC inference
print("\nRunning ZDLC inference...")
outputs = predictor._zdlc_session.run([pixel_values, target_sizes])

print(f"\nOutput shapes:")
for i, output in enumerate(outputs):
    print(f"  Output {i}: {output.shape}, dtype: {output.dtype}")
    print(f"    Min: {output.min():.4f}, Max: {output.max():.4f}, Mean: {output.mean():.4f}")

# Check logits and boxes
logits = outputs[0]
pred_boxes = outputs[1]

print(f"\nLogits analysis:")
print(f"  Shape: {logits.shape}")
print(f"  Sample logits[0,0,:5]: {logits[0,0,:5]}")

# Apply softmax
exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
scores = np.max(probs, axis=-1)
labels = np.argmax(probs, axis=-1)

print(f"\nScores analysis:")
print(f"  Shape: {scores.shape}")
print(f"  Min: {scores.min():.4f}, Max: {scores.max():.4f}")
print(f"  Top 10 scores: {np.sort(scores[0])[-10:]}")
print(f"  Number above 0.01: {np.sum(scores[0] > 0.01)}")
print(f"  Number above 0.1: {np.sum(scores[0] > 0.1)}")
print(f"  Number above 0.2: {np.sum(scores[0] > 0.2)}")
print(f"  Number above 0.3: {np.sum(scores[0] > 0.3)}")

print(f"\nBoxes analysis:")
print(f"  Shape: {pred_boxes.shape}")
print(f"  Sample boxes[0,:5]: {pred_boxes[0,:5]}")
print(f"  Min: {pred_boxes.min():.4f}, Max: {pred_boxes.max():.4f}")

# Try post-processing
print("\nPost-processing with threshold 0.01...")
results = predictor._post_process_object_detection(
    logits, pred_boxes, target_sizes, 0.01
)

print(f"Number of results: {len(results)}")
if len(results) > 0:
    result = results[0]
    print(f"  Scores: {len(result['scores'])}")
    print(f"  Labels: {len(result['labels'])}")
    print(f"  Boxes: {len(result['boxes'])}")
    if len(result['scores']) > 0:
        print(f"  Top 5 scores: {result['scores'][:5]}")
        print(f"  Top 5 labels: {result['labels'][:5]}")
        print(f"  Top 5 boxes: {result['boxes'][:5]}")

# Made with Bob
