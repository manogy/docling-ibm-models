#!/usr/bin/env python3
"""
Debug script to compare PyTorch and ZDLC model outputs.
This helps identify where the ZDLC model diverges from expected behavior.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image

# Test with PyTorch model first
print("=" * 70)
print("TESTING PYTORCH MODEL")
print("=" * 70)

from huggingface_hub import snapshot_download

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor,
)

# Download model
artifact_path = snapshot_download(
    repo_id="ds4sd/DocumentFigureClassifier",
    revision="v1.0.0"
)

# Load test image
test_image = Image.open("tests/test_data/figure_classifier/images/bar_chart.jpg")
print(f"\nTest image: bar_chart.jpg")
print(f"Image size: {test_image.size}")
print(f"Image mode: {test_image.mode}")

# PyTorch prediction
pytorch_predictor = DocumentFigureClassifierPredictor(
    artifact_path,
    device="cpu",
    num_threads=4
)

pytorch_results = pytorch_predictor.predict([test_image])
print("\nPyTorch Top 5 Predictions:")
for i, (class_name, prob) in enumerate(pytorch_results[0][:5], 1):
    print(f"  {i}. {class_name:30s} {prob:.4f} ({prob*100:.2f}%)")

# Now test ZDLC
print("\n" + "=" * 70)
print("TESTING ZDLC MODEL")
print("=" * 70)

try:
    from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
        DocumentFigureClassifierPredictorZDLC,
    )
    
    zdlc_model_path = "../DocumentFigureClassifier-V2.so"
    
    zdlc_predictor = DocumentFigureClassifierPredictorZDLC(
        artifact_path,
        zdlc_model_path=zdlc_model_path,
        num_threads=4
    )
    
    # Get preprocessed image to inspect
    import torchvision.transforms as transforms
    
    image_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163],
        ),
    ])
    
    rgb_image = test_image.convert("RGB")
    processed = image_processor(rgb_image)
    numpy_image = processed.numpy().astype(np.float32)
    
    print(f"\nPreprocessed image shape: {numpy_image.shape}")
    print(f"Preprocessed image dtype: {numpy_image.dtype}")
    print(f"Preprocessed image range: [{numpy_image.min():.3f}, {numpy_image.max():.3f}]")
    print(f"Preprocessed image mean: {numpy_image.mean():.3f}")
    print(f"Preprocessed image std: {numpy_image.std():.3f}")
    
    # ZDLC prediction
    zdlc_results = zdlc_predictor.predict([test_image])
    
    print("\nZDLC Top 5 Predictions:")
    for i, (class_name, prob) in enumerate(zdlc_results[0][:5], 1):
        print(f"  {i}. {class_name:30s} {prob:.4f} ({prob*100:.2f}%)")
    
    # Compare
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    pytorch_top = pytorch_results[0][0]
    zdlc_top = zdlc_results[0][0]
    
    print(f"\nPyTorch top prediction: {pytorch_top[0]} ({pytorch_top[1]:.4f})")
    print(f"ZDLC top prediction:    {zdlc_top[0]} ({zdlc_top[1]:.4f})")
    
    if pytorch_top[0] == zdlc_top[0]:
        print("✓ Both models agree on top class")
    else:
        print("✗ Models disagree on top class")
    
    prob_diff = abs(pytorch_top[1] - zdlc_top[1])
    print(f"\nConfidence difference: {prob_diff:.4f}")
    
    if prob_diff < 0.01:
        print("✓ Predictions are very close")
    elif prob_diff < 0.1:
        print("⚠ Predictions are somewhat different")
    else:
        print("✗ Predictions are very different - model issue likely")
    
    # Check if ZDLC output looks uniform
    zdlc_probs = [prob for _, prob in zdlc_results[0]]
    zdlc_std = np.std(zdlc_probs)
    
    print(f"\nZDLC probability std dev: {zdlc_std:.4f}")
    if zdlc_std < 0.01:
        print("✗ ZDLC outputs are too uniform - model not working correctly")
        print("\nPossible issues:")
        print("  1. ONNX export didn't preserve preprocessing")
        print("  2. ZDLC compilation has issues")
        print("  3. Input format mismatch")
    
except Exception as e:
    print(f"\n✗ ZDLC test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)

print("""
If ZDLC predictions are uniform/wrong:

1. Verify ONNX export includes preprocessing:
   - The model should accept raw pixel values [0-255]
   - OR preprocessing should be done before ZDLC

2. Check ZDLC compilation:
   - Ensure correct opset version
   - Verify no optimization issues

3. Test with original PyTorch model:
   python -m demo.demo_document_figure_classifier_predictor \\
       -i tests/test_data/figure_classifier/images -d cpu

4. Re-export ONNX with preprocessing included:
   See ZDLC_DEMO_GUIDE.md for export instructions
""")

# Made with Bob
