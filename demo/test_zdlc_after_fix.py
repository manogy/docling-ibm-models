#!/usr/bin/env python3
"""
Test ZDLC predictor after fixing preprocessing.
This should now give high confidence predictions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

print("=" * 70)
print("TESTING ZDLC PREDICTOR WITH FIXED PREPROCESSING")
print("=" * 70)

# Paths
artifacts_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
zdlc_model_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.so"
test_image_path = "tests/test_data/figure_classifier/images/bar_chart.jpg"

print(f"\nArtifacts: {artifacts_path}")
print(f"ZDLC Model: {zdlc_model_path}")
print(f"Test Image: {test_image_path}")

# Check if model.so exists
if not Path(zdlc_model_path).exists():
    print(f"\n✗ ZDLC model not found: {zdlc_model_path}")
    print("\nYou need to compile the ONNX model first:")
    print(f"  cd {artifacts_path}")
    print("  zdlc -O3 model.onnx -o model.so")
    sys.exit(1)

print("\n" + "=" * 70)
print("INITIALIZING PREDICTOR")
print("=" * 70)

try:
    predictor = DocumentFigureClassifierPredictorZDLC(
        artifacts_path=artifacts_path,
        zdlc_model_path=zdlc_model_path,
        num_threads=4,
    )
    print("✓ Predictor initialized successfully")
    print(f"  Backend: {predictor.info()['backend']}")
    print(f"  Classes: {len(predictor.info()['classes'])}")
except Exception as e:
    print(f"✗ Failed to initialize predictor: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)

try:
    # Load test image
    img = Image.open(test_image_path)
    print(f"\n✓ Loaded image: {img.size}")
    
    # Run prediction
    predictions = predictor.predict([img])
    
    print(f"\n✓ Prediction successful")
    print(f"  Number of predictions: {len(predictions)}")
    print(f"  Classes per prediction: {len(predictions[0])}")
    
    # Display top 5 predictions
    print(f"\nTop 5 Predictions for {Path(test_image_path).name}:")
    for i, (class_name, confidence) in enumerate(predictions[0][:5], 1):
        print(f"  {i}. {class_name:30s} ({confidence*100:5.2f}%)")
    
    # Check if predictions look good
    top_confidence = predictions[0][0][1]
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    
    if top_confidence > 0.5:
        print(f"\n✓ EXCELLENT! Top prediction has {top_confidence*100:.2f}% confidence")
        print("  The preprocessing fix worked!")
        print("\nNext steps:")
        print("  1. Test with more images")
        print("  2. Apply same fix to other model predictors")
        print("  3. Benchmark ZDLC vs PyTorch performance")
    elif top_confidence > 0.2:
        print(f"\n⚠ MODERATE: Top prediction has {top_confidence*100:.2f}% confidence")
        print("  Better than before, but could be improved")
    else:
        print(f"\n✗ LOW: Top prediction has {top_confidence*100:.2f}% confidence")
        print("  Still having issues - may need further investigation")
        
except Exception as e:
    print(f"\n✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)

# Made with Bob
