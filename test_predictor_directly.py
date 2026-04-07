#!/usr/bin/env python3
"""
Direct test of DocumentFigureClassifierPredictorZDLC class.

This script uses the predictor class directly without the demo wrapper.
"""

import sys
from pathlib import Path

from PIL import Image

# Import the predictor class
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

# Configuration
NNPA_MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
ARTIFACTS_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
TEST_IMAGES_DIR = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images"
NUM_THREADS = 4

def main():
    print("="*70)
    print("Direct Predictor Test - NNPA Model")
    print("="*70)
    
    # Initialize predictor
    print(f"\n1. Initializing predictor...")
    print(f"   Model: {NNPA_MODEL_PATH}")
    print(f"   Artifacts: {ARTIFACTS_PATH}")
    print(f"   Threads: {NUM_THREADS}")
    
    predictor = DocumentFigureClassifierPredictorZDLC(
        artifacts_path=ARTIFACTS_PATH,
        zdlc_model_path=NNPA_MODEL_PATH,
        num_threads=NUM_THREADS,
    )
    
    print("   ✅ Predictor initialized")
    
    # Show model info
    info = predictor.info()
    print(f"\n2. Model info:")
    print(f"   Backend: {info['backend']}")
    print(f"   Threads: {info['num_threads']}")
    print(f"   Classes: {len(info['classes'])}")
    
    # Load test images
    print(f"\n3. Loading test images from: {TEST_IMAGES_DIR}")
    test_dir = Path(TEST_IMAGES_DIR)
    
    images = []
    image_names = []
    
    for img_path in sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png")):
        image = Image.open(img_path)
        images.append(image)
        image_names.append(img_path.name)
        print(f"   Loaded: {img_path.name} ({image.size})")
    
    print(f"   ✅ Loaded {len(images)} images")
    
    # Run prediction
    print(f"\n4. Running prediction...")
    predictions = predictor.predict(images)
    print(f"   ✅ Prediction complete")
    
    # Display results
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    
    for i, (image_name, preds) in enumerate(zip(image_names, predictions)):
        print(f"\nImage {i+1}: {image_name}")
        print("-"*70)
        
        # Show top 5 predictions
        for j, (class_name, prob) in enumerate(preds[:5]):
            confidence_icon = "✅" if j == 0 and prob > 0.5 else "  "
            print(f"{confidence_icon} {j+1}. {class_name:30s} - {prob:.4f} ({prob*100:.2f}%)")
    
    print(f"\n{'='*70}")
    print("✅ Test complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Made with Bob
