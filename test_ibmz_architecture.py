#!/usr/bin/env python3
"""
Test script for IBM Z (s390x) architecture with ZDLC backend.
This script uses your existing models and test data.
"""

import platform
import sys
from pathlib import Path

from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (  # noqa: E501
    DocumentFigureClassifierPredictor,
)
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor


def print_system_info():
    """Print system and architecture information"""
    print("=" * 80)
    print("SYSTEM INFORMATION")
    print("=" * 80)
    print(f"Platform: {platform.platform()}")
    print(f"Machine: {platform.machine()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {sys.version}")
    
    # Check if ZDLC is available
    try:
        import zdlc_pyrt
        print(f"ZDLC version: Available")
        print(f"ZDLC backend: Will be used automatically on s390x")
    except ImportError:
        print("ZDLC: Not available (will use PyTorch)")
    
    print("=" * 80)
    print()


def test_layout_predictor():
    """Test LayoutPredictor with IBM Z setup"""
    print("=" * 80)
    print("TESTING LAYOUT PREDICTOR")
    print("=" * 80)

    # Paths for IBM Z system
    # artifact_path: Directory containing config.json, preprocessor_config.json
    # zdlc_model_path: Path to the compiled ZDLC .so file for inference
    model_path = "/root/manogya/manogya/docling-layout-heron"
    zdlc_model_path = "/root/manogya/manogya/docling-layout-heron-NNPA.so"
    test_image = "tests/test_data/samples/empty_iocr.png"

    print(f"Model artifacts path: {model_path}")
    print(f"  (contains: config.json, preprocessor_config.json)")
    print(f"ZDLC model path: {zdlc_model_path}")
    print(f"  (compiled .so file for NNPA inference)")
    print(f"Test image: {test_image}")
    print()

    try:
        # Initialize predictor - will auto-detect s390x and use ZDLC
        print("Initializing LayoutPredictor...")
        predictor = LayoutPredictor(
            artifact_path=model_path,
            zdlc_model_path=zdlc_model_path,
            device="cpu",
        )

        # Print predictor info
        info = predictor.info()
        print("\nPredictor Configuration:")
        for key, value in info.items():
            print(f"  {key}: {value}")

        # Load and predict
        print(f"\nLoading image: {test_image}")
        if not Path(test_image).exists():
            print(f"❌ Image not found: {test_image}")
            return False
            
        image = Image.open(test_image)
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        print("\nRunning prediction...")
        predictions = list(predictor.predict(image))

        print(f"\n✅ Prediction completed!")
        print(f"Found {len(predictions)} layout elements:")
        
        # Group by label
        label_counts = {}
        for pred in predictions:
            label = pred['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("\nLayout elements by type:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label}: {count}")
        
        # Show first 5 predictions
        print("\nFirst 5 predictions:")
        for i, pred in enumerate(predictions[:5], 1):
            print(
                f"  {i}. {pred['label']}: "
                f"conf={pred['confidence']:.3f}, "
                f"bbox=({pred['l']:.0f},{pred['t']:.0f},"
                f"{pred['r']:.0f},{pred['b']:.0f})"
            )

        print("\n✅ Layout Predictor test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Layout Predictor test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_figure_classifier():
    """Test DocumentFigureClassifierPredictor with IBM Z setup"""
    print("\n" + "=" * 80)
    print("TESTING DOCUMENT FIGURE CLASSIFIER")
    print("=" * 80)

    # Paths for IBM Z system
    # artifacts_path: Directory containing config.json, preprocessor_config.json
    # zdlc_model_path: Path to the compiled ZDLC .so file for inference
    model_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
    zdlc_model_path = (
        "/root/manogya/manogya/"
        "DocumentFigureClassifier-V2-NNPA.so"
    )
    test_image = "tests/test_data/figure_classifier/images/bar_chart.jpg"

    print(f"Model artifacts path: {model_path}")
    print(f"  (contains: config.json, preprocessor_config.json)")
    print(f"ZDLC model path: {zdlc_model_path}")
    print(f"  (compiled .so file for NNPA inference)")
    print(f"Test image: {test_image}")
    print()

    try:
        # Initialize predictor - will auto-detect s390x and use ZDLC
        print("Initializing DocumentFigureClassifierPredictor...")
        predictor = DocumentFigureClassifierPredictor(
            artifacts_path=model_path,
            zdlc_model_path=zdlc_model_path,
            device="cpu",
        )

        # Print predictor info
        info = predictor.info()
        print("\nPredictor Configuration:")
        for key, value in info.items():
            if key == "classes":
                print(f"  {key}: {len(value)} classes")
            else:
                print(f"  {key}: {value}")

        # Load and predict
        print(f"\nLoading image: {test_image}")
        if not Path(test_image).exists():
            print(f"❌ Image not found: {test_image}")
            return False
            
        image = Image.open(test_image)
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        print("\nRunning prediction...")
        # predict() expects a list of images
        results = predictor.predict([image])

        print("\n✅ Prediction completed!")
        print("\nTop 5 predictions:")
        for i, (label, confidence) in enumerate(results[0][:5], 1):
            print(f"  {i}. {label}: {confidence:.4f}")

        print("\n✅ Figure Classifier test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Figure Classifier test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print_system_info()

    # Check architecture
    is_s390x = platform.machine().lower() in ['s390x', 's390']
    if not is_s390x:
        print("⚠️  WARNING: This script is configured for IBM Z (s390x)")
        print(f"   Current architecture: {platform.machine()}")
        print("   The script will still run but may need path adjustments")
        print()

    # Run tests
    results = []

    # Test 1: Layout Predictor
    print("Starting Layout Predictor test...")
    result = test_layout_predictor()
    results.append(("Layout Predictor", result))

    # Test 2: Figure Classifier
    print("\nStarting Figure Classifier test...")
    result = test_figure_classifier()
    results.append(("Figure Classifier", result))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result for _, result in results)
    print("=" * 80)
    if all_passed:
        print("🎉 All tests PASSED!")
        print("\nThe ZDLC backend is working correctly on your IBM Z system!")
        return 0
    else:
        print("⚠️  Some tests FAILED")
        print("\nPlease check the error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
