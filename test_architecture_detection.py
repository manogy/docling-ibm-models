#!/usr/bin/env python3
"""
Test script for architecture-based model selection (ZDLC vs PyTorch).
This script automatically detects the architecture and uses the
appropriate backend.
"""

import platform
import sys
from pathlib import Path
from typing import Optional

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
    print("=" * 80)
    print()


def test_layout_predictor(
    image_path: str,
    model_path: str,
    zdlc_model_path: Optional[str] = None
):
    """Test LayoutPredictor with automatic architecture detection"""
    print("=" * 80)
    print("TESTING LAYOUT PREDICTOR")
    print("=" * 80)

    # Initialize predictor - it will automatically detect architecture
    print(f"Initializing LayoutPredictor from: {model_path}")
    if zdlc_model_path:
        print(f"ZDLC model path: {zdlc_model_path}")

    try:
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
        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        print("\nRunning prediction...")
        predictions = list(predictor.predict(image))

        print(f"\nFound {len(predictions)} predictions:")
        for i, pred in enumerate(predictions[:5], 1):  # Show first 5
            print(
                f"  {i}. Label: {pred['label']}, "
                f"Confidence: {pred['confidence']:.3f}, "
                f"BBox: ({pred['l']:.1f}, {pred['t']:.1f}, "
                f"{pred['r']:.1f}, {pred['b']:.1f})"
            )

        if len(predictions) > 5:
            print(f"  ... and {len(predictions) - 5} more predictions")

        print("\n✅ Layout Predictor test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Layout Predictor test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_figure_classifier(image_path: str, model_path: str):
    """Test DocumentFigureClassifierPredictor with auto detection"""
    print("\n" + "=" * 80)
    print("TESTING DOCUMENT FIGURE CLASSIFIER")
    print("=" * 80)

    print(f"Initializing DocumentFigureClassifierPredictor "
          f"from: {model_path}")

    try:
        predictor = DocumentFigureClassifierPredictor(
            artifacts_path=model_path,
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
        print(f"\nLoading image: {image_path}")
        image = Image.open(image_path)
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")

        print("\nRunning prediction...")
        # predict() expects a list of images
        results = predictor.predict([image])

        print("\nPrediction results (top 5):")
        for i, (label, confidence) in enumerate(results[0][:5], 1):
            print(f"  {i}. {label}: {confidence:.3f}")

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

    # Default paths - adjust these based on your setup
    layout_model_path = "models/layout"
    layout_zdlc_path = None  # Path to ZDLC .so file (s390x only)
    figure_model_path = "models/figure_classifier"

    # Test images
    test_image_layout = "tests/test_data/samples/empty_iocr.png"
    test_image_figure = (
        "tests/test_data/figure_classifier/images/bar_chart.jpg"
    )

    # Check if running on s390x
    is_s390x = platform.machine().lower() in ['s390x', 's390']
    if is_s390x:
        print("⚠️  Detected s390x architecture - "
              "ZDLC backend will be used")
        print("⚠️  Make sure to set layout_zdlc_path "
              "to your ZDLC .so file")
        print()
    else:
        print("ℹ️  Detected non-s390x architecture - "
              "PyTorch backend will be used")
        print()

    # Run tests
    results = []

    # Test 1: Layout Predictor
    if Path(test_image_layout).exists():
        result = test_layout_predictor(
            test_image_layout, layout_model_path, layout_zdlc_path
        )
        results.append(("Layout Predictor", result))
    else:
        print(f"⚠️  Test image not found: {test_image_layout}")
        print("   Skipping Layout Predictor test")

    # Test 2: Figure Classifier
    if Path(test_image_figure).exists():
        result = test_figure_classifier(
            test_image_figure, figure_model_path
        )
        results.append(("Figure Classifier", result))
    else:
        print(f"⚠️  Test image not found: {test_image_figure}")
        print("   Skipping Figure Classifier test")

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
        return 0
    else:
        print("⚠️  Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
