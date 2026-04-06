#!/usr/bin/env python3
"""
Standalone script to test ZDLC Document Figure Classifier.
Can be run directly without module imports.

Usage:
    python demo/test_zdlc_classifier.py \
        -i tests/test_data/figure_classifier/images \
        -z /path/to/model.so
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to path to import docling_ibm_models
sys.path.insert(0, str(Path(__file__).parent.parent))

from huggingface_hub import snapshot_download
from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)


def demo(
    logger: logging.Logger,
    artifact_path: str,
    zdlc_model_path: str,
    num_threads: int,
    image_dir: str,
):
    """Run ZDLC Document Figure Classifier demo"""
    
    logger.info("=" * 70)
    logger.info("ZDLC Document Figure Classifier Demo")
    logger.info("=" * 70)
    logger.info(f"Artifacts path: {artifact_path}")
    logger.info(f"ZDLC model path: {zdlc_model_path}")
    logger.info(f"Image directory: {image_dir}")
    logger.info(f"Number of threads: {num_threads}")
    logger.info("=" * 70)

    # Initialize predictor
    logger.info("\nInitializing ZDLC predictor...")
    try:
        predictor = DocumentFigureClassifierPredictorZDLC(
            artifacts_path=artifact_path,
            zdlc_model_path=zdlc_model_path,
            num_threads=num_threads
        )
        logger.info("✓ Predictor initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize predictor: {e}")
        return

    # Display model info
    info = predictor.info()
    logger.info(f"\nModel Info:")
    for key, value in info.items():
        if key == "classes":
            logger.info(f"  {key}: {len(value)} classes")
        else:
            logger.info(f"  {key}: {value}")

    # Load images
    image_path = Path(image_dir)
    if not image_path.exists():
        logger.error(f"✗ Image directory not found: {image_dir}")
        return

    image_files = sorted([
        f for f in os.listdir(image_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not image_files:
        logger.error(f"✗ No images found in {image_dir}")
        return

    logger.info(f"\nLoading {len(image_files)} images...")
    images = []
    for img_file in image_files:
        try:
            img = Image.open(image_path / img_file)
            images.append(img)
            logger.info(f"  ✓ {img_file} - Size: {img.size}")
        except Exception as e:
            logger.warning(f"  ✗ Failed to load {img_file}: {e}")

    if not images:
        logger.error("✗ No images loaded successfully")
        return

    # Run inference
    logger.info(f"\nRunning inference on {len(images)} images...")
    try:
        t0 = time.perf_counter()
        outputs = predictor.predict(images)
        total_ms = 1000 * (time.perf_counter() - t0)
        avg_ms = total_ms / len(images)
        
        logger.info("✓ Inference complete!")
        logger.info(f"  Total time: {total_ms:.1f} ms")
        logger.info(f"  Average per image: {avg_ms:.1f} ms")
        
        # Debug: Check if predictions look reasonable
        if outputs and len(outputs) > 0:
            first_pred = outputs[0]
            if first_pred and len(first_pred) > 0:
                top_conf = first_pred[0][1]
                if top_conf < 0.1:
                    logger.warning(
                        f"⚠ Low confidence detected ({top_conf:.4f}). "
                        "Model output may need adjustment."
                    )
    except Exception as e:
        logger.error(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 70)

    for i, (img_file, predictions) in enumerate(zip(image_files, outputs)):
        logger.info(f"\n[{i+1}/{len(image_files)}] {img_file}")
        logger.info("-" * 70)
        
        # Show top 5 predictions
        for j, (class_name, probability) in enumerate(predictions[:5]):
            bar_length = int(probability * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            logger.info(
                f"  {j+1}. {class_name:30s} {bar} {probability:.4f} "
                f"({probability*100:.2f}%)"
            )

    logger.info("\n" + "=" * 70)
    logger.info("Demo completed successfully!")
    logger.info("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Test ZDLC Document Figure Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings
  python demo/test_zdlc_classifier.py \\
      -i tests/test_data/figure_classifier/images \\
      -z /path/to/model.so

  # Run with custom threads
  python demo/test_zdlc_classifier.py \\
      -i tests/test_data/figure_classifier/images \\
      -z /path/to/model.so \\
      -n 8

  # Use custom artifact path (skip HuggingFace download)
  python demo/test_zdlc_classifier.py \\
      -i tests/test_data/figure_classifier/images \\
      -z /path/to/model.so \\
      -a /path/to/artifacts
        """
    )
    
    parser.add_argument(
        "-i", "--image_dir",
        required=True,
        help="Directory containing input images (JPG/PNG)"
    )
    parser.add_argument(
        "-z", "--zdlc_model_path",
        required=True,
        help="Path to ZDLC compiled .so model file"
    )
    parser.add_argument(
        "-n", "--num_threads",
        type=int,
        default=4,
        help="Number of threads for inference (default: 4)"
    )
    parser.add_argument(
        "-a", "--artifact_path",
        default=None,
        help="Path to model artifacts (if not provided, downloads from HF)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger(__name__)

    # Get artifact path
    if args.artifact_path:
        artifact_path = args.artifact_path
        logger.info(f"Using provided artifact path: {artifact_path}")
    else:
        logger.info("Downloading model artifacts from HuggingFace...")
        try:
            artifact_path = snapshot_download(
                repo_id="ds4sd/DocumentFigureClassifier",
                revision="v1.0.0"
            )
            logger.info(f"✓ Downloaded to: {artifact_path}")
        except Exception as e:
            logger.error(f"✗ Failed to download artifacts: {e}")
            logger.error("Try providing artifact path with -a option")
            return 1

    # Verify ZDLC model exists
    if not os.path.exists(args.zdlc_model_path):
        logger.error(f"✗ ZDLC model not found: {args.zdlc_model_path}")
        return 1

    # Run demo
    try:
        demo(
            logger=logger,
            artifact_path=artifact_path,
            zdlc_model_path=args.zdlc_model_path,
            num_threads=args.num_threads,
            image_dir=args.image_dir
        )
        return 0
    except Exception as e:
        logger.error(f"✗ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
