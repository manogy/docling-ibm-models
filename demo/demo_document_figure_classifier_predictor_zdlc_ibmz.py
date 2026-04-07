#!/usr/bin/env python3
"""
ZDLC Document Figure Classifier Demo - IBM Z Version
Hardcoded paths for IBM Z system at /root/manogya/manogya/

This script runs the DocumentFigureClassifierPredictorZDLC with hardcoded paths
for the IBM Z system setup.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to Python path to import docling_ibm_models
script_dir = Path(__file__).parent.parent
sys.path.insert(0, str(script_dir))

from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

# ============================================================================
# HARDCODED PATHS FOR IBM Z SYSTEM
# ============================================================================
IBM_Z_BASE_PATH = "/root/manogya/manogya"
IBM_Z_MODEL_DIR = f"{IBM_Z_BASE_PATH}/DocumentFigureClassifier-v2.0"
IBM_Z_NNPA_MODEL = f"{IBM_Z_MODEL_DIR}/DocumentFigureClassifier-V2-NNPA.so"
IBM_Z_CPU_MODEL = f"{IBM_Z_MODEL_DIR}/DocumentFigureClassifier-V2-CPU.so"
IBM_Z_ARTIFACTS = IBM_Z_MODEL_DIR  # Contains config.json and preprocessor_config.json
IBM_Z_DOCLING_REPO = f"{IBM_Z_BASE_PATH}/docling-ibm-models"
IBM_Z_TEST_IMAGES = f"{IBM_Z_DOCLING_REPO}/tests/test_data/figure_classifier/images"
# ============================================================================


def demo(
    logger: logging.Logger,
    artifact_path: str,
    zdlc_model_path: str,
    num_threads: int,
    image_dir: str,
    viz_dir: str,
):
    r"""
    Apply DocumentFigureClassifierPredictorZDLC on the input image directory
    """
    # Verify paths exist
    if not Path(zdlc_model_path).exists():
        logger.error(f"❌ ZDLC model not found: {zdlc_model_path}")
        sys.exit(1)
    
    if not Path(artifact_path).exists():
        logger.error(f"❌ Artifacts directory not found: {artifact_path}")
        sys.exit(1)
    
    config_file = Path(artifact_path) / "config.json"
    if not config_file.exists():
        logger.error(f"❌ config.json not found in: {artifact_path}")
        sys.exit(1)
    
    if not Path(image_dir).exists():
        logger.error(f"❌ Image directory not found: {image_dir}")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("ZDLC Document Figure Classifier - IBM Z Demo")
    logger.info("=" * 70)
    logger.info(f"✅ ZDLC model: {zdlc_model_path}")
    logger.info(f"✅ Artifacts: {artifact_path}")
    logger.info(f"✅ Images: {image_dir}")
    logger.info(f"✅ Threads: {num_threads}")
    logger.info("=" * 70)
    
    # Create the document figure classifier predictor with ZDLC
    logger.info("\nInitializing ZDLC Document Figure Classifier...")
    
    try:
        document_figure_classifier_predictor = (
            DocumentFigureClassifierPredictorZDLC(
                artifact_path, zdlc_model_path=zdlc_model_path, num_threads=num_threads
            )
        )
        logger.info("✅ Predictor initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize predictor: {e}")
        raise

    # Display model info
    info = document_figure_classifier_predictor.info()
    logger.info(f"\nModel info:")
    logger.info(f"  Backend: {info['backend']}")
    logger.info(f"  Threads: {info['num_threads']}")
    logger.info(f"  Classes: {len(info['classes'])}")

    image_dir = Path(image_dir)
    images = []
    image_names = []
    
    # Get all image files
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_names.extend([f.name for f in image_dir.glob(ext)])
    
    image_names.sort()
    
    if not image_names:
        logger.error(f"❌ No images found in {image_dir}")
        sys.exit(1)

    logger.info(f"\nLoading {len(image_names)} images from {image_dir}...")
    for image_name in image_names:
        try:
            image = Image.open(image_dir / image_name)
            images.append(image)
            logger.debug(f"  Loaded: {image_name} - Size: {image.size}")
        except Exception as e:
            logger.warning(f"  ⚠️  Failed to load {image_name}: {e}")

    if not images:
        logger.error("❌ No images loaded successfully")
        sys.exit(1)

    logger.info(f"✅ Loaded {len(images)} images")
    
    logger.info("\nRunning inference...")
    try:
        t0 = time.perf_counter()
        outputs = document_figure_classifier_predictor.predict(images)
        total_ms = 1000 * (time.perf_counter() - t0)
        avg_ms = (total_ms / len(images)) if len(images) > 0 else 0
        
        logger.info("✅ Inference completed")
        logger.info(
            f"Performance: {len(images)} images in {total_ms:.1f}ms "
            f"(avg: {avg_ms:.1f}ms per image)"
        )
    except Exception as e:
        logger.error(f"❌ Inference failed: {e}")
        raise

    # Display results
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTION RESULTS")
    logger.info("=" * 70)

    for i, output in enumerate(outputs):
        image_name = image_names[i]
        logger.info(f"\nImage: '{image_name}'")
        logger.info("-" * 70)

        # Show top 5 predictions
        for j, pred in enumerate(output[:5]):
            class_name, probability = pred
            confidence_icon = "✅" if j == 0 and probability > 0.5 else "  "
            logger.info(
                f"{confidence_icon} {j+1}. {class_name:30s} - {probability:.4f} "
                f"({probability*100:.2f}%)"
            )

    logger.info("\n" + "=" * 70)
    
    # Save results summary
    if viz_dir:
        viz_path = Path(viz_dir)
        viz_path.mkdir(parents=True, exist_ok=True)
        summary_file = viz_path / "predictions_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("ZDLC Document Figure Classifier - Prediction Results\n")
            f.write("=" * 70 + "\n\n")
            for i, output in enumerate(outputs):
                image_name = image_names[i]
                f.write(f"Image: {image_name}\n")
                f.write("-" * 70 + "\n")
                for j, pred in enumerate(output[:5]):
                    class_name, probability = pred
                    f.write(f"  {j+1}. {class_name:30s} - {probability:.4f} ({probability*100:.2f}%)\n")
                f.write("\n")
        
        logger.info(f"✅ Results saved to: {summary_file}")


def main(args):
    # Use hardcoded paths or command line arguments
    num_threads = int(args.num_threads) if args.num_threads is not None else 4
    
    # Use hardcoded paths if not provided via command line
    image_dir = args.image_dir if args.image_dir else IBM_Z_TEST_IMAGES
    zdlc_model_path = args.zdlc_model_path if args.zdlc_model_path else IBM_Z_NNPA_MODEL
    artifact_path = args.artifact_path if args.artifact_path else IBM_Z_ARTIFACTS
    viz_dir = args.viz_dir if args.viz_dir else "viz_nnpa_results/"
    
    # Initialize logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger("DocumentFigureClassifierZDLC")
    
    # Ensure the viz dir
    Path(viz_dir).mkdir(parents=True, exist_ok=True)
    
    # Run demo
    demo(logger, artifact_path, zdlc_model_path, num_threads, image_dir, viz_dir)


if __name__ == "__main__":
    r"""
    ZDLC Document Figure Classifier Demo - IBM Z Version
    
    This script has hardcoded paths for the IBM Z system.
    You can run it without any arguments to use defaults:
    
    python demo_document_figure_classifier_predictor_zdlc_ibmz.py
    
    Or override specific paths:
    
    python demo_document_figure_classifier_predictor_zdlc_ibmz.py \
        -z /path/to/model.so \
        -i /path/to/images \
        -n 8
    
    Hardcoded defaults:
    - Model: /root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so
    - Artifacts: /root/manogya/manogya/DocumentFigureClassifier-v2.0
    - Images: /root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images
    - Threads: 4
    """
    parser = argparse.ArgumentParser(
        description="ZDLC Document Figure Classifier Demo - IBM Z Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Hardcoded Paths (used if not specified):
  Model (NNPA):  {IBM_Z_NNPA_MODEL}
  Model (CPU):   {IBM_Z_CPU_MODEL}
  Artifacts:     {IBM_Z_ARTIFACTS}
  Test Images:   {IBM_Z_TEST_IMAGES}

Examples:
  # Use all defaults (NNPA model, test images)
  python {sys.argv[0]}
  
  # Use CPU model instead
  python {sys.argv[0]} -z {IBM_Z_CPU_MODEL}
  
  # Use different image directory
  python {sys.argv[0]} -i /path/to/your/images
  
  # Use more threads
  python {sys.argv[0]} -n 8
        """
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        required=False,
        default=4,
        type=int,
        help=f"Number of threads for inference (default: 4)",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        required=False,
        default=None,
        help=f"Directory containing input images (default: {IBM_Z_TEST_IMAGES})",
    )
    parser.add_argument(
        "-z",
        "--zdlc_model_path",
        required=False,
        default=None,
        help=f"Path to ZDLC compiled .so model file (default: {IBM_Z_NNPA_MODEL})",
    )
    parser.add_argument(
        "-a",
        "--artifact_path",
        required=False,
        default=None,
        help=f"Path to model artifacts directory (default: {IBM_Z_ARTIFACTS})",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default=None,
        help="Directory to save prediction results (default: viz_nnpa_results/)",
    )

    args = parser.parse_args()
    main(args)

# Made with Bob for IBM Z