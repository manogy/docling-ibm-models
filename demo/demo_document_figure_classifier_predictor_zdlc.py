import argparse
import logging
import os
import sys
import time
from pathlib import Path

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
    viz_dir: str,
):
    r"""
    Apply DocumentFigureClassifierPredictorZDLC on the input image directory
    """
    # Create the document figure classifier predictor with ZDLC
    logger.info("Initializing ZDLC Document Figure Classifier...")
    logger.info(f"Artifacts path: {artifact_path}")
    logger.info(f"ZDLC model path: {zdlc_model_path}")

    document_figure_classifier_predictor = (
        DocumentFigureClassifierPredictorZDLC(
            artifact_path, zdlc_model_path=zdlc_model_path, num_threads=num_threads
        )
    )

    # Display model info
    info = document_figure_classifier_predictor.info()
    logger.info(f"Model info: {info}")

    image_dir = Path(image_dir)
    images = []
    image_names = os.listdir(image_dir)
    image_names.sort()

    logger.info(f"Loading {len(image_names)} images from {image_dir}...")
    for image_name in image_names:
        image = Image.open(image_dir / image_name)
        images.append(image)
        logger.debug(f"Loaded: {image_name} - Size: {image.size}")

    logger.info("Running inference...")
    t0 = time.perf_counter()
    outputs = document_figure_classifier_predictor.predict(images)
    total_ms = 1000 * (time.perf_counter() - t0)
    avg_ms = (total_ms / len(image_names)) if len(image_names) > 0 else 0

    logger.info(
        "For {} images(ms): [total|avg] = [{:.1f}|{:.1f}]".format(
            len(image_names), total_ms, avg_ms
        )
    )

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
            logger.info(
                f"  {j+1}. {class_name:30s} - {probability:.4f} "
                f"({probability*100:.2f}%)"
            )

    logger.info("\n" + "=" * 70)


def main(args):
    num_threads = int(args.num_threads) if args.num_threads is not None else 4
    image_dir = args.image_dir
    viz_dir = args.viz_dir
    zdlc_model_path = args.zdlc_model_path
    artifact_path = args.artifact_path

    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DocumentFigureClassifierPredictorZDLC")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Ensure the viz dir
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Use local artifacts or download from HF if not provided
    if artifact_path and Path(artifact_path).exists():
        logger.info(f"Using local model artifacts from: {artifact_path}")
        download_path = artifact_path
    else:
        logger.info("Downloading model artifacts from HuggingFace...")
        download_path = snapshot_download(
            repo_id="ds4sd/DocumentFigureClassifier", revision="v1.0.0"
        )
        logger.info(f"Downloaded to: {download_path}")

    # Test the figure classifier model with ZDLC
    demo(logger, download_path, zdlc_model_path, num_threads, image_dir, viz_dir)


if __name__ == "__main__":
    r"""
    Demo script for testing DocumentFigureClassifierPredictorZDLC

    Usage with local artifacts (recommended):
    python -m demo.demo_document_figure_classifier_predictor_zdlc \
        -i tests/test_data/figure_classifier/images \
        -z /path/to/compiled/model.so \
        -a /path/to/model/artifacts

    Usage with HuggingFace download:
    python -m demo.demo_document_figure_classifier_predictor_zdlc \
        -i tests/test_data/figure_classifier/images \
        -z /path/to/compiled/model.so

    This will:
    1. Load model config from local path or download from HuggingFace
    2. Load your ZDLC compiled .so model
    3. Run inference on test images
    4. Display classification results with probabilities
    """
    parser = argparse.ArgumentParser(
        description="Test the DocumentFigureClassifierPredictorZDLC"
    )
    parser.add_argument(
        "-n",
        "--num_threads",
        required=False,
        default=4,
        help="Number of threads for inference",
    )
    parser.add_argument(
        "-i",
        "--image_dir",
        required=True,
        help="Directory containing input images (PNG/JPG)",
    )
    parser.add_argument(
        "-z",
        "--zdlc_model_path",
        required=True,
        help="Path to ZDLC compiled .so model file",
    )
    parser.add_argument(
        "-a",
        "--artifact_path",
        required=False,
        default=None,
        help="Path to local model artifacts (config.json, preprocessor_config.json). "
             "If not provided, will download from HuggingFace.",
    )
    parser.add_argument(
        "-v",
        "--viz_dir",
        required=False,
        default="viz/",
        help="Directory to save prediction visualizations",
    )

    args = parser.parse_args()
    main(args)

# Made with Bob
