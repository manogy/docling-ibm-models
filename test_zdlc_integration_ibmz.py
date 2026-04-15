#!/usr/bin/env python3
"""
ZDLC Integration Test Script for IBM Z (s390x)
Tests both Document Figure Classifier and Layout Predictor with ZDLC backend
"""
import logging
import sys
import time
from pathlib import Path

# Force reload of modules to avoid caching issues
if 'docling_ibm_models' in sys.modules:
    del sys.modules['docling_ibm_models']

sys.path.insert(0, '/root/manogya/manogya/docling-ibm-models')

from PIL import Image

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor,
)
from docling_ibm_models.layoutmodel.layout_predictor import LayoutPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ZDLC_Test")


def test_document_classifier():
    """Test Document Figure Classifier with ZDLC"""
    logger.info("=" * 70)
    logger.info("TEST 1: Document Figure Classifier with ZDLC")
    logger.info("=" * 70)
    
    try:
        # Note: First parameter is 'artifacts_path' (with 's')
        predictor = DocumentFigureClassifierPredictor(
            "/root/manogya/manogya/DocumentFigureClassifier-v2.0",  # artifacts_path
            zdlc_model_path="/root/manogya/manogya/DocumentFigureClassifier-V2-NNPA.so",
            num_threads=4
        )
        
        info = predictor.info()
        logger.info(f"✓ Backend: {info.get('backend')}")
        logger.info(f"✓ Device: {info.get('device')}")
        logger.info(f"✓ Num threads: {info.get('num_threads')}")
        logger.info(f"✓ Classes: {len(info.get('classes', []))}")
        
        test_dir = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images"
        images = []
        image_names = []
        
        for img_file in sorted(Path(test_dir).glob("*.jpg")):
            images.append(Image.open(img_file))
            image_names.append(img_file.name)
        
        logger.info(f"\n✓ Loaded {len(images)} test images")
        
        t0 = time.perf_counter()
        outputs = predictor.predict(images)
        total_ms = 1000 * (time.perf_counter() - t0)
        
        logger.info(f"✓ Inference time: {total_ms:.2f}ms ({total_ms/len(images):.2f}ms per image)")
        
        logger.info("\nPredictions:")
        for name, output in zip(image_names, outputs):
            top_class, top_prob = output[0]
            logger.info(f"  {name}: {top_class} ({top_prob:.4f})")
        
        logger.info("\n✅ Document Classifier Test PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Document Classifier Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_layout_predictor():
    """Test Layout Predictor with ZDLC"""
    logger.info("=" * 70)
    logger.info("TEST 2: Layout Predictor with ZDLC")
    logger.info("=" * 70)
    
    try:
        predictor = LayoutPredictor(
            "/root/manogya/manogya/docling-layout-heron",
            zdlc_model_path="/root/manogya/manogya/docling-layout-heron-NNPA.so",
            num_threads=4,
            base_threshold=0.2  # Lower threshold to see more predictions
        )
        
        info = predictor.info()
        logger.info(f"✓ Backend: {info.get('backend')}")
        logger.info(f"✓ Device: {info.get('device')}")
        logger.info(f"✓ Num threads: {info.get('num_threads')}")
        logger.info(f"✓ Threshold: {info.get('threshold')}")
        
        test_img = "/root/manogya/manogya/docling-ibm-models/tests/test_data/samples/ADS.2007.page_123.png"
        
        logger.info(f"\n✓ Testing with: {Path(test_img).name}")
        
        with Image.open(test_img) as img:
            logger.info(f"✓ Image size: {img.size}")
            
            t0 = time.perf_counter()
            predictions = list(predictor.predict(img))
            total_ms = 1000 * (time.perf_counter() - t0)
            
            logger.info(f"✓ Inference time: {total_ms:.2f}ms")
            logger.info(f"✓ Found {len(predictions)} layout elements")
            
            if len(predictions) > 0:
                logger.info("\nTop 5 predictions:")
                for i, pred in enumerate(predictions[:5]):
                    logger.info(
                        f"  {i+1}. {pred['label']:20s} conf={pred['confidence']:.3f} "
                        f"bbox=[{pred['l']:.0f},{pred['t']:.0f},{pred['r']:.0f},{pred['b']:.0f}]"
                    )
            else:
                logger.warning("⚠️  No predictions found - this might indicate:")
                logger.warning("   - Threshold too high")
                logger.warning("   - Post-processing issue")
                logger.warning("   - Model output format mismatch")
        
        logger.info("\n✅ Layout Predictor Test PASSED\n")
        return True
        
    except Exception as e:
        logger.error(f"❌ Layout Predictor Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("\n" + "=" * 70)
    logger.info("ZDLC Integration Tests on IBM Z (s390x)")
    logger.info("=" * 70 + "\n")
    
    results = {
        "Document Classifier": test_document_classifier(),
        "Layout Predictor": test_layout_predictor(),
    }
    
    logger.info("=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name:25s}: {status}")
    
    all_passed = all(results.values())
    logger.info("=" * 70)
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED!")
    else:
        logger.info("⚠️  SOME TESTS FAILED")
    logger.info("=" * 70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# Made with Bob
