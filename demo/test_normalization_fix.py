#!/usr/bin/env python3
"""
Test script to verify the normalization fix for ZDLC Document Figure Classifier.

This script:
1. Tests the original PyTorch model
2. Tests the ZDLC model with the normalization fix
3. Compares the predictions to verify they match

Usage:
    python demo/test_normalization_fix.py \
        --pytorch-artifacts /path/to/pytorch/model \
        --zdlc-model /path/to/model.so \
        --zdlc-artifacts /path/to/zdlc/artifacts \
        --image tests/test_data/figure_classifier/images/bar_chart.jpg
     
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor import (
    DocumentFigureClassifierPredictor,
)
from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)


def print_predictions(predictions, model_name):
    """Print predictions in a formatted way."""
    print(f"\n{'='*70}")
    print(f"{model_name} Predictions:")
    print(f"{'='*70}")
    for i, pred_list in enumerate(predictions):
        print(f"\nImage {i+1}:")
        print("-" * 70)
        for j, (class_name, prob) in enumerate(pred_list[:5]):
            print(f"  {j+1}. {class_name:30s} - {prob:.6f} ({prob*100:.2f}%)")


def compare_predictions(pytorch_preds, zdlc_preds, threshold=0.01):
    """Compare predictions from PyTorch and ZDLC models."""
    print(f"\n{'='*70}")
    print("Comparison Results:")
    print(f"{'='*70}")
    
    all_match = True
    for i, (pt_preds, zd_preds) in enumerate(zip(pytorch_preds, zdlc_preds)):
        print(f"\nImage {i+1}:")
        print("-" * 70)
        
        # Compare top prediction
        pt_top_class, pt_top_prob = pt_preds[0]
        zd_top_class, zd_top_prob = zd_preds[0]
        
        if pt_top_class == zd_top_class:
            print(f"✅ Top class MATCHES: {pt_top_class}")
            print(f"   PyTorch prob: {pt_top_prob:.6f}")
            print(f"   ZDLC prob:    {zd_top_prob:.6f}")
            print(f"   Difference:   {abs(pt_top_prob - zd_top_prob):.6f}")
            
            if abs(pt_top_prob - zd_top_prob) > threshold:
                print(f"   ⚠️  WARNING: Probability difference > {threshold}")
                all_match = False
        else:
            print(f"❌ Top class MISMATCH!")
            print(f"   PyTorch: {pt_top_class} ({pt_top_prob:.6f})")
            print(f"   ZDLC:    {zd_top_class} ({zd_top_prob:.6f})")
            all_match = False
    
    print(f"\n{'='*70}")
    if all_match:
        print("✅ ALL PREDICTIONS MATCH! Normalization fix is working correctly.")
    else:
        print("❌ PREDICTIONS DON'T MATCH! There may still be an issue.")
    print(f"{'='*70}\n")
    
    return all_match


def main():
    parser = argparse.ArgumentParser(
        description="Test normalization fix for ZDLC Document Figure Classifier"
    )
    parser.add_argument(
        "--pytorch-artifacts",
        required=True,
        help="Path to PyTorch model artifacts directory",
    )
    parser.add_argument(
        "--zdlc-model",
        required=True,
        help="Path to ZDLC compiled .so model file",
    )
    parser.add_argument(
        "--zdlc-artifacts",
        required=True,
        help="Path to ZDLC model artifacts directory (config files)",
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for inference",
    )
    
    args = parser.parse_args()
    
    # Load image
    print(f"\n{'='*70}")
    print(f"Loading image: {args.image}")
    print(f"{'='*70}")
    image = Image.open(args.image)
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")
    
    # Test PyTorch model
    print(f"\n{'='*70}")
    print("Initializing PyTorch model...")
    print(f"{'='*70}")
    pytorch_predictor = DocumentFigureClassifierPredictor(
        artifacts_path=args.pytorch_artifacts,
        device="cpu",
        num_threads=args.num_threads,
    )
    print("✅ PyTorch model loaded successfully")
    
    print("\nRunning PyTorch inference...")
    pytorch_predictions = pytorch_predictor.predict([image])
    print_predictions(pytorch_predictions, "PyTorch")
    
    # Test ZDLC model
    print(f"\n{'='*70}")
    print("Initializing ZDLC model...")
    print(f"{'='*70}")
    print(f"ZDLC model path: {args.zdlc_model}")
    print(f"ZDLC artifacts path: {args.zdlc_artifacts}")
    
    zdlc_predictor = DocumentFigureClassifierPredictorZDLC(
        artifacts_path=args.zdlc_artifacts,
        zdlc_model_path=args.zdlc_model,
        num_threads=args.num_threads,
    )
    print("✅ ZDLC model loaded successfully")
    
    print("\nRunning ZDLC inference...")
    zdlc_predictions = zdlc_predictor.predict([image])
    print_predictions(zdlc_predictions, "ZDLC")
    
    # Compare predictions
    matches = compare_predictions(pytorch_predictions, zdlc_predictions)
    
    return 0 if matches else 1


if __name__ == "__main__":
    sys.exit(main())

# Made with Bob
