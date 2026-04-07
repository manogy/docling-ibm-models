#!/usr/bin/env python3
"""
Compare ONNX model vs ZDLC compiled model outputs to verify correctness.

This script runs the same input through both:
1. Original ONNX model (using onnxruntime)
2. ZDLC compiled .so model (using zdlc_pyrt)

And compares:
- Raw logits (before softmax)
- Probabilities (after softmax)
- Top predictions
- Statistical metrics (mean, std, min, max)

Usage:
    python compare_onnx_vs_zdlc.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import onnxruntime as ort
    print("✅ onnxruntime available")
except ImportError:
    print("❌ onnxruntime not available - install with: pip install onnxruntime")
    sys.exit(1)

try:
    import zdlc_pyrt
    print("✅ zdlc_pyrt available")
except ImportError:
    print("❌ zdlc_pyrt not available")
    sys.exit(1)

from docling_ibm_models.document_figure_classifier_model.document_figure_classifier_predictor_zdlc import (
    DocumentFigureClassifierPredictorZDLC,
)

# ============================================================================
# PATHS - Update these for your system
# ============================================================================

# IBM Z paths
ARTIFACTS_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0"
ONNX_MODEL_PATH = f"{ARTIFACTS_PATH}/model.onnx"
CPU_MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-CPU.so"
NNPA_MODEL_PATH = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/DocumentFigureClassifier-V2-NNPA.so"
TEST_IMAGES_DIR = "/root/manogya/manogya/docling-ibm-models/tests/test_data/figure_classifier/images"


def preprocess_image(image_path: str, target_size: tuple = (224, 224)):
    """
    Preprocess image exactly as the predictor does.
    
    Args:
        image_path: Path to image file
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed numpy array ready for model input
    """
    # Load and resize
    image = Image.open(image_path).convert("RGB")
    image = image.resize(target_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.47853944, 0.4732864, 0.47434163], dtype=np.float32)
    
    img_array = (img_array - mean) / std
    
    # Convert from HWC to CHW format
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def softmax(logits):
    """Apply softmax to convert logits to probabilities."""
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)


def compare_outputs(onnx_logits, zdlc_logits, class_names, image_name):
    """
    Compare ONNX and ZDLC outputs in detail.
    
    Args:
        onnx_logits: Raw logits from ONNX model
        zdlc_logits: Raw logits from ZDLC model
        class_names: List of class names
        image_name: Name of the image being tested
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {image_name}")
    print(f"{'='*80}\n")
    
    # 1. Raw Logits Comparison
    print("1. RAW LOGITS STATISTICS")
    print("-" * 80)
    print(f"{'Metric':<20} {'ONNX':<20} {'ZDLC':<20} {'Difference':<20}")
    print("-" * 80)
    
    onnx_mean = np.mean(onnx_logits)
    zdlc_mean = np.mean(zdlc_logits)
    print(f"{'Mean':<20} {onnx_mean:<20.6f} {zdlc_mean:<20.6f} {abs(onnx_mean - zdlc_mean):<20.6f}")
    
    onnx_std = np.std(onnx_logits)
    zdlc_std = np.std(zdlc_logits)
    print(f"{'Std Dev':<20} {onnx_std:<20.6f} {zdlc_std:<20.6f} {abs(onnx_std - zdlc_std):<20.6f}")
    
    onnx_min = np.min(onnx_logits)
    zdlc_min = np.min(zdlc_logits)
    print(f"{'Min':<20} {onnx_min:<20.6f} {zdlc_min:<20.6f} {abs(onnx_min - zdlc_min):<20.6f}")
    
    onnx_max = np.max(onnx_logits)
    zdlc_max = np.max(zdlc_logits)
    print(f"{'Max':<20} {onnx_max:<20.6f} {zdlc_max:<20.6f} {abs(onnx_max - zdlc_max):<20.6f}")
    
    onnx_range = onnx_max - onnx_min
    zdlc_range = zdlc_max - zdlc_min
    print(f"{'Range':<20} {onnx_range:<20.6f} {zdlc_range:<20.6f} {abs(onnx_range - zdlc_range):<20.6f}")
    
    # 2. Logits Difference Analysis
    print(f"\n2. LOGITS DIFFERENCE ANALYSIS")
    print("-" * 80)
    
    diff = np.abs(onnx_logits - zdlc_logits)
    print(f"Mean Absolute Error:     {np.mean(diff):.6f}")
    print(f"Max Absolute Error:      {np.max(diff):.6f}")
    print(f"Root Mean Square Error:  {np.sqrt(np.mean(diff**2)):.6f}")
    
    # Correlation
    correlation = np.corrcoef(onnx_logits.flatten(), zdlc_logits.flatten())[0, 1]
    print(f"Correlation:             {correlation:.6f}")
    
    # 3. Top Logits Comparison
    print(f"\n3. TOP 5 LOGITS (Raw Values)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Class':<30} {'ONNX Logit':<15} {'ZDLC Logit':<15} {'Diff':<10}")
    print("-" * 80)
    
    onnx_top_indices = np.argsort(onnx_logits)[::-1][:5]
    zdlc_top_indices = np.argsort(zdlc_logits)[::-1][:5]
    
    for i in range(5):
        onnx_idx = onnx_top_indices[i]
        onnx_val = onnx_logits[onnx_idx]
        zdlc_val = zdlc_logits[onnx_idx]
        diff_val = abs(onnx_val - zdlc_val)
        
        match = "✅" if onnx_idx in zdlc_top_indices else "❌"
        print(f"{i+1:<6} {class_names[onnx_idx]:<30} {onnx_val:<15.6f} {zdlc_val:<15.6f} {diff_val:<10.6f} {match}")
    
    # 4. Probabilities Comparison
    print(f"\n4. TOP 5 PROBABILITIES (After Softmax)")
    print("-" * 80)
    print(f"{'Rank':<6} {'Class':<30} {'ONNX Prob':<15} {'ZDLC Prob':<15} {'Diff':<10}")
    print("-" * 80)
    
    onnx_probs = softmax(onnx_logits)
    zdlc_probs = softmax(zdlc_logits)
    
    onnx_top_prob_indices = np.argsort(onnx_probs)[::-1][:5]
    
    for i in range(5):
        idx = onnx_top_prob_indices[i]
        onnx_p = onnx_probs[idx]
        zdlc_p = zdlc_probs[idx]
        diff_p = abs(onnx_p - zdlc_p)
        
        print(f"{i+1:<6} {class_names[idx]:<30} {onnx_p:<15.6f} {zdlc_p:<15.6f} {diff_p:<10.6f}")
    
    # 5. Prediction Agreement
    print(f"\n5. PREDICTION AGREEMENT")
    print("-" * 80)
    
    onnx_pred = class_names[np.argmax(onnx_probs)]
    zdlc_pred = class_names[np.argmax(zdlc_probs)]
    
    print(f"ONNX Prediction:  {onnx_pred} ({onnx_probs[np.argmax(onnx_probs)]:.4f})")
    print(f"ZDLC Prediction:  {zdlc_pred} ({zdlc_probs[np.argmax(zdlc_probs)]:.4f})")
    
    if onnx_pred == zdlc_pred:
        print("✅ PREDICTIONS MATCH!")
    else:
        print("❌ PREDICTIONS DO NOT MATCH!")
    
    # 6. Verdict
    print(f"\n6. VERDICT")
    print("-" * 80)
    
    mae = np.mean(diff)
    
    if mae < 0.01:
        print("✅ EXCELLENT: Models are nearly identical (MAE < 0.01)")
    elif mae < 0.1:
        print("✅ GOOD: Models are very similar (MAE < 0.1)")
    elif mae < 1.0:
        print("⚠️  ACCEPTABLE: Models have minor differences (MAE < 1.0)")
    elif mae < 5.0:
        print("⚠️  CONCERNING: Models have significant differences (MAE < 5.0)")
    else:
        print("❌ CRITICAL: Models are very different (MAE >= 5.0)")
    
    if correlation > 0.99:
        print("✅ EXCELLENT: Very high correlation (> 0.99)")
    elif correlation > 0.95:
        print("✅ GOOD: High correlation (> 0.95)")
    elif correlation > 0.90:
        print("⚠️  ACCEPTABLE: Moderate correlation (> 0.90)")
    else:
        print("❌ CRITICAL: Low correlation (< 0.90)")
    
    return {
        'mae': mae,
        'correlation': correlation,
        'predictions_match': onnx_pred == zdlc_pred,
        'onnx_pred': onnx_pred,
        'zdlc_pred': zdlc_pred,
    }


def main():
    """Main comparison function."""
    
    print("="*80)
    print("ONNX vs ZDLC Model Comparison")
    print("="*80)
    
    # Load class names
    config_path = Path(ARTIFACTS_PATH) / "config.json"
    with open(config_path) as f:
        config = json.load(f)
    
    # Get class names in id2label order (not alphabetical!)
    id2label = config["id2label"]
    class_names = [id2label[str(i)] for i in range(len(id2label))]
    
    print(f"\nModel has {len(class_names)} classes")
    print(f"Classes: {', '.join(class_names[:5])}...")
    
    # Test images
    test_images = [
        ("bar_chart.jpg", "bar_chart"),
        ("map.jpg", "geographical_map"),
    ]
    
    # Choose which ZDLC model to test
    print("\n" + "="*80)
    print("Which ZDLC model do you want to test?")
    print("="*80)
    print("1. CPU model")
    print("2. NNPA model")
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        zdlc_model_path = CPU_MODEL_PATH
        model_type = "CPU"
    elif choice == "2":
        zdlc_model_path = NNPA_MODEL_PATH
        model_type = "NNPA"
    else:
        print("Invalid choice, using CPU model")
        zdlc_model_path = CPU_MODEL_PATH
        model_type = "CPU"
    
    print(f"\n✅ Testing {model_type} model: {zdlc_model_path}")
    
    # Load ONNX model
    print(f"\n1. Loading ONNX model: {ONNX_MODEL_PATH}")
    onnx_session = ort.InferenceSession(ONNX_MODEL_PATH)
    input_name = onnx_session.get_inputs()[0].name
    print(f"   ✅ ONNX model loaded")
    print(f"   Input name: {input_name}")
    
    # Load ZDLC model
    print(f"\n2. Loading ZDLC model: {zdlc_model_path}")
    zdlc_session = zdlc_pyrt.InferenceSession(zdlc_model_path)
    print(f"   ✅ ZDLC model loaded")
    
    # Run comparisons
    results = []
    
    for img_name, expected_class in test_images:
        img_path = Path(TEST_IMAGES_DIR) / img_name
        
        if not img_path.exists():
            print(f"\n❌ Image not found: {img_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {img_name} (expected: {expected_class})")
        print(f"{'='*80}")
        
        # Preprocess image
        input_data = preprocess_image(str(img_path))
        print(f"Input shape: {input_data.shape}")
        print(f"Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
        
        # Run ONNX inference
        print("\n3. Running ONNX inference...")
        onnx_output = onnx_session.run(None, {input_name: input_data})[0]
        onnx_logits = onnx_output[0]  # Remove batch dimension
        print(f"   ✅ ONNX output shape: {onnx_logits.shape}")
        
        # Run ZDLC inference
        print("\n4. Running ZDLC inference...")
        zdlc_output = zdlc_session.run([input_data])[0]
        zdlc_logits = zdlc_output[0]  # Remove batch dimension
        print(f"   ✅ ZDLC output shape: {zdlc_logits.shape}")
        
        # Compare outputs
        result = compare_outputs(onnx_logits, zdlc_logits, class_names, img_name)
        result['image'] = img_name
        result['expected'] = expected_class
        results.append(result)
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    for result in results:
        print(f"\n{result['image']}:")
        print(f"  Expected:         {result['expected']}")
        print(f"  ONNX Predicted:   {result['onnx_pred']}")
        print(f"  ZDLC Predicted:   {result['zdlc_pred']}")
        print(f"  Match:            {'✅' if result['predictions_match'] else '❌'}")
        print(f"  MAE:              {result['mae']:.6f}")
        print(f"  Correlation:      {result['correlation']:.6f}")
    
    # Overall verdict
    all_match = all(r['predictions_match'] for r in results)
    avg_mae = np.mean([r['mae'] for r in results])
    avg_corr = np.mean([r['correlation'] for r in results])
    
    print("\n" + "="*80)
    print("OVERALL VERDICT")
    print("="*80)
    print(f"Model Type:           {model_type}")
    print(f"All Predictions Match: {'✅ YES' if all_match else '❌ NO'}")
    print(f"Average MAE:          {avg_mae:.6f}")
    print(f"Average Correlation:  {avg_corr:.6f}")
    
    if all_match and avg_mae < 0.1 and avg_corr > 0.99:
        print("\n✅ VERDICT: ZDLC model is CORRECT and matches ONNX model!")
    elif all_match and avg_mae < 1.0:
        print("\n✅ VERDICT: ZDLC model is ACCEPTABLE with minor numerical differences")
    elif all_match:
        print("\n⚠️  VERDICT: ZDLC model predictions match but has significant numerical differences")
    else:
        print("\n❌ VERDICT: ZDLC model is INCORRECT - predictions do not match ONNX!")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

# Made with Bob
