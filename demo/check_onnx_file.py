#!/usr/bin/env python3
"""
Quick check to see if ONNX file is a Git LFS pointer or actual model.
"""

import os
import sys

onnx_path = "/root/manogya/manogya/DocumentFigureClassifier-v2.0/model.onnx"

print("=" * 70)
print("CHECKING ONNX MODEL FILE")
print("=" * 70)

if not os.path.exists(onnx_path):
    print(f"\n✗ File not found: {onnx_path}")
    sys.exit(1)

file_size = os.path.getsize(onnx_path)
file_size_mb = file_size / (1024 * 1024)

print(f"\nFile: {onnx_path}")
print(f"Size: {file_size} bytes ({file_size_mb:.2f} MB)")

# Read first few lines to check if it's a Git LFS pointer
with open(onnx_path, 'rb') as f:
    first_bytes = f.read(200)
    
try:
    first_text = first_bytes.decode('utf-8', errors='ignore')
except:
    first_text = ""

print("\nFirst 200 bytes:")
print("-" * 70)
print(first_text[:200])
print("-" * 70)

# Check if it's a Git LFS pointer
if "version https://git-lfs.github.com" in first_text:
    print("\n✗ THIS IS A GIT LFS POINTER FILE, NOT THE ACTUAL MODEL!")
    print("\nThe file contains a pointer to the actual model stored in Git LFS.")
    print("\nTo download the actual model, run:")
    print("  cd /root/manogya/manogya/DocumentFigureClassifier-v2.0")
    print("  git lfs install")
    print("  git lfs pull")
    print("\nAfter that, the model.onnx file should be >10MB")
    sys.exit(1)
elif file_size < 1024 * 1024:  # Less than 1MB
    print("\n⚠ WARNING: File is very small (<1MB)")
    print("This is likely not a complete model file.")
    print("\nPossible issues:")
    print("1. Git LFS pointer that wasn't detected")
    print("2. Incomplete download")
    print("3. Model weights stored separately")
    print("\nExpected size for vision models: 10-400MB")
    sys.exit(1)
else:
    print("\n✓ File size looks reasonable for a model file")
    print("\nNext step: Test the ONNX model")
    print("  python demo/test_onnx_model.py")

# Made with Bob
