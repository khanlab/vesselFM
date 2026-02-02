#!/bin/bash
# Example script demonstrating how to use the vesselFM CLI tool for inference

# Basic usage - minimum required arguments
python -m vesselfm.cli \
    --input-folder /path/to/images \
    --output-folder /path/to/output

# Use CPU instead of GPU
python -m vesselfm.cli \
    --input-folder /path/to/images \
    --output-folder /path/to/output \
    --device cpu

# Advanced usage with custom parameters
python -m vesselfm.cli \
    --input-folder /path/to/images \
    --output-folder /path/to/output \
    --device cuda:0 \
    --batch-size 8 \
    --patch-size 128 128 128 \
    --overlap 0.5 \
    --threshold 0.6 \
    --apply-postprocessing

# With evaluation (provide masks)
python -m vesselfm.cli \
    --input-folder /path/to/images \
    --output-folder /path/to/output \
    --mask-folder /path/to/masks \
    --device cuda:0

# With test-time augmentation
python -m vesselfm.cli \
    --input-folder /path/to/images \
    --output-folder /path/to/output \
    --tta-scales 0.5 1.0 1.5

# If installed via pip, you can also use:
# vesselfm-infer --input-folder /path/to/images --output-folder /path/to/output
