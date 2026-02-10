# Dask-based Parallel Chunking Implementation

## Overview

This implementation enables Dask to chunk individual images, run inference on each chunk in parallel, and merge the results using Gaussian blending. This is particularly beneficial for processing very large 3D medical images.

## Key Features

### 1. Two Levels of Parallelism

The implementation now supports two types of parallelism:

- **Multi-image parallelism** (existing): Parallel loading and preprocessing of multiple images
- **Chunk-level parallelism** (new): Parallel processing of chunks within a single image

### 2. Flexible Configuration

Users can control the behavior via CLI arguments or configuration file:

```bash
# Default: Both multi-image and chunk-level parallelism enabled
python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output

# Disable chunk-level parallelism (use traditional sliding window)
python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --disable-dask-chunking

# Disable all Dask parallelism
python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --disable-dask

# Control number of workers
python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --dask-workers 8
```

### 3. Seamless Integration

The implementation integrates seamlessly with the existing inference pipeline:
- Automatically falls back to sequential processing for single chunks
- Compatible with all existing features (TTA, post-processing, etc.)
- No changes required to existing code or workflows

## Implementation Details

### New Module: `vesselfm/seg/utils/dask_patching.py`

This module provides the core functionality for Dask-based chunking:

1. **`compute_patch_coords()`**: Computes coordinates for all patches in an image
   - Handles overlap between patches
   - Adjusts edge patches to maintain consistent size

2. **`extract_patch()`**: Extracts a single patch from an image

3. **`compute_gaussian_importance_map()`**: Creates a Gaussian weight map for blending
   - Center of each patch has higher weight
   - Edges have lower weight for smooth blending

4. **`stitch_patches()`**: Merges patch results using Gaussian blending
   - Weighted average based on importance maps
   - Handles edge cases and boundary patches

5. **`process_image_with_dask_chunks()`**: End-to-end processing pipeline
   - Coordinates patch extraction, parallel inference, and stitching
   - Supports both Dask and sequential processing

### Modified Files

1. **`vesselfm/seg/inference.py`**
   - Updated `process_image()` to support Dask chunking
   - Automatically selects processing method based on configuration

2. **`vesselfm/cli.py`**
   - Added `--enable-dask-chunking` and `--disable-dask-chunking` flags
   - Flags are mutually exclusive

3. **`vesselfm/seg/configs/inference.yaml`**
   - Added `chunk_images: true` to enable chunk-level parallelism by default

## Testing

### Unit Tests
- `tests/test_dask_chunking.py`: Comprehensive unit tests for all functions
- `tests/validate_dask_chunking.py`: Standalone validation script
- `tests/test_cli_config.py`: CLI configuration tests

### Integration Tests
- `tests/integration_test.py`: End-to-end integration testing
- All tests verify consistency between Dask and sequential processing

### Existing Tests
- Updated `tests/test_dask_integration.py` to include new arguments
- Updated `tests/test_cli.py` to include new arguments

## Performance Considerations

### When to Use Chunk-Level Parallelism

Chunk-level parallelism is most beneficial when:
- Processing very large 3D volumes (>500MB)
- Multiple CPU cores are available
- I/O is not the bottleneck

### When to Disable Chunk-Level Parallelism

Consider disabling chunk-level parallelism when:
- Processing many small images (multi-image parallelism is more efficient)
- Limited CPU resources
- Running on single-core systems

## Backward Compatibility

The implementation is fully backward compatible:
- Existing code continues to work without modification
- Default behavior enables chunk-level parallelism
- Can be disabled via CLI or configuration

## Security

- No security vulnerabilities detected by CodeQL scanner
- No external dependencies added (uses existing Dask installation)
- No sensitive data exposure

## Future Enhancements

Potential future improvements:
- Multi-GPU support for chunk-level parallelism
- Adaptive chunk size based on available memory
- Distributed cluster support for very large datasets

## References

- Original SlidingWindowInfererAdapt from MONAI
- Dask documentation: https://docs.dask.org/
- Gaussian blending for seamless tiling
