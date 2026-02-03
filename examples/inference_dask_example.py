"""Example usage of dask-based inference for vesselFM.

This example demonstrates how to use the dask-based inference module
to process large 3D images that don't fit in memory.
"""

import dask.array as da
from omegaconf import OmegaConf
from vesselfm.seg.inference_dask import run_inference_dask


def example_basic_usage():
    """Basic example of using dask inference."""
    
    # Load a large image as a dask array from a zarr file
    # (Zarr is a good format for chunked array storage)
    image = da.from_zarr('path/to/large_image.zarr')
    
    # Or load from a numpy array with specific chunking
    # import numpy as np
    # large_image = np.random.rand(512, 512, 512).astype(np.float32)
    # image = da.from_array(large_image, chunks=(128, 128, 128))
    
    # Load configuration (same as regular inference)
    cfg = OmegaConf.load('vesselfm/seg/configs/inference.yaml')
    
    # Override some config values for your use case
    cfg.device = 'cpu'  # or 'cuda' if GPU is available
    cfg.batch_size = 2
    cfg.patch_size = [64, 64, 64]
    
    # Run inference - this creates a lazy computation graph
    result = run_inference_dask(image, cfg)
    
    # Option 1: Compute the entire result
    # segmentation = result.compute()
    
    # Option 2: Save to disk without loading into memory (recommended for large arrays)
    result.to_zarr('path/to/output_segmentation.zarr', overwrite=True)
    
    # Option 3: Compute only a slice
    # slice_result = result[100:200, :, :].compute()
    
    print("Inference complete!")


def example_with_custom_chunks():
    """Example showing how to control chunking for optimal performance."""
    
    import numpy as np
    
    # Create a large synthetic image
    # In practice, load your actual data
    image = da.from_array(
        np.random.rand(256, 256, 256).astype(np.float32),
        chunks=(64, 64, 64)  # Adjust chunk size based on available memory
    )
    
    # Create configuration
    cfg = OmegaConf.create({
        'device': 'cpu',
        'transforms_config': [
            {'ScaleIntensityRange': {
                'a_min': 0, 'a_max': 1, 
                'b_min': 0, 'b_max': 1, 
                'clip': True
            }}
        ],
        'patch_size': [32, 32, 32],
        'batch_size': 2,
        'overlap': 0.25,
        'mode': 'gaussian',
        'sigma_scale': 0.125,
        'padding_mode': 'constant',
        'tta': {
            'scales': [1.0],
            'invert': False,
            'invert_mean_thresh': 0.5,
            'equalize_hist': False,
            'hist_bins': 256,
        },
        'merging': {
            'max': False,
            'threshold': 0.5,
        },
        'post': {
            'apply': False,
            'small_objects_min_size': 10,
            'small_objects_connectivity': 2,
        }
    })
    
    # Run inference
    result = run_inference_dask(image, cfg)
    
    # Compute and save
    segmentation = result.compute()
    print(f"Segmentation shape: {segmentation.shape}")
    print(f"Segmentation dtype: {segmentation.dtype}")
    print(f"Unique values: {np.unique(segmentation)}")


def example_distributed_computing():
    """Example showing how to use dask with distributed computing."""
    
    from dask.distributed import Client
    import dask.array as da
    
    # Set up a local cluster (or connect to an existing one)
    # Note: memory_limit is per worker, so total memory usage = n_workers × memory_limit
    # In this example: 4 workers × 4GB = 16GB total (cumulative across workers)
    # Ensure your system has at least 16GB + overhead for scheduler and main process (recommend 20GB+ available)
    client = Client(n_workers=4, threads_per_worker=2, memory_limit='4GB')
    
    try:
        # Load image
        image = da.from_zarr('path/to/large_image.zarr')
        
        # Load configuration
        cfg = OmegaConf.load('vesselfm/seg/configs/inference.yaml')
        cfg.device = 'cpu'  # Use 'cuda' if workers have GPUs
        
        # Run inference with distributed computing
        result = run_inference_dask(image, cfg)
        
        # Persist the result in distributed memory (optional)
        result = result.persist()
        
        # Save to disk
        result.to_zarr('path/to/output_segmentation.zarr', overwrite=True)
        
    finally:
        # Clean up
        client.close()
    
    print("Distributed inference complete!")


if __name__ == '__main__':
    # Run the basic example
    # Uncomment to run:
    # example_basic_usage()
    # example_with_custom_chunks()
    # example_distributed_computing()
    
    print("Examples ready to run. Uncomment the function calls above.")
