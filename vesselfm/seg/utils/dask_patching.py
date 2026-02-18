"""Dask-based image patching and inference utilities."""

import logging
import os
import time
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import psutil

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CPU_COUNT = 4  # Fallback when os.cpu_count() returns None
EPSILON = 1e-8  # Small value to prevent division by zero in normalization


def get_available_gpus() -> List[str]:
    """
    Get list of available GPU devices.
    
    Returns:
        List of GPU device strings (e.g., ['cuda:0', 'cuda:1'])
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        return []
    
    n_gpus = torch.cuda.device_count()
    logger.info(f"Found {n_gpus} GPU(s) available")
    return [f"cuda:{i}" for i in range(n_gpus)]


def setup_gpu_workers(
    gpu_ids: Optional[List[int]] = None,
    use_distributed: bool = False
) -> Tuple[List[str], Optional[Any]]:
    """
    Setup GPU workers for multi-GPU processing.
    
    Args:
        gpu_ids: List of GPU IDs to use (None = use all available GPUs)
        use_distributed: Whether to use Dask distributed scheduler
        
    Returns:
        Tuple of (list of device strings, dask client or None)
    """
    available_gpus = get_available_gpus()
    
    if not available_gpus:
        logger.info("No GPUs available, using CPU")
        return ["cpu"], None
    
    # Determine which GPUs to use
    if gpu_ids is not None:
        devices = [f"cuda:{gpu_id}" for gpu_id in gpu_ids if gpu_id < len(available_gpus)]
        if not devices:
            logger.warning(f"Specified GPU IDs {gpu_ids} are not available, using all GPUs")
            devices = available_gpus
    else:
        devices = available_gpus
    
    logger.info(f"Using GPUs: {devices}")
    
    # Setup distributed scheduler if requested
    client = None
    if use_distributed and len(devices) > 1:
        try:
            from dask.distributed import Client, LocalCluster
            
            # Create a local cluster with one worker per GPU
            logger.info(f"Setting up Dask distributed cluster with {len(devices)} GPU workers")
            cluster = LocalCluster(
                n_workers=len(devices),
                threads_per_worker=2,
                processes=True,  # Use processes for better GPU isolation
                memory_limit='auto'
            )
            client = Client(cluster)
            logger.info(f"Dask cluster dashboard available at: {client.dashboard_link}")
        except Exception as e:
            logger.warning(f"Could not setup distributed scheduler: {e}. Using threaded scheduler.")
            client = None
    
    return devices, client


def compute_patch_coords(
    image_shape: Tuple[int, int, int, int, int],
    patch_size: Tuple[int, int, int],
    overlap: float
) -> List[Tuple[int, int, int, int, int, int]]:
    """
    Compute coordinates for all patches in the image.
    
    Args:
        image_shape: Shape of the image (batch, channel, D, H, W)
        patch_size: Size of each patch (D, H, W)
        overlap: Overlap ratio between patches (0.0-1.0)
        
    Returns:
        List of tuples (start_d, end_d, start_h, end_h, start_w, end_w)
    """
    _, _, img_d, img_h, img_w = image_shape
    patch_d, patch_h, patch_w = patch_size
    
    # Compute step size based on overlap for each dimension
    steps = [max(1, int(size * (1 - overlap))) for size in patch_size]
    step_d, step_h, step_w = steps
    
    coords = []
    
    # Generate patch coordinates
    for start_d in range(0, img_d, step_d):
        for start_h in range(0, img_h, step_h):
            for start_w in range(0, img_w, step_w):
                end_d = min(start_d + patch_d, img_d)
                end_h = min(start_h + patch_h, img_h)
                end_w = min(start_w + patch_w, img_w)
                
                # Adjust start coordinates for edge patches to maintain patch size
                if end_d - start_d < patch_d:
                    start_d = max(0, end_d - patch_d)
                if end_h - start_h < patch_h:
                    start_h = max(0, end_h - patch_h)
                if end_w - start_w < patch_w:
                    start_w = max(0, end_w - patch_w)
                
                coords.append((start_d, end_d, start_h, end_h, start_w, end_w))
    
    # Remove duplicate coordinates (can happen at boundaries)
    coords = list(set(coords))
    coords.sort()
    
    return coords


def extract_patch(
    image: torch.Tensor,
    coords: Tuple[int, int, int, int, int, int]
) -> torch.Tensor:
    """
    Extract a patch from the image.
    
    Args:
        image: Input image tensor (batch, channel, D, H, W)
        coords: Patch coordinates (start_d, end_d, start_h, end_h, start_w, end_w)
        
    Returns:
        Patch tensor
    """
    start_d, end_d, start_h, end_h, start_w, end_w = coords
    return image[:, :, start_d:end_d, start_h:end_h, start_w:end_w]


def compute_gaussian_importance_map(
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125
) -> np.ndarray:
    """
    Compute a 3D Gaussian importance map for patch blending.
    
    Args:
        patch_size: Size of the patch (D, H, W)
        sigma_scale: Scale factor for Gaussian sigma
        
    Returns:
        3D Gaussian importance map
    """
    patch_d, patch_h, patch_w = patch_size
    
    # Create 1D Gaussian profiles for each dimension
    def gaussian_1d(length, sigma_scale):
        sigma = length * sigma_scale
        x = np.arange(length)
        center = (length - 1) / 2.0
        gauss = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        return gauss
    
    gauss_d = gaussian_1d(patch_d, sigma_scale)
    gauss_h = gaussian_1d(patch_h, sigma_scale)
    gauss_w = gaussian_1d(patch_w, sigma_scale)
    
    # Create 3D Gaussian map using outer product
    importance_map = np.einsum('i,j,k->ijk', gauss_d, gauss_h, gauss_w)
    
    return importance_map


def stitch_patches(
    patch_results: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int, int, int],
    patch_size: Tuple[int, int, int],
    sigma_scale: float = 0.125
) -> torch.Tensor:
    """
    Stitch patch results back together with Gaussian blending.
    
    Args:
        patch_results: List of dicts with 'logits' and 'coords'
        image_shape: Original image shape (batch, channel, D, H, W)
        patch_size: Size of each patch (D, H, W)
        sigma_scale: Scale factor for Gaussian blending
        
    Returns:
        Stitched output tensor
    """
    _, _, img_d, img_h, img_w = image_shape
    
    # Initialize output and weight maps
    output = torch.zeros((1, 1, img_d, img_h, img_w), dtype=torch.float32)
    weights = torch.zeros((1, 1, img_d, img_h, img_w), dtype=torch.float32)
    
    # Compute Gaussian importance map
    importance_map = compute_gaussian_importance_map(patch_size, sigma_scale)
    importance_map = torch.from_numpy(importance_map).float()
    
    # Aggregate patches
    for result in patch_results:
        logits = result['logits']
        start_d, end_d, start_h, end_h, start_w, end_w = result['coords']
        
        # Get actual patch size (may be smaller at boundaries)
        actual_d = end_d - start_d
        actual_h = end_h - start_h
        actual_w = end_w - start_w
        
        # Resize importance map if needed (for edge patches)
        if (actual_d, actual_h, actual_w) != patch_size:
            weight = F.interpolate(
                importance_map.unsqueeze(0).unsqueeze(0),
                size=(actual_d, actual_h, actual_w),
                mode='trilinear',
                align_corners=False
            ).squeeze()
        else:
            weight = importance_map
        
        # Ensure logits has the right shape
        if logits.dim() == 3:
            logits = logits.unsqueeze(0).unsqueeze(0)
        
        # Add weighted patch to output
        output[:, :, start_d:end_d, start_h:end_h, start_w:end_w] += logits * weight
        weights[:, :, start_d:end_d, start_h:end_h, start_w:end_w] += weight
    
    # Normalize by weights (add epsilon to prevent division by zero)
    output = output / (weights + EPSILON)
    
    return output


def infer_patch_batch(
    patch_batch: torch.Tensor,
    model: torch.nn.Module,
    device: str
) -> torch.Tensor:
    """
    Run inference on a batch of patches.
    
    Args:
        patch_batch: Batch of patches (batch_size, channel, D, H, W)
        model: PyTorch model
        device: Device to run inference on
        
    Returns:
        Logits for the batch
    """
    with torch.no_grad():
        patch_batch = patch_batch.to(device)
        logits = model(patch_batch)
        return logits.cpu()


def process_patch_on_gpu(
    coords: Tuple[int, int, int, int, int, int],
    image: torch.Tensor,
    model: torch.nn.Module,
    device: str
) -> Dict[str, Any]:
    """
    Process a single patch on a specific GPU.
    
    This function is designed to be called by Dask workers, with each worker
    assigned to a specific GPU device.
    
    Args:
        coords: Patch coordinates (start_d, end_d, start_h, end_h, start_w, end_w)
        image: Input image tensor (shared across workers)
        model: PyTorch model (will be moved to specified device)
        device: Device to run inference on (e.g., 'cuda:0', 'cuda:1')
        
    Returns:
        Dictionary with 'logits' and 'coords'
    """
    # Extract the patch
    patch = extract_patch(image, coords)
    
    # Run inference on the specified device
    logits = infer_patch_batch(patch, model, device)
    
    return {'logits': logits.squeeze(0).squeeze(0), 'coords': coords}


def process_image_with_dask_chunks(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    patch_size: Tuple[int, int, int],
    overlap: float,
    batch_size: int,
    sigma_scale: float = 0.125,
    use_dask: bool = True,
    n_workers: int = None,
    gpu_ids: Optional[List[int]] = None,
    use_multi_gpu: bool = False
) -> torch.Tensor:
    """
    Process an image using Dask-based chunking for parallel inference.
    
    Supports both single-GPU and multi-GPU processing:
    - Single GPU: Patches are processed in parallel on one GPU
    - Multi-GPU: Patches are distributed across multiple GPUs for parallel processing
    
    Args:
        image: Input image tensor (1, 1, D, H, W)
        model: PyTorch model
        device: Device for inference (used if multi-GPU is disabled)
        patch_size: Size of each patch (D, H, W)
        overlap: Overlap ratio between patches
        batch_size: Batch size for patch processing
        sigma_scale: Gaussian blending scale
        use_dask: Whether to use Dask for parallel processing
        n_workers: Number of Dask workers (None = auto)
        gpu_ids: List of GPU IDs to use for multi-GPU (None = use all available)
        use_multi_gpu: Whether to distribute work across multiple GPUs
        
    Returns:
        Output logits tensor
    """
    # Start timing
    start_time = time.time()
    
    # Get current process for memory tracking
    process = psutil.Process(os.getpid())
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    
    image_shape = image.shape
    
    # Compute patch coordinates
    coords_list = compute_patch_coords(image_shape, patch_size, overlap)
    # image_shape[2:] extracts spatial dimensions (D, H, W) from (batch, channel, D, H, W)
    logger.info(f"Image shape: {image_shape[2:]}, Chunk size: {patch_size}, Number of chunks: {len(coords_list)}")
    logger.debug(f"Generated {len(coords_list)} patches for image of shape {image_shape}")
    
    # Setup GPUs if multi-GPU is requested
    devices = [device]
    dask_client = None
    
    if use_multi_gpu and use_dask:
        devices, dask_client = setup_gpu_workers(gpu_ids, use_distributed=True)
        if len(devices) > 1:
            logger.info(f"Multi-GPU processing enabled with {len(devices)} GPUs: {devices}")
        else:
            logger.info("Only one GPU available or multi-GPU setup failed, using single GPU")
            use_multi_gpu = False
    
    if use_dask and len(coords_list) > 1:
        if use_multi_gpu and len(devices) > 1:
            # Multi-GPU parallel processing with Dask
            logger.info(f"Processing {len(coords_list)} patches in parallel across {len(devices)} GPUs")
            import dask
            import dask.bag as db
            from dask.diagnostics import ProgressBar
            
            # If using distributed scheduler, don't override
            if dask_client is None:
                dask.config.set(scheduler='threads')
            
            # Create a function that assigns patches to GPUs in round-robin fashion
            def process_patch_multi_gpu(args):
                idx, coords = args
                # Assign GPU in round-robin fashion
                gpu_device = devices[idx % len(devices)]
                return process_patch_on_gpu(coords, image, model, gpu_device)
            
            # Create Dask bag with enumerated coordinates
            npartitions = min(len(coords_list), len(devices) * 2)  # 2 tasks per GPU
            bag = db.from_sequence(list(enumerate(coords_list)), npartitions=npartitions)
            
            # Process in parallel
            with ProgressBar():
                if dask_client:
                    patch_results = bag.map(process_patch_multi_gpu).compute()
                else:
                    patch_results = bag.map(process_patch_multi_gpu).compute()
                    
        else:
            # Single-GPU parallel processing
            logger.info(f"Processing {len(coords_list)} patches in parallel with Dask")
            import dask
            import dask.bag as db
            from dask.diagnostics import ProgressBar
            
            # Configure Dask
            dask.config.set(scheduler='threads')
            
            # Determine number of partitions
            if n_workers:
                npartitions = min(len(coords_list), n_workers)
            else:
                cpu_count = os.cpu_count() or DEFAULT_CPU_COUNT
                npartitions = min(len(coords_list), cpu_count)
            
            # Create function for processing a single patch
            def process_single_patch(coords):
                patch = extract_patch(image, coords)
                # Process in batches of 1 since we're parallelizing across patches
                logits = infer_patch_batch(patch, model, device)
                return {'logits': logits.squeeze(0).squeeze(0), 'coords': coords}
            
            # Create Dask bag and process in parallel
            bag = db.from_sequence(coords_list, npartitions=npartitions)
            with ProgressBar():
                patch_results = bag.map(process_single_patch).compute()
    else:
        # Sequential processing with batching
        if use_dask:
            logger.info(f"Dask enabled but only 1 patch, using sequential processing")
        else:
            logger.info(f"Processing {len(coords_list)} patches sequentially")
        
        patch_results = []
        for i in range(0, len(coords_list), batch_size):
            batch_coords = coords_list[i:i+batch_size]
            batch_patches = [extract_patch(image, coords) for coords in batch_coords]
            batch_tensor = torch.cat(batch_patches, dim=0)
            
            logits_batch = infer_patch_batch(batch_tensor, model, device)
            
            for j, coords in enumerate(batch_coords):
                patch_results.append({
                    'logits': logits_batch[j],
                    'coords': coords
                })
    
    # Clean up Dask client if created
    if dask_client is not None:
        try:
            dask_client.close()
        except:
            pass
    
    # Stitch patches back together
    output = stitch_patches(patch_results, image_shape, patch_size, sigma_scale)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Report memory usage and running time
    final_memory_mb = process.memory_info().rss / 1024 / 1024
    memory_used_mb = final_memory_mb - initial_memory_mb
    logger.info(f"Memory usage: Current={final_memory_mb:.2f} MB, Increase={memory_used_mb:.2f} MB")
    logger.info(f"Processing time: {elapsed_time:.2f} seconds")
    
    return output
