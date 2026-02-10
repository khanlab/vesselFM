"""Dask-based image patching and inference utilities."""

import logging
from typing import Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CPU_COUNT = 4  # Fallback when os.cpu_count() returns None
EPSILON = 1e-8  # Small value to prevent division by zero in normalization


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


def process_image_with_dask_chunks(
    image: torch.Tensor,
    model: torch.nn.Module,
    device: str,
    patch_size: Tuple[int, int, int],
    overlap: float,
    batch_size: int,
    sigma_scale: float = 0.125,
    use_dask: bool = True,
    n_workers: int = None
) -> torch.Tensor:
    """
    Process an image using Dask-based chunking for parallel inference.
    
    Args:
        image: Input image tensor (1, 1, D, H, W)
        model: PyTorch model
        device: Device for inference
        patch_size: Size of each patch (D, H, W)
        overlap: Overlap ratio between patches
        batch_size: Batch size for patch processing
        sigma_scale: Gaussian blending scale
        use_dask: Whether to use Dask for parallel processing
        n_workers: Number of Dask workers (None = auto)
        
    Returns:
        Output logits tensor
    """
    image_shape = image.shape
    
    # Compute patch coordinates
    coords_list = compute_patch_coords(image_shape, patch_size, overlap)
    logger.debug(f"Generated {len(coords_list)} patches for image of shape {image_shape}")
    
    if use_dask and len(coords_list) > 1:
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
            import os
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
    
    # Stitch patches back together
    output = stitch_patches(patch_results, image_shape, patch_size, sigma_scale)
    
    return output
