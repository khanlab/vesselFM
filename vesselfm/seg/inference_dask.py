"""Dask-based inference module for vesselFM.

This module provides functions for running inference on dask arrays,
processing each chunk separately and returning a new dask array with
the inference results.
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import dask.array as da
from omegaconf import DictConfig

from vesselfm.seg.inference import load_model, resample
from vesselfm.seg.utils.data import generate_transforms
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def inference_chunk(
    image_chunk: np.ndarray,
    block_info: Optional[dict] = None,
    model: Optional[torch.nn.Module] = None,
    inferer: Optional[SlidingWindowInfererAdapt] = None,
    transforms=None,
    device: str = "cpu",
    cfg: Optional[DictConfig] = None,
) -> np.ndarray:
    """
    Run inference on a single chunk of the dask array.
    
    This function is called by dask.array.map_blocks for each chunk.
    
    Args:
        image_chunk: Numpy array chunk to process (can be 3D or 4D).
        block_info: Block information from dask (optional).
        model: PyTorch model for inference.
        inferer: MONAI sliding window inferer.
        transforms: Data transforms to apply.
        device: Device to run inference on ('cpu' or 'cuda').
        cfg: Configuration object with inference settings.
    
    Returns:
        Numpy array with inference results (same shape as input).
    """
    if model is None or inferer is None or transforms is None or cfg is None:
        raise ValueError("model, inferer, transforms, and cfg must be provided")
    
    # Handle both 3D and 4D input (add batch and channel dims if needed)
    if image_chunk.ndim == 3:
        # Add batch and channel dimensions
        has_batch_channel = False
        image_chunk_expanded = image_chunk[None, None, ...]
    elif image_chunk.ndim == 4:
        # Assume shape is (C, D, H, W) or (B, D, H, W)
        has_batch_channel = True
        if image_chunk.shape[0] == 1:
            # Assume (C, D, H, W) where C=1
            image_chunk_expanded = image_chunk[None, ...]  # Add batch dim
        else:
            # Assume already has batch dim
            image_chunk_expanded = image_chunk
    else:
        raise ValueError(f"Expected 3D or 4D input, got shape {image_chunk.shape}")
    
    # Convert to tensor and apply transforms
    image_tensor = torch.from_numpy(image_chunk_expanded.astype(np.float32)).to(device)
    
    # Apply transforms if needed (note: transforms expect specific format)
    # MONAI transforms may return either numpy arrays or torch tensors depending on configuration
    # We handle both cases for compatibility
    if image_tensor.shape[0] == 1:
        # Single image, apply transforms
        image_transformed = transforms(image_tensor.squeeze(0).cpu().numpy())
        # Convert to tensor if needed, add batch dim, and move to device
        image_transformed = torch.from_numpy(image_transformed) if isinstance(image_transformed, np.ndarray) else image_transformed
        image_transformed = image_transformed[None].to(device)
    else:
        # Multiple images in batch - process separately
        transformed_list = []
        for i in range(image_tensor.shape[0]):
            transformed = transforms(image_tensor[i].cpu().numpy())
            transformed = torch.from_numpy(transformed) if isinstance(transformed, np.ndarray) else transformed
            transformed_list.append(transformed[None])
        image_transformed = torch.cat(transformed_list, dim=0).to(device)
    
    # Apply test time augmentation if configured
    preds = []
    for scale in cfg.tta.scales:
        image_scaled = image_transformed.clone()
        
        # Apply TTA transformations
        if cfg.tta.invert:
            # Invert image intensities if mean intensity is above threshold
            if image_scaled.mean() > cfg.tta.invert_mean_thresh:
                image_scaled = 1 - image_scaled
        
        if cfg.tta.equalize_hist:
            # Apply histogram equalization
            image_np = image_scaled.cpu().squeeze().numpy()
            if image_np.ndim == 3:  # Single image
                image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                image_scaled = torch.from_numpy(image_equal_hist_np).to(device)[None][None]
            else:  # Batch of images
                equalized_list = []
                for i in range(image_np.shape[0]):
                    img_eq = equalize_hist(image_np[i], nbins=cfg.tta.hist_bins)
                    equalized_list.append(img_eq)
                image_equal_hist_np = np.stack(equalized_list, axis=0)
                image_scaled = torch.from_numpy(image_equal_hist_np).to(device)[:, None]
        
        # Resample if scale != 1
        original_shape = image_scaled.shape
        if scale != 1:
            image_scaled = resample(image_scaled, factor=scale)
        
        # Run inference
        with torch.no_grad():
            logits = inferer(image_scaled, model)
        
        # Resample back to original shape
        if scale != 1:
            logits = resample(logits, target_shape=original_shape)
        
        preds.append(logits.cpu())
    
    # Merge predictions from different scales
    if cfg.merging.max:
        pred = torch.stack(preds).max(dim=0)[0].sigmoid()
    else:
        pred = torch.stack(preds).mean(dim=0).sigmoid()
    
    # Apply threshold
    pred_thresh = (pred > cfg.merging.threshold).numpy()
    
    # Post-processing
    if cfg.post.apply:
        # Process each image in batch separately for post-processing
        if pred_thresh.ndim == 4:  # Batch
            processed = []
            for i in range(pred_thresh.shape[0]):
                img_processed = remove_small_objects(
                    pred_thresh[i, 0],
                    min_size=cfg.post.small_objects_min_size,
                    connectivity=cfg.post.small_objects_connectivity
                )
                processed.append(img_processed)
            pred_thresh = np.stack(processed, axis=0)[:, None]
        else:  # Single image
            pred_thresh = remove_small_objects(
                pred_thresh[0, 0],
                min_size=cfg.post.small_objects_min_size,
                connectivity=cfg.post.small_objects_connectivity
            )[None, None]
    
    # Remove batch/channel dims to match input shape
    if not has_batch_channel and pred_thresh.ndim == 4:
        # Remove batch and channel dimensions (B, C, D, H, W) -> (D, H, W)
        pred_thresh = pred_thresh[0, 0]
    elif pred_thresh.ndim == 4 and image_chunk.ndim == 4:
        # Keep as (C, D, H, W) or similar
        pred_thresh = pred_thresh.squeeze(0)  # Remove batch if present
    
    return pred_thresh.astype(np.uint8)


def run_inference_dask(
    dask_array: da.Array,
    cfg: DictConfig,
    model: Optional[torch.nn.Module] = None,
) -> da.Array:
    """
    Run inference on a dask array using map_blocks.
    
    This function processes each chunk of the dask array separately,
    running the vesselFM model inference on each chunk and returning
    a new dask array with the results.
    
    Args:
        dask_array: Input dask array with shape (D, H, W) or (C, D, H, W).
        cfg: Configuration object with model and inference settings.
            Should contain:
            - device: Device to run on ('cpu' or 'cuda')
            - model: Model configuration (if model not provided)
            - ckpt_path: Path to model checkpoint (if model not provided)
            - transforms_config: Transform configurations
            - patch_size: Sliding window patch size
            - batch_size: Sliding window batch size
            - overlap: Sliding window overlap
            - mode: Sliding window mode
            - sigma_scale: Sigma scale for blending
            - padding_mode: Padding mode
            - tta: Test-time augmentation settings
            - merging: Prediction merging settings
            - post: Post-processing settings
        model: Optional pre-loaded PyTorch model. If None, will load from cfg.
    
    Returns:
        Dask array with inference results (same shape as input, dtype uint8).
    
    Example:
        >>> import dask.array as da
        >>> from omegaconf import OmegaConf
        >>> 
        >>> # Create a sample dask array (e.g., from a large image file)
        >>> image = da.from_zarr('large_image.zarr')
        >>> 
        >>> # Load configuration
        >>> cfg = OmegaConf.load('config.yaml')
        >>> 
        >>> # Run inference
        >>> result = run_inference_dask(image, cfg)
        >>> 
        >>> # Compute result and save
        >>> result.to_zarr('segmentation_result.zarr')
    """
    # Set device
    device = cfg.device
    logger.info(f"Using device {device} for dask inference.")
    
    # Load model if not provided
    if model is None:
        logger.info("Loading model...")
        model = load_model(cfg, device)
        model.to(device)
    model.eval()
    
    # Initialize pre-processing transforms
    logger.info("Initializing transforms...")
    transforms = generate_transforms(cfg.transforms_config)
    
    # Initialize sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}")
    logger.debug(f"Sliding window overlap: {cfg.overlap}")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size,
        sw_batch_size=cfg.batch_size,
        overlap=cfg.overlap,
        mode=cfg.mode,
        sigma_scale=cfg.sigma_scale,
        padding_mode=cfg.padding_mode
    )
    
    # Use map_blocks to apply inference to each chunk
    logger.info(f"Setting up dask inference on array with shape {dask_array.shape}...")
    logger.info(f"Chunk size: {dask_array.chunksize}")
    
    result = da.map_blocks(
        inference_chunk,
        dask_array,
        model=model,
        inferer=inferer,
        transforms=transforms,
        device=device,
        cfg=cfg,
        dtype=np.uint8,
        drop_axis=[],  # Keep same dimensions
        new_axis=[],   # Don't add dimensions
    )
    
    logger.info("Dask inference graph created. Call .compute() to execute.")
    return result
