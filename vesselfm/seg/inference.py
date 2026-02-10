""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist


from vesselfm.seg.utils.data import generate_transforms
from vesselfm.seg.utils.io import determine_reader_writer
from vesselfm.seg.utils.evaluation import Evaluator, calculate_mean_metrics
from vesselfm.seg.utils.dask_patching import process_image_with_dask_chunks


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=True)
    except:
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)
    return model

def get_paths(cfg):
    image_paths = list(Path(cfg.image_path).iterdir())
    if cfg.mask_path:
        mask_paths = [Path(cfg.mask_path) / f"{p.name}" for p in image_paths]
        assert all(
            mask_path.exists() for mask_path in mask_paths
        ), "All mask paths must exist mask name has to be the same as the image name."
    else:
        mask_paths = None
    return image_paths, mask_paths

def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image
    
    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def process_image(image_path, image_data, mask_data, cfg, model, transforms, save_writer, inferer, file_ending, device, output_folder):
    """
    Process a single image through the inference pipeline.
    
    Args:
        image_path: Path to the input image
        image_data: Loaded image data as numpy array
        mask_data: Loaded mask data as numpy array (or None)
        cfg: Configuration object
        model: The neural network model
        transforms: Pre-processing transforms
        save_writer: Writer for saving predictions
        inferer: Sliding window inferer (can be None if using Dask chunking)
        file_ending: File extension for output files
        device: Device to use for computation
        output_folder: Path to output folder
        
    Returns:
        dict: Dictionary containing image name and metrics (if mask_data is provided)
    """
    image_name = image_path.name.split('.')[0]
    
    # Check if we should use Dask-based chunking for this image
    # Note: Chunking requires both Dask to be enabled and chunk_images to be True
    use_dask_chunks = cfg.dask.get("enabled", True) and cfg.dask.get("chunk_images", False)
    
    preds = []  # average over test time augmentations
    with torch.no_grad():
        for scale in cfg.tta.scales:
            # apply pre-processing transforms
            image = transforms(image_data)[None].to(device)
            mask = torch.tensor(mask_data).bool() if mask_data is not None else None

            # apply test time augmentation
            if cfg.tta.invert:
                image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image
                
            if cfg.tta.equalize_hist:
                image_np = image.cpu().squeeze().numpy()
                image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

            original_shape = image.shape
            image = resample(image, factor=scale)
            
            # Choose inference method based on configuration
            if use_dask_chunks:
                # Use Dask-based parallel chunking
                logger.debug(f"Using Dask-based chunking for {image_name}")
                logits = process_image_with_dask_chunks(
                    image=image,
                    model=model,
                    device=device,
                    patch_size=cfg.patch_size,
                    overlap=cfg.overlap,
                    batch_size=cfg.batch_size,
                    sigma_scale=cfg.sigma_scale,
                    use_dask=True,
                    n_workers=cfg.dask.get("n_workers", None)
                )
            else:
                # Use traditional sliding window inferer
                logits = inferer(image, model)
            
            logits = resample(logits, target_shape=original_shape)
            preds.append(logits.cpu().squeeze())

        # merging
        if cfg.merging.max:
            pred = torch.stack(preds).max(dim=0)[0].sigmoid()
        else:
            pred = torch.stack(preds).mean(dim=0).sigmoid()
        pred_thresh = (pred > cfg.merging.threshold).numpy()

        # post-processing
        if cfg.post.apply:
            pred_thresh = remove_small_objects(
                pred_thresh, min_size=cfg.post.small_objects_min_size, connectivity=cfg.post.small_objects_connectivity
            )

        # save final pred
        save_writer.write_seg(
            pred_thresh.astype(np.uint8), output_folder / f"{image_name}_{cfg.file_app}pred.{file_ending}"
        )

        result = {"image_name": image_name}
        
        if mask_data is not None:
            metrics = Evaluator().estimate_metrics(pred, mask, threshold=cfg.merging.threshold)  # no post-processing
            logger.info(f"Dice of {image_name}: {metrics['dice'].item()}")
            logger.info(f"clDice of {image_name}: {metrics['cldice'].item()}")
            result["metrics"] = metrics
            
        return result


def run_inference(cfg):
    """
    Run inference with the given configuration.
    
    Args:
        cfg: Configuration object (can be from Hydra or programmatically created).
    
    Returns:
        None: This function performs inference, writes outputs to disk, and logs results.
            If no images are found in ``cfg.image_path``, it logs an error message and
            returns early without performing inference.
    """
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # Check for images first before loading model
    image_paths, mask_paths = get_paths(cfg)
    logger.info(f"Found {len(image_paths)} images in {cfg.image_path}.")
    
    if not image_paths:
        logger.error(f"No images found in {cfg.image_path}. Please ensure the folder contains image files.")
        return

    # load model and ckpt
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    file_ending = (cfg.image_file_ending if cfg.image_file_ending else image_paths[0].suffix)
    image_reader_writer = determine_reader_writer(file_ending)()
    save_writer = determine_reader_writer(file_ending)()

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap, 
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    # Check if Dask is enabled
    use_dask = cfg.dask.get("enabled", True)
    
    if use_dask and len(image_paths) > 1:
        logger.info("Using Dask for parallel image loading and pre-processing")
        import dask
        import dask.bag as db
        from dask.diagnostics import ProgressBar
        
        # Configure Dask to use threads for I/O bound tasks
        dask.config.set(scheduler='threads')
        
        # Determine number of partitions based on config or image count
        n_workers = cfg.dask.get("n_workers", None)
        if n_workers:
            npartitions = min(len(image_paths), n_workers)
        else:
            # Auto-detect: use reasonable default based on CPU count and image count
            import os
            cpu_count = os.cpu_count() or 4
            npartitions = min(len(image_paths), cpu_count)
        
        # Create a function to load and preprocess a single image
        def load_and_preprocess(args):
            idx, image_path = args
            mask_path = mask_paths[idx] if mask_paths else None
            
            # Load image
            image_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            mask_data = image_reader_writer.read_images(mask_path)[0] if mask_path else None
            
            return {
                'idx': idx,
                'image_path': image_path,
                'image_data': image_data,
                'mask_data': mask_data
            }
        
        # Create Dask bag for parallel loading
        logger.info(f"Loading images in parallel with Dask using {npartitions} partitions...")
        image_args = list(enumerate(image_paths))
        bag = db.from_sequence(image_args, npartitions=npartitions)
        
        # Load images in parallel
        with ProgressBar():
            loaded_data = bag.map(load_and_preprocess).compute()
        
        # Process each loaded image with the model (sequential for GPU processing)
        metrics_dict = {}
        for data in tqdm(loaded_data, desc="Processing images"):
            result = process_image(
                data['image_path'],
                data['image_data'],
                data['mask_data'],
                cfg,
                model,
                transforms,
                save_writer,
                inferer,
                file_ending,
                device,
                output_folder
            )
            if "metrics" in result:
                metrics_dict[result["image_name"]] = result["metrics"]
    else:
        if use_dask:
            logger.info("Dask enabled but only 1 image found, using sequential processing")
        else:
            logger.info("Using sequential processing (Dask disabled)")
        
        # Sequential processing
        metrics_dict = {}
        for idx, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images."):
            # Load image and mask
            image_data = image_reader_writer.read_images(image_path)[0].astype(np.float32)
            mask_data = image_reader_writer.read_images(mask_paths[idx])[0] if mask_paths else None
            
            # Process the image
            result = process_image(
                image_path,
                image_data,
                mask_data,
                cfg,
                model,
                transforms,
                save_writer,
                inferer,
                file_ending,
                device,
                output_folder
            )
            if "metrics" in result:
                metrics_dict[result["image_name"]] = result["metrics"]

    if mask_paths is not None and metrics_dict:
        mean_metrics = calculate_mean_metrics(list(metrics_dict.values()), round_to=cfg.round_to)
        logger.info(f"Mean metrics: dice {mean_metrics['dice'].item()}, cldice {mean_metrics['cldice'].item()}")
    logger.info("Done.")


@hydra.main(config_path="configs", config_name="inference", version_base="1.3.2")
def main(cfg):
    """Main entry point when using Hydra configuration."""
    run_inference(cfg)


if __name__ == "__main__":
    main()