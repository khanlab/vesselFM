#!/usr/bin/env python
"""Command-line interface for vesselFM inference."""

import argparse
import logging
import sys
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_config(args):
    """
    Create a configuration object from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        OmegaConf configuration object
    """
    # Import here to avoid loading heavy dependencies until needed
    from omegaconf import OmegaConf
    
    # Load default config
    default_config_path = Path(__file__).parent / "seg" / "configs" / "inference.yaml"
    cfg = OmegaConf.load(default_config_path)
    
    # Override with command-line arguments
    cfg.image_path = str(args.input_folder)
    cfg.output_folder = str(args.output_folder)
    
    if args.mask_folder:
        cfg.mask_path = str(args.mask_folder)
    
    if args.device:
        cfg.device = args.device
    
    if args.batch_size:
        cfg.batch_size = args.batch_size
    
    if args.patch_size:
        cfg.patch_size = args.patch_size
    
    if args.overlap is not None:
        cfg.overlap = args.overlap
    
    if args.threshold is not None:
        cfg.merging.threshold = args.threshold
    
    if args.tta_scales:
        cfg.tta.scales = args.tta_scales
    
    if args.apply_postprocessing:
        cfg.post.apply = True
    
    return cfg


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Run vesselFM inference on 3D medical images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Path to folder containing input images"
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        required=True,
        help="Path to folder for saving output segmentations"
    )
    
    # Optional arguments
    parser.add_argument(
        "--mask-folder",
        type=Path,
        default=None,
        help="Path to folder containing ground truth masks for evaluation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (e.g., 'cuda:0', 'cpu')"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Sliding window batch size"
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=None,
        help="Patch size for sliding window inference (3 values for D, H, W)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=None,
        help="Overlap for sliding window inference (0.0-1.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary segmentation (0.0-1.0)"
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=None,
        help="Test-time augmentation scales (e.g., 0.5 1.0 1.5)"
    )
    parser.add_argument(
        "--apply-postprocessing",
        action="store_true",
        help="Apply post-processing to remove small objects"
    )
    
    args = parser.parse_args()
    
    # Validate input folder exists
    if not args.input_folder.exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)
    
    # Create output folder if it doesn't exist
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create config from arguments
    cfg = create_config(args)
    
    # Import inference module here (after argument parsing) to avoid loading 
    # heavy dependencies when just showing help
    from vesselfm.seg.inference import run_inference
    
    # Run inference with the created config
    run_inference(cfg)


if __name__ == "__main__":
    main()
