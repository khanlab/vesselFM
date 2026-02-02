#!/usr/bin/env python
"""Command-line interface for vesselFM inference."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_overrides(args) -> list[str]:
    overrides = [
        f"image_path={str(args.input_folder)}",
        f"output_folder={str(args.output_folder)}",
    ]

    if args.mask_folder:
        overrides.append(f"mask_path={str(args.mask_folder)}")

    if args.device:
        overrides.append(f"device={args.device}")

    if args.batch_size is not None:
        overrides.append(f"batch_size={args.batch_size}")

    if args.patch_size is not None:
        # Hydra override syntax for lists
        overrides.append(f"patch_size=[{args.patch_size[0]},{args.patch_size[1]},{args.patch_size[2]}]")

    if args.overlap is not None:
        overrides.append(f"overlap={args.overlap}")

    if args.threshold is not None:
        overrides.append(f"merging.threshold={args.threshold}")

    if args.tta_scales is not None and len(args.tta_scales) > 0:
        scales_str = ",".join(str(s) for s in args.tta_scales)
        overrides.append(f"tta.scales=[{scales_str}]")

    if args.apply_postprocessing:
        overrides.append("post.apply=True")

    return overrides


def create_config(args):
    """
    Create a Hydra-composed configuration, then override from CLI args.

    Args:
        args (argparse.Namespace): Parsed command-line arguments used to build
            the Hydra overrides for inference (e.g., input/output folders,
            device, batch size, patch size, overlap, threshold, TTA scales,
            and post-processing flag).

    Returns:
        DictConfig: A configuration object compatible with ``hydra.utils.instantiate``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).parent / "seg" / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")

    overrides = build_overrides(args)

    # If create_config could be called multiple times in the same process (tests, notebooks),
    # Hydra needs to be cleared before re-initializing.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference", overrides=overrides)

    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Run vesselFM inference on 3D medical images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input-folder", type=Path, required=True, help="Path to folder containing input images")
    parser.add_argument("--output-folder", type=Path, required=True, help="Path to folder for saving output segmentations")

    parser.add_argument("--mask-folder", type=Path, default=None, help="Path to folder containing ground truth masks")
    parser.add_argument("--device", type=str, default=None, help="Device to use (e.g. 'cuda:0', 'cpu')")
    parser.add_argument("--batch-size", type=int, default=None, help="Sliding window batch size")
    parser.add_argument("--patch-size", type=int, nargs=3, default=None, help="Patch size D H W")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap (0.0-1.0)")
    parser.add_argument("--threshold", type=float, default=None, help="Segmentation threshold (0.0-1.0)")
    parser.add_argument("--tta-scales", type=float, nargs="+", default=None, help="TTA scales, e.g. 0.5 1.0 1.5")
    parser.add_argument("--apply-postprocessing", action="store_true", help="Apply post-processing")

    args = parser.parse_args()

    if not args.input_folder.exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)

    args.output_folder.mkdir(parents=True, exist_ok=True)

    cfg = create_config(args)

    from vesselfm.seg.inference import run_inference
    run_inference(cfg)


if __name__ == "__main__":
    main()



