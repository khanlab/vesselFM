#!/usr/bin/env python
"""Command-line interface for vesselFM fine-tuning."""

import argparse
import functools
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _build_model():
    """Build the default DynUNet model matching dyn_unet_base.yaml."""
    from monai.networks.nets import DynUNet

    return DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[[3, 3, 3]] * 6,
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        upsample_kernel_size=[[2, 2, 2]] * 5,
        filters=[32, 64, 128, 256, 320, 320],
        res_block=True,
    )


def _load_baseline_model(checkpoint_path, device):
    """Load baseline model weights, downloading from HuggingFace if needed."""
    from huggingface_hub import hf_hub_download

    model = _build_model()

    if checkpoint_path is not None:
        logger.info(f"Loading baseline model from {checkpoint_path}.")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    else:
        logger.info("Downloading baseline model from Hugging Face.")
        hf_hub_download(repo_id="bwittmann/vesselFM", filename="meta.yaml")
        ckpt = torch.load(
            hf_hub_download(repo_id="bwittmann/vesselFM", filename="vesselFM_base.pt"),
            map_location=device,
            weights_only=True,
        )

    # Handle both plain state dict and Lightning checkpoint formats.
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = {
            k.replace("model.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    return model


def _detect_file_ending(path: Path) -> str:
    """Detect the file ending for a given path, handling compound extensions."""
    name = path.name
    if name.endswith(".ome.zarr"):
        return "ome.zarr"
    if name.endswith(".nii.gz"):
        return "nii.gz"
    return name.split(".")[-1]


def _build_transforms(patch_size: list, mode: str = "train"):
    """Build MONAI dictionary transforms for training or validation."""
    from monai.transforms import (
        CenterSpatialCropd,
        Compose,
        EnsureChannelFirstd,
        RandAffined,
        RandFlipd,
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandHistogramShiftd,
        RandZoomd,
        ScaleIntensityRangePercentilesd,
        SpatialPadd,
        ToTensord,
    )

    if mode == "train":
        return Compose(
            [
                EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
                CenterSpatialCropd(keys=["Image", "Mask"], roi_size=patch_size),
                ScaleIntensityRangePercentilesd(
                    keys=["Image"], lower=1, upper=99, b_min=0, b_max=1
                ),
                RandZoomd(
                    keys=["Image", "Mask"],
                    prob=0.3,
                    min_zoom=1,
                    max_zoom=1.3,
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                ),
                SpatialPadd(
                    keys=["Image", "Mask"],
                    spatial_size=patch_size,
                    mode="reflect",
                    method="symmetric",
                ),
                RandAffined(
                    keys=["Image", "Mask"],
                    prob=0.3,
                    shear_range=[0.4, 0.4, 0.4],
                    mode=["bilinear", "nearest"],
                    padding_mode="zeros",
                ),
                RandFlipd(keys=["Image", "Mask"], prob=0.3, spatial_axis=0),
                RandFlipd(keys=["Image", "Mask"], prob=0.3, spatial_axis=1),
                RandGaussianSmoothd(
                    keys=["Image"],
                    prob=0.2,
                    sigma_x=[0.0, 0.5],
                    sigma_y=[0.0, 0.5],
                    sigma_z=[0.0, 0.5],
                ),
                RandGaussianNoised(keys=["Image"], prob=0.2, std=0.01, mean=0.3),
                RandHistogramShiftd(
                    keys=["Image"], prob=0.3, num_control_points=[5, 10]
                ),
                CenterSpatialCropd(keys=["Image", "Mask"], roi_size=patch_size),
                ToTensord(keys=["Image", "Mask"]),
            ]
        )
    else:
        return Compose(
            [
                EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
                ScaleIntensityRangePercentilesd(
                    keys=["Image"], lower=1, upper=99, b_min=0, b_max=1
                ),
                ToTensord(keys=["Image", "Mask"]),
            ]
        )


class FlatFolderDataset(Dataset):
    """Dataset for paired image/mask files stored in two flat folders.

    Files in *input_folder* and *mask_folder* must share identical filenames.
    Each item in the dataset is a ``(image_tensor, mask_tensor)`` pair.
    """

    def __init__(self, input_folder: Path, mask_folder: Path, transforms):
        from vesselfm.seg.utils.io import determine_reader_writer

        input_paths = sorted(p for p in Path(input_folder).iterdir() if p.is_file())
        if not input_paths:
            raise FileNotFoundError(f"No files found in {input_folder}")

        mask_paths = [Path(mask_folder) / p.name for p in input_paths]
        missing = [str(p) for p in mask_paths if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Mask files not found (expected same names as images): "
                + ", ".join(missing[:5])
            )

        self.image_paths = input_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

        file_ending = _detect_file_ending(input_paths[0])
        self.reader = determine_reader_writer(file_ending)()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = self.reader.read_images(str(self.image_paths[idx]))[0].astype(np.float32)
        mask = self.reader.read_images(str(self.mask_paths[idx]))[0].astype(bool)
        transformed = self.transforms({"Image": img, "Mask": mask})
        return transformed["Image"], transformed["Mask"] > 0


def _build_lightning_module(model, patch_size: tuple, batch_size: int, lr: float, dataset_name: str):
    """Build a PLModuleFinetune with standard fine-tuning settings."""
    from monai.losses import DiceCELoss

    from vesselfm.seg.module import PLModuleFinetune
    from vesselfm.seg.utils.evaluation import Evaluator

    loss = DiceCELoss(
        sigmoid=True,
        reduction="mean",
        squared_pred=True,
        lambda_ce=0.1,
        lambda_dice=0.9,
    )
    optimizer_factory = functools.partial(
        torch.optim.AdamW, lr=lr, weight_decay=0.1, betas=(0.9, 0.999)
    )
    scheduler_configs = {
        "cosine_annealing_few": {
            "interval": "step",
            "frequency": 10,
            "scheduler": functools.partial(
                torch.optim.lr_scheduler.CosineAnnealingLR,
                T_max=1000,
                eta_min=1e-7,
                last_epoch=-1,
            ),
        },
        "linear_warmup": None,
        "cosine_annealing": None,
    }
    return PLModuleFinetune(
        model=model,
        loss=loss,
        optimizer_factory=optimizer_factory,
        prediction_threshold=0.62,
        scheduler_configs=scheduler_configs,
        evaluator=Evaluator(),
        dataset_name=dataset_name,
        input_size=patch_size,
        batch_size=batch_size,
    )


def run_finetune(args):
    """Execute the full fine-tuning pipeline."""
    import lightning as L
    from lightning.pytorch import seed_everything
    from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision("medium")

    # Resolve device
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    device_idx = int(device.split(":")[-1]) if ":" in device else 0

    # ── Gather all file paths ────────────────────────────────────────────────
    all_image_paths = sorted(p for p in args.input_folder.iterdir() if p.is_file())
    if not all_image_paths:
        logger.error(f"No files found in {args.input_folder}")
        sys.exit(1)

    n_total = len(all_image_paths)

    # ── Build train / val indices ────────────────────────────────────────────
    val_from_separate = args.val_input_folder is not None and args.val_mask_folder is not None

    if args.num_shots == 0:
        # Zero-shot: no training, evaluate on all provided data
        train_indices = []
        val_indices = None if val_from_separate else list(range(n_total))
    else:
        n_train_max = min(args.num_shots, n_total) if args.num_shots is not None else n_total
        if val_from_separate:
            train_indices = list(range(n_train_max))
            val_indices = None
        else:
            n_val = max(1, int(n_train_max * args.val_split)) if (args.val_split > 0 and n_train_max > 1) else 0
            if n_val == 0:
                logger.warning(
                    "No validation split possible with the current settings "
                    "(too few samples or --val-split 0). "
                    "Training data will be reused for validation — metrics will NOT reflect "
                    "generalisation. Provide --val-input-folder / --val-mask-folder for reliable evaluation."
                )
                train_indices = list(range(n_train_max))
                val_indices = list(range(n_train_max))  # reuse train as fallback
            else:
                train_indices = list(range(n_train_max - n_val))
                val_indices = list(range(n_train_max - n_val, n_train_max))

    # ── Build transforms ─────────────────────────────────────────────────────
    patch_size = list(args.patch_size)
    train_transforms = _build_transforms(patch_size, mode="train")
    val_transforms = _build_transforms(patch_size, mode="val")

    # ── Build datasets ───────────────────────────────────────────────────────
    train_base = FlatFolderDataset(args.input_folder, args.mask_folder, train_transforms)
    train_dataset = Subset(train_base, train_indices)

    if val_from_separate:
        val_dataset = FlatFolderDataset(
            args.val_input_folder, args.val_mask_folder, val_transforms
        )
    else:
        val_base = FlatFolderDataset(args.input_folder, args.mask_folder, val_transforms)
        val_dataset = Subset(val_base, val_indices)

    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # ── Build data loaders ───────────────────────────────────────────────────
    # Draw enough samples to cover max_steps with a comfortable margin.
    num_train_samples = max(args.max_steps * args.batch_size, len(train_dataset))
    random_sampler = RandomSampler(train_dataset, replacement=True, num_samples=num_train_samples)
    n_train_workers = 4 if train_indices else 0
    train_loader = DataLoader(
        train_dataset,
        sampler=random_sampler,
        batch_size=args.batch_size,
        num_workers=n_train_workers,
        pin_memory=(accelerator == "gpu"),
        persistent_workers=(n_train_workers > 0),
    )
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=2, shuffle=False)

    # ── Load baseline model ──────────────────────────────────────────────────
    model = _load_baseline_model(args.checkpoint, device)

    # ── Build Lightning module ───────────────────────────────────────────────
    dataset_name = args.input_folder.name
    lightning_module = _build_lightning_module(
        model, tuple(patch_size), args.batch_size, args.lr, dataset_name
    )

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Callbacks ────────────────────────────────────────────────────────────
    run_name = f"finetune_{dataset_name}"
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(output_dir / "checkpoints"),
        monitor="val_DiceMetric",
        save_top_k=1,
        mode="max",
        filename=f"{run_name}_best",
        save_last=True,
    )
    checkpoint_callback.CHECKPOINT_EQUALS_CHAR = ":"
    checkpoint_callback.CHECKPOINT_NAME_LAST = f"{run_name}_last"

    callbacks = [LearningRateMonitor(), checkpoint_callback]

    # ── Logger ───────────────────────────────────────────────────────────────
    trainer_kwargs = {}
    if not args.no_wandb:
        from lightning.pytorch.loggers import WandbLogger

        wnb_logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            offline=args.wandb_offline,
        )
        wnb_logger.watch(model, log="all", log_freq=20)
        trainer_kwargs["logger"] = wnb_logger

    # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=[device_idx] if accelerator == "gpu" else 1,
        max_steps=args.max_steps,
        max_epochs=1,
        val_check_interval=min(args.val_check_interval, args.max_steps),
        precision="16-mixed" if accelerator == "gpu" else "32",
        callbacks=callbacks,
        enable_checkpointing=True,
        log_every_n_steps=10,
        **trainer_kwargs,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.num_shots == 0:
        logger.info("num_shots=0: running zero-shot evaluation only (no training).")
        trainer.validate(lightning_module, val_loader)
    else:
        logger.info("Starting fine-tuning…")
        trainer.validate(lightning_module, val_loader)
        trainer.fit(lightning_module, train_loader, val_loader)
        logger.info("Fine-tuning complete.")

    # ── Export inference-compatible weights ───────────────────────────────────
    best_path = checkpoint_callback.best_model_path
    if best_path:
        logger.info(f"Loading best checkpoint: {best_path}")
        best_ckpt = torch.load(best_path, map_location="cpu", weights_only=True)
        state_dict = {
            k.replace("model.", ""): v
            for k, v in best_ckpt["state_dict"].items()
            if k.startswith("model.")
        }
    else:
        logger.warning("No best checkpoint found; exporting current model weights.")
        state_dict = model.state_dict()

    weights_path = output_dir / "vesselFM_finetuned.pt"
    torch.save(state_dict, weights_path)
    logger.info(f"Saved fine-tuned model weights to {weights_path}")
    logger.info(
        f"Use with inference CLI: vesselfm-infer --input-folder <images> "
        f"--output-folder <out> --checkpoint {weights_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune vesselFM on custom paired image/mask data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required I/O ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Folder containing training image patches (nii.gz, npy, …)",
    )
    parser.add_argument(
        "--mask-folder",
        type=Path,
        required=True,
        help="Folder containing training mask files (same filenames as images)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the fine-tuned model and Lightning checkpoints",
    )

    # ── Optional validation folders ──────────────────────────────────────────
    parser.add_argument(
        "--val-input-folder",
        type=Path,
        default=None,
        help="Folder with validation image patches (optional; auto-split used if absent)",
    )
    parser.add_argument(
        "--val-mask-folder",
        type=Path,
        default=None,
        help="Folder with validation mask files (optional; auto-split used if absent)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of training data reserved for validation when val folders are not provided (0.0–1.0)",
    )

    # ── Baseline model ───────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to baseline model weights (.pt). Downloads vesselFM_base.pt from HuggingFace if omitted.",
    )

    # ── Architecture ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=[128, 128, 128],
        metavar=("D", "H", "W"),
        help="Spatial patch size (depth height width) used during training",
    )

    # ── Training hyper-parameters ─────────────────────────────────────────────
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--max-steps", type=int, default=1200, help="Maximum number of training steps"
    )
    parser.add_argument(
        "--val-check-interval",
        type=int,
        default=200,
        help="Number of optimizer steps between validation runs",
    )
    parser.add_argument(
        "--num-shots",
        type=int,
        default=None,
        help="Limit training to the first N samples (None = use all). Use 0 for zero-shot evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. 'cuda:0' or 'cpu'. Auto-detected if not provided.",
    )

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="vesselfm",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb-offline",
        action="store_true",
        help="Run Weights & Biases in offline mode",
    )

    args = parser.parse_args()

    # ── Validate paths ────────────────────────────────────────────────────────
    if not args.input_folder.exists():
        logger.error(f"Input folder does not exist: {args.input_folder}")
        sys.exit(1)
    if not args.mask_folder.exists():
        logger.error(f"Mask folder does not exist: {args.mask_folder}")
        sys.exit(1)
    if args.val_input_folder is not None and not args.val_input_folder.exists():
        logger.error(f"Validation input folder does not exist: {args.val_input_folder}")
        sys.exit(1)
    if args.val_mask_folder is not None and not args.val_mask_folder.exists():
        logger.error(f"Validation mask folder does not exist: {args.val_mask_folder}")
        sys.exit(1)

    run_finetune(args)


if __name__ == "__main__":
    main()
