import torch
import torch.nn.functional as F
import hydra
import numpy as np
from monai.inferers import SlidingWindowInfererAdapt
from pathlib import Path
from vesselfm.seg.utils.data import generate_transforms
from zarrnii.plugins import hookimpl



import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


def create_config():
    """
    Create a Hydra-composed configuration.

    Returns:
        DictConfig: A configuration object compatible with ``hydra.utils.instantiate``.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = Path(__file__).parent.parent / "seg" / "configs"
    if not config_dir.exists():
        raise FileNotFoundError(f"Config dir not found: {config_dir}")

    # If create_config could be called multiple times in the same process (tests, notebooks),
    # Hydra needs to be cleared before re-initializing.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name="inference_zarrnii")

    return cfg

def load_model(cfg, device, model_path=None):
    """Load the vesselFM model checkpoint.

    Tries to load from *model_path* first (if provided), then falls back to
    ``cfg.ckpt_path``, and finally downloads from Hugging Face.

    Args:
        cfg: Hydra configuration object with ``model`` and ``ckpt_path`` fields.
        device: Torch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        model_path: Optional explicit path to a ``.pt`` checkpoint file.  When
            ``None`` the path is taken from *cfg.ckpt_path*.

    Returns:
        torch.nn.Module: Loaded model with weights applied (not yet moved to
        *device* or set to eval mode).
    """
    ckpt_path = model_path if model_path is not None else cfg.ckpt_path
    try:
        logger.info(f"Loading model from {ckpt_path}.")
        ckpt = torch.load(Path(ckpt_path), map_location=device, weights_only=True)
    except Exception as e:
        logger.warning(f"Failed to load model from {ckpt_path}: {e}. Falling back to Hugging Face.")
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml') # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)
    return model


class VesselFMPlugin:
    """ZarrNii segmentation plugin that runs vesselFM inference on each Dask block.

    Each Dask block (3-D numpy array) is independently pre-processed, run
    through the sliding-window inferer, and thresholded to produce a binary
    ``uint8`` result of the same spatial shape.

    All constructor parameters that are not explicitly provided fall back to
    the values in the bundled ``inference_zarrnii.yaml`` configuration file.
    """

    @hookimpl
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda:0",
        patch_size: Optional[List[int]] = None,
        sw_batch_size: Optional[int] = None,
        overlap: Optional[float] = None,
        mode: Optional[str] = None,
        sigma_scale: Optional[float] = None,
        padding_mode: Optional[str] = None,
        threshold: float = 0.5,
    ):
        """
        Args:
            model_path: Path to a ``.pt`` checkpoint file.  When ``None`` the
                path from the bundled config is tried first; if that also fails
                the checkpoint is downloaded from Hugging Face automatically.
            device: Torch device string, e.g. ``"cuda:0"`` or ``"cpu"``
                (default ``"cuda:0"``).
            patch_size: Spatial size of each sliding-window patch, e.g.
                ``[128, 128, 128]``.  Defaults to the value in the bundled
                config.
            sw_batch_size: Number of patches to process in a single forward
                pass.  Defaults to the value in the bundled config.
            overlap: Fractional overlap between adjacent patches in
                ``[0, 1)``.  Defaults to the value in the bundled config.
            mode: Blending mode for overlapping patches; either
                ``"constant"`` or ``"gaussian"``.  Defaults to the value in
                the bundled config.
            sigma_scale: Sigma value for Gaussian blending (only used when
                *mode* is ``"gaussian"``).  Defaults to the value in the
                bundled config.
            padding_mode: Padding strategy applied to image borders; e.g.
                ``"constant"``.  Defaults to the value in the bundled config.
            threshold: Sigmoid threshold for binarising logits (default
                ``0.5``).
        """

        cfg = create_config()

        # seed libraries
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)

        # set device
        logger.info(f"Using device {device}.")

        # load model and ckpt
        model = load_model(cfg, device, model_path=model_path)
        model.to(device)
        model.eval()

        # init pre-processing transforms
        transforms = generate_transforms(cfg.transforms_config)

        # resolve sliding window inferer parameters (explicit args override cfg)
        _patch_size = patch_size if patch_size is not None else cfg.patch_size
        _sw_batch_size = sw_batch_size if sw_batch_size is not None else cfg.batch_size
        _overlap = overlap if overlap is not None else cfg.overlap
        _mode = mode if mode is not None else cfg.mode
        _sigma_scale = sigma_scale if sigma_scale is not None else cfg.sigma_scale
        _padding_mode = padding_mode if padding_mode is not None else cfg.padding_mode

        # init sliding window inferer
        logger.debug(f"Sliding window patch size: {_patch_size}")
        logger.debug(f"Sliding window batch size: {_sw_batch_size}.")
        logger.debug(f"Sliding window overlap: {_overlap}.")
        inferer = SlidingWindowInfererAdapt(
            roi_size=_patch_size, sw_batch_size=_sw_batch_size, overlap=_overlap,
            mode=_mode, sigma_scale=_sigma_scale, padding_mode=_padding_mode
        )

        self.model = model
        self.device = device
        self.transforms = transforms
        self.inferer = inferer
        self.threshold = threshold


    @hookimpl
    def segment(
        self,
        image: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Run vesselFM inference on a single block from a Dask map_blocks call.

        ZarrNii data has shape ``(c, z, y, x)``.  Each block may therefore be
        either 3-D ``(z, y, x)`` or 4-D ``(c, z, y, x)`` depending on the chunk
        layout.  The method squeezes / restores the channel dimension as needed.

        Args:
            image: numpy array block (3-D or 4-D float32).
            metadata: Optional metadata dict passed by ZarrNii (unused).

        Returns:
            Binary segmentation as a ``uint8`` numpy array of the same shape.
        """
        # Squeeze channel dim for the inference pipeline, track original ndim
        squeeze_channel = image.ndim == 4
        if squeeze_channel:
            img_3d = image.squeeze(0)
        else:
            img_3d = image

        with torch.no_grad():
            img_tensor = self.transforms(img_3d.astype(np.float32))[None].to(
                self.device
            )
            logits = self.inferer(img_tensor, self.model)
            # Explicit batch/channel handling
            pred = (logits.cpu()[0,0].sigmoid() > self.threshold).numpy()

        result = pred.astype(np.uint8)

        # Restore channel dimension so the output shape matches the input
        if squeeze_channel:
            result = result[np.newaxis, ...]

        return result

    @hookimpl
    def segmentation_plugin_name(self) -> str:
        return "VesselFM"

    @hookimpl
    def segmentation_plugin_description(self) -> str:
        return "VesselFM vessel segmentation via sliding-window inference"
