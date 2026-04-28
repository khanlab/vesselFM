<img src="docs/vesselfm_banner_updated.png">

**TL;DR**: VesselFM is a foundation model for universal 3D blood vessel segmentation. It is trained on three heterogeneous data sources: a large, curated annotated dataset, synthetic data generated through domain randomization, and data sampled from a flow matching-based deep generative model. These data sources provide enough diversity to enable vesselFM to achieve exceptional *zero*-shot blood vessel segmentation, even in completely unseen domains. For details, please refer to our [manuscript](https://openaccess.thecvf.com/content/CVPR2025/html/Wittmann_vesselFM_A_Foundation_Model_for_Universal_3D_Blood_Vessel_Segmentation_CVPR_2025_paper.html).

---


## 🟢 Installation
First, set up a conda environment and install dependencies:

    conda create -n vesselfm python=3.9

    conda activate vesselfm

    pip install -e .


## 🟢 *Zero*-Shot Segmentation
To run vesselFM's inference for *zero*-shot segmentation:

### Quick Start (CLI)
The easiest way is using the command-line interface:

    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output

Or, if you have installed the package:

    vesselfm-infer --input-folder /path/to/images --output-folder /path/to/output

**Dask Integration for Large Datasets**: VesselFM now includes built-in Dask support for parallel processing of large image datasets. Dask provides two types of parallelism:

1. **Multi-image parallelism** (default): Automatically parallelizes loading and preprocessing of multiple images
2. **Chunk-level parallelism** (new): Splits individual large images into chunks, processes them in parallel, and merges results with Gaussian blending

To control Dask behavior:

    # Disable Dask completely (use sequential processing)
    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --disable-dask

    # Specify number of workers (default: auto-detect based on CPU cores)
    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --dask-workers 4

    # Enable chunk-level parallelism for individual images (processes image chunks in parallel)
    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --enable-dask-chunking

    # Disable chunk-level parallelism (use traditional sliding window)
    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output --disable-dask-chunking

**Note**: Chunk-level parallelism is enabled by default in the configuration and is particularly beneficial for very large 3D volumes (>500MB).

### Advanced Usage (Config File)
For more control, adjust the [config file](vesselfm/seg/configs/inference.yaml) (see `#TODO`) and run:

    python vesselfm/seg/inference.py

Additional information on inference, pre-training, and fine-tuning are available [here](./vesselfm/seg). Checkpoints will be downloaded automatically and are also available on [Hugging Face 🤗](https://huggingface.co/bwittmann/vesselFM).


## 🟢 Fine-Tuning

VesselFM includes a CLI for fine-tuning the baseline model on your own paired image/mask data.

### Quick Start (CLI)

Prepare two folders that contain matching filenames — one for images and one for masks:

    vessels/
      images/  sample_001.nii.gz  sample_002.nii.gz  ...
      masks/   sample_001.nii.gz  sample_002.nii.gz  ...

Then run fine-tuning with pixi or directly:

    # with pixi
    pixi run vesselfm-finetune \
        --input-folder vessels/images \
        --mask-folder vessels/masks \
        --output-dir vessels/finetuned

    # or after package installation
    vesselfm-finetune \
        --input-folder vessels/images \
        --mask-folder vessels/masks \
        --output-dir vessels/finetuned

The baseline model (`vesselFM_base.pt`) is downloaded automatically from [Hugging Face 🤗](https://huggingface.co/bwittmann/vesselFM) if no `--checkpoint` is provided.
After training, inference-compatible weights are saved to `vessels/finetuned/vesselFM_finetuned.pt`.

### Using the Fine-Tuned Model for Inference

    vesselfm-infer \
        --input-folder /path/to/images \
        --output-folder /path/to/output \
        --checkpoint vessels/finetuned/vesselFM_finetuned.pt

### Key Options

| Argument | Default | Description |
|---|---|---|
| `--input-folder` | *(required)* | Folder with training image patches |
| `--mask-folder` | *(required)* | Folder with training masks (same filenames) |
| `--output-dir` | *(required)* | Directory for checkpoints and exported weights |
| `--checkpoint` | auto-download | Path to baseline `.pt` weights |
| `--val-input-folder` | — | Separate validation images (auto-split if omitted) |
| `--val-mask-folder` | — | Separate validation masks |
| `--val-split` | `0.2` | Fraction of training data used for validation |
| `--patch-size` | `128 128 128` | Spatial patch size D H W |
| `--batch-size` | `2` | Training batch size |
| `--lr` | `1e-5` | Learning rate |
| `--max-steps` | `1200` | Maximum training steps |
| `--num-shots` | all | Limit training to first N samples; `0` = zero-shot eval |
| `--device` | auto | Device, e.g. `cuda:0` or `cpu` |
| `--no-wandb` | — | Disable Weights & Biases logging |
| `--wandb-project` | `vesselfm` | W&B project name |
| `--wandb-offline` | — | Run W&B in offline mode |

For advanced configuration (custom datasets, multi-GPU, etc.) see [here](./vesselfm/seg).


## 🟢 Data Sources
<img src="docs/data_sources.png">

We also provide individual instructions for generating our three proposed data sources.

$\mathcal{D}_\text{drand}$: Domain randomized synthetic data ([here](./vesselfm/d_drand)).

$\mathcal{D}_\text{flow}$: Synthetic data sampled from our flow matching-based deep generative model ([here](./vesselfm/d_flow)).

$\mathcal{D}_\text{real}$: Real data curated from 17 annotated blood vessel segmentation datasets ([here](./vesselfm/d_real)).


## 🟢 Citing vesselFM
If you find our work useful for your research, please cite:

```bibtex
@InProceedings{Wittmann_2025_CVPR,
    author    = {Wittmann, Bastian and Wattenberg, Yannick and Amiranashvili, Tamaz and Shit, Suprosanna and Menze, Bjoern},
    title     = {vesselFM: A Foundation Model for Universal 3D Blood Vessel Segmentation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {20874-20884}
}
```

## 🟢 License
Code in this repository is licensed under [GNU General Public License v3.0](LICENSE). Model weights are released under [Open RAIL++-M License](https://huggingface.co/bwittmann/vesselFM/blob/main/LICENSE) and are restricted to research and non-commercial use only. Model use must comply with potential licenses, regulations, and restrictions arising from the use of named data sets during model training.
