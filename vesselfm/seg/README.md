Config files are stored under [`configs`](./configs). Please adjust the placeholder paths and parameters marked with `#TODO`.

## Inference

### Option 1: CLI Tool (Recommended)
The easiest way to run inference is using the CLI tool:

    python -m vesselfm.cli --input-folder /path/to/images --output-folder /path/to/output

Or, if you have installed the package:

    vesselfm-infer --input-folder /path/to/images --output-folder /path/to/output

The CLI tool accepts the following arguments:
- `--input-folder`: Path to folder containing input images (required)
- `--output-folder`: Path to folder for saving output segmentations (required)
- `--mask-folder`: Path to folder containing ground truth masks for evaluation (optional)
- `--device`: Device to use (e.g., 'cuda:0', 'cpu'), default: 'cuda:0'
- `--batch-size`: Sliding window batch size (optional)
- `--patch-size`: Patch size for sliding window inference, 3 values for D, H, W (optional)
- `--overlap`: Overlap for sliding window inference (0.0-1.0) (optional)
- `--threshold`: Threshold for binary segmentation (0.0-1.0) (optional)
- `--tta-scales`: Test-time augmentation scales (e.g., 0.5 1.0 1.5) (optional)
- `--apply-postprocessing`: Apply post-processing to remove small objects (flag)

Example with custom parameters:

    python -m vesselfm.cli \
        --input-folder /path/to/images \
        --output-folder /path/to/output \
        --device cuda:0 \
        --batch-size 8 \
        --threshold 0.6 \
        --apply-postprocessing

### Option 2: Config File (Advanced)
Adjust the [inference](./configs/inference.yaml) config file (see `#TODO`) and run:

    python vesselfm/seg/inference.py

Images to be segmented should be placed in `/path/to/image_folder` as `.nii.gz` files. The results will be saved in `/path/to/output_folder`. Although we did not use them in our experiments, test-time augmentations (see `tta`) and post-processing steps (see `post`) can further improve the quality of the predicted segmentation mask. We have, therefore, included these features in the inference script as well. It is further strongly advised to adjust other inference parameters (e.g., `upper` and `lower` percentiles in `transforms_config`) to suit your data.

## Pre-Train on Three Data Sources
Adjust the [training](./configs/train.yaml) and [dataset](./configs/data/real_drand_flow.yaml) config files (see `#TODO`) and run:

    python vesselfm/seg/train.py

## Finetune and Evaluation (*Zero*-, *One*-, and *Few*-Shot)
Adjust the [finetune](./configs/finetune.yaml) and dataset ([BvEM](./configs/data/eval_bvem.yaml), [MSD8](./configs/data/eval_msd8.yaml), [OCTA](./configs/data/eval_octa.yaml), [SMILE-UHURA](./configs/data/eval_smile.yaml)) config files (see `#TODO`) and run:

    python vesselfm/seg/finetune.py data=<eval_smile/eval_octa/eval_msd8/eval_bvem> num_shots=<0/1/3>
