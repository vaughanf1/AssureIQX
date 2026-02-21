# AssureXRay

Reproducible 3-class bone tumor classification (Normal / Benign / Malignant) on BTXRD radiographs with Grad-CAM explainability.

> **NOT FOR CLINICAL USE -- Research Prototype Only**
>
> This model has not been validated for diagnostic purposes.
> It is intended solely as a proof-of-concept for bone tumor radiograph classification research.

## Overview

AssureXRay is a proof-of-concept that trains an EfficientNet-B0 classifier on the
[BTXRD dataset](https://figshare.com/articles/dataset/BTXRD/22813073) (3,746 bone
tumor radiographs from 3 medical centers) and produces a complete evaluation pack
with clinician-facing Grad-CAM heatmaps.

- **What:** 3-class bone tumor classification PoC (Normal / Benign / Malignant)
- **Dataset:** BTXRD -- 3,746 radiographs from 3 centers (1,879 Normal, 1,525 Benign, 342 Malignant)
- **Model:** EfficientNet-B0 with ImageNet transfer learning, inverse-frequency weighted loss
- **Splits:** Dual strategy -- stratified random (70/15/15) and center-holdout (Center 1 train, Centers 2+3 test)
- **Explainability:** Grad-CAM heatmaps with annotation comparison
- **Key result:** Stratified macro AUC 0.846, center-holdout macro AUC 0.627

For full documentation see [`docs/model_card.md`](docs/model_card.md) and [`docs/poc_report.md`](docs/poc_report.md).

## Key Results

| Split | Macro AUC | Malignant Sensitivity | Malignant Specificity |
|-------|-----------|----------------------|----------------------|
| Stratified | 0.846 (0.814--0.873) | 60.8% (47.4%--74.3%) | 95.7% |
| Center-holdout | 0.627 (0.594--0.658) | 36.4% (27.0%--44.8%) | 79.9% |

The generalization gap (center-holdout minus stratified) is -0.219 macro AUC and -24.3 pp
malignant sensitivity, reflecting distribution shift between training centers and unseen
centers. See [`docs/poc_report.md`](docs/poc_report.md) for full analysis and
[`docs/model_card.md`](docs/model_card.md) for model documentation.

## Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip
- ~2 GB disk space for the BTXRD dataset
- (Optional) NVIDIA GPU with CUDA 12.1 for faster training

### Installation

```bash
# Clone the repository
git clone <repo-url> AssureXRay
cd AssureXRay

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies (CPU)
pip install -r requirements.txt

# Install dependencies (GPU with CUDA 12.1)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Verify Installation

```bash
python -c "import torch; import timm; import albumentations; print('All dependencies installed')"
```

## Quick Start

Run the entire pipeline end-to-end with a single command:

```bash
make all
```

Or run each stage individually:

```bash
make download    # Download BTXRD dataset from figshare (~2 GB)
make audit       # Profile dataset and generate audit report
make split       # Create train/val/test split manifests
make train-all   # Train on both stratified and center-holdout splits
make evaluate    # Evaluate models with full metric suite
make gradcam     # Generate Grad-CAM heatmap gallery
make report      # Verify documentation is complete
```

## CLI Reference

All pipeline stages are accessible via `make` targets. Each target runs an underlying
Python script that accepts `--config` and `--override` arguments.

### Download Dataset

```bash
make download
# Equivalent to:
python scripts/download.py --config configs/default.yaml
```

Downloads the BTXRD dataset from figshare, extracts images into `data_raw/images/`,
annotations into `data_raw/annotations/`, and the labels file as `data_raw/dataset.csv`.

### Audit Dataset

```bash
make audit
# Equivalent to:
python scripts/audit.py --config configs/default.yaml
```

Profiles dataset quality and generates `docs/data_audit_report.md` with class distribution,
image dimensions, per-center breakdowns, duplicate detection, and annotation coverage.

### Create Splits

```bash
make split
# Equivalent to:
python scripts/split.py --config configs/default.yaml
```

Generates train/val/test CSV manifests in `data/splits/` for both strategies:
- `stratified_{train,val,test}.csv` -- 70/15/15 image-level stratified split
- `center_{train,val,test}.csv` -- Center 1 train/val, Centers 2+3 test

Duplicate-aware grouping ensures same-lesion images stay in the same partition.

### Train Model

```bash
# Train on both split strategies
make train-all
# Equivalent to:
python scripts/train.py --config configs/default.yaml --override training.split_strategy=stratified
python scripts/train.py --config configs/default.yaml --override training.split_strategy=center

# Train on a single split
make train
# Equivalent to:
python scripts/train.py --config configs/default.yaml
```

Trains EfficientNet-B0 with ImageNet pretrained weights, inverse-frequency weighted
cross-entropy loss, cosine LR scheduler, and early stopping (patience=7). Saves best
checkpoint to `checkpoints/best_{strategy}.pt`.

### Evaluate Model

```bash
make evaluate
# Equivalent to:
python scripts/eval.py --config configs/default.yaml
```

Runs evaluation on both split strategies, producing per-split results in
`results/stratified/` and `results/center_holdout/`:
- ROC curves (one-vs-rest per class + macro AUC)
- PR curves (per class)
- Confusion matrices (absolute + row-normalized)
- Classification report (precision, recall, F1 per class)
- Bootstrap 95% confidence intervals (1000 iterations)
- Comparison table (`results/comparison_table.json`)

### Generate Grad-CAM Heatmaps

```bash
make gradcam
# Equivalent to:
python scripts/gradcam.py --config configs/default.yaml
```

Generates Grad-CAM heatmap overlays for selected examples (TP, FP, FN per class)
and compares attention regions against LabelMe tumor annotations. Output saved to
`results/gradcam/`.

### Run Inference

```bash
# Single image
python scripts/infer.py --image path/to/image.jpg --checkpoint checkpoints/best_stratified.pt

# Batch inference on a directory
python scripts/infer.py --input-dir path/to/images/ --checkpoint checkpoints/best_stratified.pt

# With Grad-CAM overlay
python scripts/infer.py --image path/to/image.jpg --checkpoint checkpoints/best_stratified.pt --gradcam

# Make target (uses default config)
make infer
```

Outputs class prediction, softmax confidence scores, and optional Grad-CAM overlay PNG.

### Verify Documentation

```bash
make report
```

Verifies that `docs/model_card.md` and `docs/poc_report.md` exist and reports line counts.

## Project Structure

```
AssureXRay/
|-- configs/
|   |-- default.yaml          # Central configuration (hyperparams, paths, seeds)
|-- scripts/
|   |-- download.py            # Dataset download from figshare
|   |-- audit.py               # Dataset profiling and audit report
|   |-- split.py               # Train/val/test split generation
|   |-- train.py               # Model training with early stopping
|   |-- eval.py                # Evaluation metrics and visualization
|   |-- gradcam.py             # Grad-CAM heatmap gallery generation
|   |-- infer.py               # Single-image and batch inference CLI
|-- src/
|   |-- data/                  # Dataset class, transforms, split utilities
|   |-- models/                # Classifier architecture and model factory
|   |-- evaluation/            # Metrics, visualization, bootstrap CIs
|   |-- explainability/        # Grad-CAM heatmap generation
|   |-- utils/                 # Reproducibility, logging, config helpers
|-- docs/
|   |-- model_card.md          # Model card (Mitchell et al. 2019 format)
|   |-- poc_report.md          # Comprehensive PoC report with clinical framing
|   |-- data_audit_report.md   # Dataset quality audit
|   |-- dataset_spec.md        # Column definitions and data provenance
|-- app/                       # Streamlit demo application
|-- data/
|   |-- splits/                # Train/val/test CSV manifests
|-- checkpoints/               # Saved model weights (gitignored)
|-- results/
|   |-- stratified/            # Stratified split evaluation outputs
|   |-- center_holdout/        # Center-holdout split evaluation outputs
|   |-- gradcam/               # Grad-CAM heatmap gallery
|   |-- inference/             # Inference output examples
|-- tests/                     # Unit and integration tests
|-- notebooks/                 # Exploratory analysis notebooks
|-- logs/                      # Training logs
```

## Configuration

All hyperparameters, paths, and seeds are centralized in `configs/default.yaml`.
Key settings:

| Setting | Default | Description |
|---------|---------|-------------|
| `seed` | 42 | Random seed for reproducibility |
| `model.backbone` | efficientnet_b0 | Backbone architecture |
| `model.num_classes` | 3 | Number of output classes |
| `training.batch_size` | 32 | Training batch size |
| `training.learning_rate` | 0.001 | Initial learning rate |
| `training.epochs` | 50 | Maximum training epochs |
| `training.early_stopping_patience` | 7 | Early stopping patience |
| `training.scheduler` | cosine | Learning rate scheduler |
| `training.split_strategy` | stratified | Split strategy (stratified or center) |
| `evaluation.bootstrap_iterations` | 1000 | Bootstrap CI iterations |
| `data.image_size` | 224 | Input image size (pixels) |

Override any setting via command-line dot-notation:

```bash
python scripts/train.py --config configs/default.yaml --override training.batch_size=64
python scripts/train.py --config configs/default.yaml --override training.learning_rate=0.0005
```

Or provide a custom YAML configuration file:

```bash
python scripts/train.py --config configs/custom.yaml
```

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/model_card.md`](docs/model_card.md) | Model card following Mitchell et al. (2019) format |
| [`docs/poc_report.md`](docs/poc_report.md) | Comprehensive PoC report with clinical framing |
| [`docs/data_audit_report.md`](docs/data_audit_report.md) | Dataset quality audit with figures |
| [`docs/dataset_spec.md`](docs/dataset_spec.md) | Column definitions and data provenance |

## License and Disclaimer

The BTXRD dataset is licensed under **CC BY-NC-ND 4.0** (non-commercial,
no derivatives). This project is a research prototype.

**NOT FOR CLINICAL USE.** This model has not been validated for diagnostic
purposes. It is intended solely as a proof-of-concept for bone tumor
radiograph classification research. Do not use predictions from this model
to inform clinical decisions.

## Citation

If you use the BTXRD dataset, please cite the original paper:

```bibtex
@article{yao2025btxrd,
  title={A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors},
  author={Yao, Shunhan and others},
  journal={Scientific Data},
  year={2025},
  publisher={Nature Publishing Group},
  doi={10.1038/s41597-024-04311-y}
}
```
