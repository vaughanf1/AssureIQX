# AssureXRay

Reproducible 3-class bone tumor classification (Normal / Benign / Malignant) on BTXRD radiographs with Grad-CAM explainability.

## Overview

AssureXRay is a proof-of-concept that trains an EfficientNet-B0 classifier on the
[BTXRD dataset](https://figshare.com/articles/dataset/BTXRD/22813073) (3,746 bone
tumor radiographs from 3 medical centers) and produces a complete evaluation pack
with clinician-facing Grad-CAM heatmaps.

**Approach:**
- Transfer learning from ImageNet-pretrained EfficientNet-B0
- Inverse-frequency weighted cross-entropy loss for class imbalance (Normal: 1,879 / Benign: 1,525 / Malignant: 342)
- Dual split strategy: stratified random (70/15/15) and center-holdout (Center 1 train, Centers 2+3 test)
- Grad-CAM explainability with annotation comparison

## Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- pip
- (Optional) CUDA 12.1 for GPU training

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

## Project Structure

```
AssureXRay/
|-- configs/              # YAML configuration files
|-- scripts/              # CLI entry points (download, audit, split, train, eval, gradcam, infer)
|-- src/
|   |-- data/             # Dataset class, transforms, split utilities
|   |-- models/           # Classifier architecture and model factory
|   |-- evaluation/       # Metrics, visualization, bootstrap CIs
|   |-- explainability/   # Grad-CAM heatmap generation
|   |-- utils/            # Reproducibility, logging, config helpers
|-- app/                  # Streamlit demo application
|-- data/
|   |-- splits/           # Train/val/test CSV manifests
|-- checkpoints/          # Saved model weights (gitignored)
|-- results/              # Evaluation outputs, plots, Grad-CAM images
|-- docs/                 # Data audit report, dataset spec, model card, PoC report
|-- tests/                # Unit and integration tests
|-- notebooks/            # Exploratory analysis notebooks
|-- logs/                 # Training logs
```

## Available Commands

All pipeline stages are accessible via `make`:

| Command         | Description                                          |
|-----------------|------------------------------------------------------|
| `make download` | Download BTXRD dataset from figshare                 |
| `make audit`    | Profile dataset and generate audit report            |
| `make split`    | Create train/val/test split manifests                |
| `make train`    | Train EfficientNet-B0 on both split strategies       |
| `make evaluate` | Run full evaluation suite with metrics and plots     |
| `make gradcam`  | Generate Grad-CAM heatmaps for selected examples     |
| `make infer`    | Run inference on a single image or directory          |
| `make report`   | Generate model card and PoC report                   |
| `make demo`     | Launch Streamlit demo app                            |
| `make all`      | Run full pipeline end-to-end                         |

## Configuration

All hyperparameters, paths, and seeds are centralized in `configs/default.yaml`.
Key settings include learning rate, batch size, epochs, early stopping patience,
backbone architecture, loss type, split ratios, and random seed.

Override any setting via command-line arguments or a custom YAML file:

```bash
python scripts/train.py --config configs/custom.yaml
```

## License and Disclaimer

The BTXRD dataset is licensed under **CC BY-NC-ND 4.0** (non-commercial,
no derivatives). This project is a research prototype.

**NOT FOR CLINICAL USE.** This model has not been validated for diagnostic
purposes. It is intended solely as a proof-of-concept for bone tumor
radiograph classification research.
