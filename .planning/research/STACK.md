# Technology Stack

**Project:** AssureXRay -- 3-class bone tumor classification PoC
**Researched:** 2026-02-19
**Overall confidence:** MEDIUM (versions based on training data up to May 2025; web verification was unavailable -- verify versions with `pip index versions <pkg>` before installing)

---

## Recommended Stack

### Core Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | >=3.10, <3.13 | Runtime | 3.10+ for match statements and typing improvements. 3.12 is the sweet spot for PyTorch compatibility. Avoid 3.13 until PyTorch officially supports it. | HIGH |
| PyTorch | >=2.2, <=2.5 | Deep learning framework | Required by project spec. 2.2+ has `torch.compile` stable, improved memory efficiency, and native `torch.amp` (replaces deprecated `torch.cuda.amp`). | MEDIUM |
| torchvision | >=0.17, <=0.20 | Pretrained models, transforms | Ships EfficientNet-B0 and ResNet50 pretrained on ImageNet. Transforms V2 API (stable since 0.17) is the correct path forward. Must match PyTorch version. | MEDIUM |

### Model Zoo / Backbone

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| timm (pytorch-image-models) | >=1.0.0 | Model registry and pretrained weights | Provides `efficientnet_b0` with more weight variants (ImageNet-1k, ImageNet-21k, noisy-student) than torchvision. Ross Wightman's `timm` is the de facto standard for image classification backbones. Use `timm.create_model("efficientnet_b0", pretrained=True, num_classes=3)` for one-liner fine-tuning setup. | HIGH |

**Decision: timm vs torchvision for backbone loading**

Use **timm** as the primary backbone source, not `torchvision.models`:

| Factor | timm | torchvision.models |
|--------|------|--------------------|
| Weight variants | 5+ EfficientNet-B0 variants (noisy-student, advprop, etc.) | 1 variant (IMAGENET1K_V1) |
| API consistency | `create_model(name, pretrained=True, num_classes=N)` -- uniform | Each model has different constructor patterns |
| Medical imaging community | Standard choice in medical imaging papers | Used but less flexible |
| Feature extraction | Built-in `forward_features()` method | Requires manual layer surgery |

torchvision is still needed for transforms and data utilities. timm is specifically for the backbone.

### Data Processing & Augmentation

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| torchvision.transforms.v2 | (ships with torchvision) | Core transforms pipeline | The V2 API is the official path forward. Supports tensor and PIL inputs uniformly. Use for: Resize, Normalize, ToImage, ToDtype. | HIGH |
| albumentations | >=1.4.0 | Advanced medical image augmentations | Richer augmentation library than torchvision alone. Critical augmentations for radiographs: CLAHE (contrast-limited adaptive histogram equalization), ElasticTransform, ShiftScaleRotate, CoarseDropout. All operate on numpy arrays. | HIGH |
| Pillow | >=10.0 | JPEG loading | Transitive dependency. Needed for image I/O. | HIGH |

**Decision: albumentations vs torchvision-only augmentations**

Use **albumentations** for the training augmentation pipeline:

- **CLAHE** is critical for radiograph preprocessing -- normalizes contrast across varying exposure levels. torchvision does not offer CLAHE.
- **ElasticTransform** simulates anatomical variability -- proven effective in medical imaging augmentation.
- **ShiftScaleRotate** with safe rotation ranges (+-15deg) handles radiograph positioning variance.
- torchvision.transforms.v2 handles Resize + Normalize + ToTensor only (the deterministic preprocessing steps).

Pattern:
```python
# Training: albumentations (stochastic augmentations) -> torchvision (normalize/tensorize)
# Validation/Test: torchvision only (deterministic resize + normalize)
```

### Loss Functions & Class Imbalance

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| torch.nn.CrossEntropyLoss | (ships with PyTorch) | Primary loss with class weights | `weight` parameter accepts per-class weights. Compute as inverse frequency: `1/class_count`, then normalize. Simple, well-understood, sufficient for a baseline PoC. | HIGH |
| focal_loss (custom) | N/A | Focal loss fallback | Implement in ~15 lines rather than adding a dependency. Focal loss down-weights easy examples (Normal class). Use alpha=class_weights, gamma=2.0. Only switch to this if weighted CE underperforms on malignant recall. | HIGH |

**Decision: Weighted CrossEntropy vs Focal Loss**

Start with **weighted CrossEntropyLoss** (simpler baseline):

```python
# Class counts: Normal=1879, Benign=1525, Malignant=342
# Inverse frequency weights, normalized
weights = torch.tensor([1/1879, 1/1525, 1/342])
weights = weights / weights.sum() * 3  # scale to sum=num_classes
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
```

Only move to focal loss if malignant sensitivity < 70% with weighted CE. Focal loss adds a hyperparameter (gamma) that complicates tuning for a PoC.

### Evaluation & Metrics

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scikit-learn | >=1.4 | Metrics computation | `roc_auc_score(multi_class='ovr')`, `precision_recall_curve`, `confusion_matrix`, `classification_report`. The standard. No reason to use anything else for offline evaluation. | HIGH |
| torchmetrics | >=1.3 | Optional: batch-level metrics during training | If you want live ROC-AUC during training loops without accumulating predictions manually. Not strictly necessary for a PoC -- scikit-learn on accumulated predictions is simpler. | MEDIUM |

**Recommendation:** Use **scikit-learn only** for evaluation. Accumulate predictions during eval loop, compute metrics in one pass. torchmetrics adds complexity without proportional benefit in a PoC with a small dataset.

For **bootstrap confidence intervals**, use scipy or a simple numpy implementation:

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| scipy | >=1.12 | Bootstrap CIs | `scipy.stats.bootstrap()` provides BCa confidence intervals out of the box. Simpler than manual implementation. | HIGH |

### Explainability

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytorch-grad-cam | >=1.5.0 | Grad-CAM heatmap generation | Jacob Gildenblatt's library. Supports EfficientNet, ResNet, and other architectures. Handles the target layer selection, gradient computation, and overlay generation. `from pytorch_grad_cam import GradCAM`. Works with timm models. | HIGH |

**Why this library over manual implementation:**
- Handles edge cases (batch norm eval mode, gradient hooks cleanup)
- Supports GradCAM, GradCAM++, ScoreCAM, AblationCAM -- easy to compare methods
- Overlay utility: `show_cam_on_image()` produces publication-quality heatmaps
- Actively maintained, works with timm models natively

**Target layer for EfficientNet-B0:**
```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# For timm EfficientNet-B0, target the last convolutional block
target_layers = [model.conv_head]  # or model.blocks[-1] for timm
cam = GradCAM(model=model, target_layers=target_layers)
```

### Training Infrastructure

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| torch.amp (native) | (ships with PyTorch 2.x) | Mixed precision training | `torch.amp.autocast('cuda')` + `torch.amp.GradScaler('cuda')`. Halves VRAM usage -- critical for 8GB GPU constraint. Native in PyTorch 2.x (no longer `torch.cuda.amp`). | HIGH |
| tqdm | >=4.65 | Progress bars | Standard training loop progress. Tiny dependency, massive QoL improvement. | HIGH |

**Decision: Plain PyTorch vs PyTorch Lightning**

Use **plain PyTorch training loops**, not PyTorch Lightning:

| Factor | Plain PyTorch | PyTorch Lightning |
|--------|--------------|-------------------|
| Codebase size | ~200 lines for training loop | ~100 lines but hidden complexity |
| Debuggability | Full visibility into every step | Callbacks/hooks hide control flow |
| PoC appropriateness | Perfect -- no abstraction overhead | Over-engineered for single-GPU 3-class classifier |
| Grad-CAM integration | Direct model access | Need to unwrap `LightningModule.model` |
| Learning curve | Zero (if you know PyTorch) | Non-trivial callback system |
| Reproducibility | Explicit seed/determinism control | Seeds set via `Trainer(deterministic=True)` but hidden |

For a lean PoC on a single GPU with a small dataset, Lightning adds indirection without benefit. A well-structured `train.py` with explicit loops is more auditable and debuggable.

### Configuration Management

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| PyYAML | >=6.0 | Config file parsing | YAML config files for hyperparameters, paths, experiment settings. Human-readable, standard in ML projects. | HIGH |
| argparse | (stdlib) | CLI arguments | Override config values from command line. `--lr 0.001 --epochs 50`. No external dependency. | HIGH |

**What NOT to use:**
- **Hydra/OmegaConf:** Over-engineered for a PoC. Hydra's config composition, multirun, and sweep features are for large experiment management -- not a single baseline run.
- **wandb/MLflow:** Not needed for PoC. Log to CSV/JSON files + console. Add experiment tracking only if the project grows beyond PoC.

### Data Management & Utilities

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pandas | >=2.1 | CSV/metadata handling | Load `dataset.csv`, compute class distributions, merge metadata. The standard for tabular data in Python. | HIGH |
| matplotlib | >=3.8 | Plotting | Confusion matrices, ROC curves, PR curves, training loss curves. Publication-quality plots with `plt.savefig()`. | HIGH |
| seaborn | >=0.13 | Statistical visualization | `seaborn.heatmap()` for confusion matrix, distribution plots for class balance. Thin wrapper over matplotlib with better defaults. | HIGH |
| numpy | >=1.26 | Array operations | Transitive dependency of everything. Pin to >=1.26 for numpy 2.x compatibility pathway. | HIGH |

### Demo (Optional)

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| streamlit | >=1.35 | Interactive demo app | Upload image -> prediction + Grad-CAM overlay. ~50 lines of code for a working demo. Fastest path to a clinician-facing prototype. | MEDIUM |

**Why Streamlit over Gradio:**
- Simpler API for single-image upload + display use case
- Better layout control for side-by-side image comparison (original vs Grad-CAM)
- More mature ecosystem, larger community
- Gradio is fine too -- but Streamlit requires less boilerplate for this specific use case

### Reproducibility

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pip-tools / uv | Latest | Dependency locking | Generate `requirements.txt` with pinned versions from `requirements.in`. `uv` is faster but `pip-tools` is more established. Either works. | MEDIUM |

---

## Alternatives Considered and Rejected

| Category | Recommended | Rejected | Why Not |
|----------|-------------|----------|---------|
| Backbone source | timm | torchvision.models only | Fewer weight variants, less flexible API |
| Training framework | Plain PyTorch | PyTorch Lightning | Over-engineered for single-GPU PoC, adds indirection |
| Training framework | Plain PyTorch | Hugging Face Trainer | Designed for NLP, awkward fit for image classification |
| Augmentation | albumentations | torchvision only | No CLAHE, limited medical-relevant transforms |
| Augmentation | albumentations | monai.transforms | MONAI is heavy (medical imaging framework); albumentations is lighter and sufficient |
| Config | YAML + argparse | Hydra/OmegaConf | Composition/sweep features unnecessary for PoC |
| Experiment tracking | CSV/JSON files | wandb/MLflow | External service dependency, overkill for baseline PoC |
| Explainability | pytorch-grad-cam | captum (Facebook) | captum has broader scope but pytorch-grad-cam is more focused, simpler API for CAM methods |
| Metrics | scikit-learn | torchmetrics | scikit-learn is simpler for offline evaluation on small dataset |
| Data format | Raw JPEG + CSV | HuggingFace Datasets | Extra abstraction layer with no benefit for 3,746 local images |
| Demo | Streamlit | Gradio | Simpler for this specific use case (single image upload + display) |
| Full framework | PyTorch only | MONAI | MONAI is a full medical imaging framework -- massive dependency for a classification PoC. Use if project grows to segmentation/detection. |

---

## What NOT to Use (Anti-Stack)

### Do NOT use MONAI for this PoC
MONAI (Medical Open Network for AI) is a PyTorch-based framework for medical imaging. It provides transforms, data loaders, network architectures, and training workflows. **It is overkill for a 3-class classification baseline.** MONAI shines for:
- 3D volumetric data (CT/MRI)
- Segmentation pipelines
- Complex preprocessing (resampling, orientation correction)

For 2D radiograph classification, standard PyTorch + timm + albumentations covers everything needed with fewer dependencies and less complexity.

### Do NOT use TensorFlow/Keras
Project spec requires PyTorch. Even if it didn't, PyTorch is the dominant framework in medical imaging research (2024-2025 papers overwhelmingly use PyTorch).

### Do NOT use fastai
fastai's high-level abstractions hide critical details (loss weighting, augmentation control, Grad-CAM integration) that this PoC needs explicit control over. It's excellent for learning but wrong for an auditable medical imaging baseline.

### Do NOT add wandb/MLflow/Neptune for experiment tracking
The PoC is a single baseline experiment, not a hyperparameter sweep. Log training metrics to a CSV file and plot with matplotlib. Add tracking tools only if the project grows to multiple experiments.

### Do NOT use Jupyter notebooks for training
Notebooks are fine for EDA and visualization. Training must be in `.py` scripts for reproducibility (`python train.py --config config.yaml`). Notebooks hide state, break reproducibility, and make CI/CD impossible.

---

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Core ML stack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Model zoo
pip install timm

# Augmentation
pip install albumentations

# Explainability
pip install grad-cam

# Evaluation & data
pip install scikit-learn scipy pandas

# Visualization
pip install matplotlib seaborn

# Utilities
pip install tqdm pyyaml

# Optional: Demo
pip install streamlit

# Optional: Dependency locking
pip install pip-tools
# Then: pip-compile requirements.in > requirements.txt
```

**Note on PyTorch installation:** The `--index-url` flag selects the CUDA 12.1 wheel. Adjust for your CUDA version. For CPU-only (Mac development), omit the `--index-url` flag. For CUDA 12.4+, use `cu124`.

### requirements.in (for pip-compile)

```
torch>=2.2
torchvision>=0.17
timm>=1.0.0
albumentations>=1.4.0
grad-cam>=1.5.0
scikit-learn>=1.4
scipy>=1.12
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
tqdm>=4.65
pyyaml>=6.0
```

### Optional requirements

```
streamlit>=1.35
```

---

## Version Verification Needed

**IMPORTANT:** Versions listed above are based on training data current to May 2025. Before creating the project environment, verify latest compatible versions:

```bash
# Quick version check
pip index versions torch | head -3
pip index versions timm | head -3
pip index versions albumentations | head -3
pip index versions grad-cam | head -3
```

Key things to verify:
1. **PyTorch + torchvision version compatibility** -- these must be matched pairs (e.g., torch 2.4.x + torchvision 0.19.x)
2. **timm compatibility with your PyTorch version** -- timm 1.x should work with PyTorch 2.2+
3. **albumentations OpenCV dependency** -- albumentations pulls in opencv-python-headless; ensure no conflicts with other OpenCV installs

---

## Project Directory Structure (Implied by Stack)

```
AssureXRay/
  configs/
    default.yaml          # Hyperparameters, paths, experiment config
  src/
    data/
      dataset.py          # PyTorch Dataset class, albumentations pipeline
      download.py         # BTXRD download script
      splits.py           # Train/val/test split logic
    models/
      classifier.py       # timm backbone + classification head
    training/
      train.py            # Training loop (plain PyTorch)
      losses.py           # Weighted CE, focal loss
    evaluation/
      evaluate.py         # Metrics computation (scikit-learn)
      gradcam.py          # Grad-CAM generation (pytorch-grad-cam)
      bootstrap.py        # Bootstrap CIs (scipy)
    reporting/
      plots.py            # ROC curves, confusion matrices (matplotlib/seaborn)
      model_card.py       # Model card generation
    demo/
      app.py              # Streamlit demo (optional)
  scripts/
    train.sh              # Reproducible training entrypoint
    evaluate.sh           # Reproducible evaluation entrypoint
  requirements.in
  requirements.txt        # Pinned (generated by pip-compile)
```

---

## Sources

- **timm library:** Author's documentation and GitHub (github.com/huggingface/pytorch-image-models). timm is the standard backbone library for image classification in PyTorch. [HIGH confidence -- well-established, stable API]
- **pytorch-grad-cam:** GitHub (github.com/jacobgil/pytorch-grad-cam). De facto standard for CAM visualization in PyTorch. [HIGH confidence]
- **albumentations:** Official docs (albumentations.ai). CLAHE and medical imaging augmentations are well-documented. [HIGH confidence]
- **PyTorch AMP:** Official PyTorch documentation for `torch.amp` module (replaced `torch.cuda.amp` in PyTorch 2.x). [HIGH confidence]
- **torchvision transforms V2:** Stable since torchvision 0.17, documented in official torchvision docs. [HIGH confidence]
- **BTXRD dataset:** Yao et al., Scientific Data 2025. DOI: 10.1038/s41597-024-04311-y [HIGH confidence -- peer-reviewed]
- **Version numbers:** Based on training data up to May 2025. [MEDIUM confidence -- may have newer releases; verify before installing]
