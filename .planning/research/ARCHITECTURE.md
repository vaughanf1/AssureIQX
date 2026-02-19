# Architecture Patterns: Medical Image Classification PoC

**Domain:** 3-class bone tumor classification from radiographs (Normal / Benign / Malignant)
**Project:** AssureXRay
**Researched:** 2026-02-19
**Overall Confidence:** HIGH (well-established domain with stable patterns; PyTorch transfer-learning image classification is a mature, widely-documented pipeline)

**Note:** WebSearch and WebFetch were unavailable during this research session. All findings are derived from training knowledge of PyTorch, torchvision, medical imaging ML, and ML PoC project patterns. These patterns are stable and well-established, so confidence remains HIGH, but specific version numbers should be verified against current PyPI/docs during implementation.

---

## Recommended Architecture

AssureXRay is a **batch-oriented, script-per-stage ML pipeline** -- not a microservice, not a monolith application, not a DAG orchestrator. Each stage is a standalone CLI script that reads from disk and writes to disk. Stages are connected by filesystem convention (agreed paths), not by in-memory coupling.

This is the correct architecture for a PoC because:
1. Each stage can be run, debugged, and rerun independently.
2. Intermediate artifacts (splits, checkpoints, metrics) persist between runs.
3. No orchestration framework overhead (Airflow, Prefect, etc. are overkill for PoC).
4. A researcher can inspect any intermediate artifact at any point.

```
                        CONFIGURATION
                     (configs/*.yaml or
                      configs/*.json)
                            |
                            v
  +-----------+    +---------------+    +-------------+
  | download   |--->| data_raw/     |--->| audit       |
  | script     |    | (raw JPEGs +  |    | script      |
  +-----------+    |  dataset.csv)  |    +------+------+
                   +---------------+           |
                                               v
                                      +-----------------+
                                      | data_audit_     |
                                      | report.md       |
                                      +-----------------+
                            |
                            v
                   +---------------+    +-------------+
                   | split         |--->| data/        |
                   | script        |    | splits/*.csv |
                   +---------------+    +------+------+
                                               |
                                               v
                                      +-----------------+
                                      | Dataset loader  |
                                      | (PyTorch        |
                                      |  Dataset class) |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | train script    |
                                      | (training loop) |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | checkpoints/    |
                                      | (model weights) |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | eval script     |
                                      | (metrics +      |
                                      |  confusion mx)  |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | results/        |
                                      | (metrics JSON,  |
                                      |  plots, tables) |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | gradcam script  |
                                      | (heatmaps for   |
                                      |  selected imgs) |
                                      +--------+--------+
                                               |
                                               v
                                      +-----------------+
                                      | results/gradcam/|
                                      | (overlay PNGs)  |
                                      +-----------------+
                                               |
                                               v
                                      +-----------------+
                                      | infer script    |
                                      | (single image   |
                                      |  prediction +   |
                                      |  Grad-CAM)      |
                                      +-----------------+
                                               |
                                               v
                                      +-----------------+
                                      | Streamlit app   |
                                      | (optional demo) |
                                      +-----------------+
```

---

## Component Boundaries

Each component is a self-contained unit with explicit inputs and outputs. No component imports another component's internals -- they communicate through the filesystem.

| Component | Responsibility | Inputs | Outputs | Communicates With |
|-----------|---------------|--------|---------|-------------------|
| **download.py** | Fetch BTXRD from figshare, extract, organize | Config (figshare URL) | `data_raw/` directory with JPEGs + `dataset.csv` | None (entry point) |
| **audit.py** | Profile data quality, class distribution, image stats | `data_raw/`, `dataset.csv` | `docs/data_audit_report.md`, audit JSON | download (consumes its output) |
| **split.py** | Create train/val/test CSVs for both split strategies | `dataset.csv`, config | `data/splits/stratified_{train,val,test}.csv`, `data/splits/center_{train,val,test}.csv` | audit (should run after audit confirms data integrity) |
| **src/data/dataset.py** | PyTorch Dataset class with transforms | Split CSVs, image directory, config | DataLoader-ready tensors (in memory) | split (reads CSVs), train/eval (imported as library) |
| **src/data/transforms.py** | Augmentation and preprocessing pipelines | Config | Transform compositions (in memory) | dataset (imported) |
| **train.py** | Training loop with early stopping, checkpointing | Config, DataLoaders | `checkpoints/*.pt`, training logs | dataset (imports), config |
| **eval.py** | Compute metrics, generate plots | Checkpoint, DataLoaders, config | `results/{split_name}/metrics.json`, confusion matrix PNG, ROC curves | train (uses checkpoint), dataset (imports) |
| **gradcam.py** | Generate Grad-CAM heatmaps for selected images | Checkpoint, image paths, config | `results/gradcam/*.png` overlays | eval (uses same checkpoint), dataset (imports) |
| **infer.py** | Single-image prediction + Grad-CAM | Checkpoint, single image path | Prediction JSON + overlay image | train (uses checkpoint) |
| **app.py** | Streamlit demo UI | Checkpoint | Browser-based interface | infer (wraps same logic) |
| **src/models/classifier.py** | Model definition (EfficientNet-B0/ResNet50 + head) | Config (backbone choice, num_classes) | nn.Module (in memory) | train, eval, gradcam, infer (all import) |
| **src/utils/metrics.py** | Metric computation functions | Predictions + labels | Metric dictionaries | eval (imports) |
| **src/utils/visualization.py** | Plotting functions (confusion matrix, ROC, PR) | Metric data | Plot images (PNG/PDF) | eval (imports) |
| **configs/*.yaml** | Hyperparameters, paths, experiment settings | None (authored by user) | Read by all scripts | All scripts |

### The Library vs. Script Distinction

This is critical for clean architecture:

- **Scripts** (`scripts/` or root-level `*.py`): Entry points. They parse CLI args, load config, orchestrate a pipeline stage, and write results to disk. They are *not* imported by other scripts.
- **Library code** (`src/`): Reusable modules imported by scripts. Dataset class, model definition, metric functions, visualization utilities. They never read config or write to disk on their own -- that is the script's job.

This separation means:
- Scripts can be swapped or extended without touching library code.
- Library code can be unit-tested without filesystem side effects.
- The Streamlit app and the CLI infer script share the same model and prediction logic from `src/`.

---

## Data Flow

### Stage 1: Data Acquisition

```
figshare (remote) --> download.py --> data_raw/
                                       |-- images/
                                       |   |-- normal/
                                       |   |-- benign/
                                       |   |-- malignant/
                                       |-- annotations/     (LabelMe JSONs)
                                       |-- dataset.csv
```

**Key decisions:**
- Raw data is never modified after download. Treat `data_raw/` as immutable.
- Images are organized into class subdirectories for quick inspection, but the CSV remains the source of truth for labels (not the directory structure).
- Annotations go in a separate subdirectory -- they are used for Grad-CAM validation, not for training.

### Stage 2: Data Audit

```
data_raw/ + dataset.csv --> audit.py --> docs/data_audit_report.md
                                          + data_raw/audit_artifacts/
                                              |-- class_distribution.png
                                              |-- image_dimensions.json
                                              |-- quality_flags.csv
```

**What gets audited:**
- Class distribution (Normal: 1879, Benign: 1525, Malignant: 342)
- Image dimension statistics (min, max, median, std for height/width)
- Missing values in metadata columns
- Annotation coverage (which images have LabelMe JSONs)
- Center distribution
- Potential duplicates or corrupt files

### Stage 3: Data Splitting

```
dataset.csv --> split.py --> data/splits/
                               |-- stratified_train.csv
                               |-- stratified_val.csv
                               |-- stratified_test.csv
                               |-- center_train.csv      (Center 1)
                               |-- center_val.csv        (Center 1 subset)
                               |-- center_test.csv       (Centers 2+3)
```

**Key decisions:**
- Split files are CSVs containing `image_id` and `label` columns (plus any metadata needed). They reference images by ID, not by path.
- Splits are generated once and committed to version control. This ensures reproducibility -- everyone trains on the same split.
- The split script takes a random seed from config.
- Stratified split: 70/15/15 with stratification on the 3-class label.
- Center holdout split: Center 1 for train+val (with internal stratified split), Centers 2+3 for test.

### Stage 4: Training

```
data/splits/*.csv + data_raw/images/ + configs/train.yaml
    --> train.py
    --> checkpoints/
         |-- best_stratified.pt
         |-- last_stratified.pt
         |-- best_center.pt
         |-- last_center.pt
    --> logs/
         |-- train_stratified.log
         |-- train_center.log
```

**Data flow through training:**
1. Script reads config (backbone, lr, epochs, batch_size, etc.)
2. Script loads split CSVs to get image IDs for train/val
3. Dataset class maps image IDs to file paths, loads JPEGs, applies transforms
4. DataLoader batches and shuffles
5. Training loop: forward, loss (weighted cross-entropy), backward, step
6. Validation every epoch, early stopping on val loss or val AUC
7. Best checkpoint saved by validation metric

**Checkpoint contents (save as dict, not just state_dict):**
```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "epoch": ...,
    "val_metric": ...,
    "config": ...,           # Full config for reproducibility
    "class_names": [...],
    "split_strategy": "stratified" | "center",
}
```

### Stage 5: Evaluation

```
checkpoints/best_*.pt + data/splits/*_test.csv + data_raw/images/
    --> eval.py
    --> results/{split_name}/
         |-- metrics.json        (AUC, sensitivity, specificity, etc.)
         |-- confusion_matrix.png
         |-- roc_curves.png
         |-- pr_curves.png
         |-- classification_report.txt
         |-- bootstrap_ci.json   (optional)
```

**Metrics computed:**
- Per-class: sensitivity (recall), specificity, precision, F1
- Overall: accuracy, macro-averaged AUC (one-vs-rest), weighted F1
- Plots: ROC curves (per-class + micro/macro avg), PR curves, confusion matrix heatmap
- Optional: bootstrap 95% CIs on AUC

### Stage 6: Explainability (Grad-CAM)

```
checkpoints/best_*.pt + selected image list + data_raw/images/
    --> gradcam.py
    --> results/gradcam/
         |-- TP_normal_001.png
         |-- FP_malignant_042.png
         |-- ...
         |-- gradcam_summary.md
```

**Image selection strategy:**
- From evaluation results, select examples: 2-3 TP, 1-2 FP, 1-2 FN per class
- Generate heatmap overlay for each
- Optionally compare Grad-CAM activation region to LabelMe annotation bounding box

**Grad-CAM target layer:**
- EfficientNet-B0: last convolutional layer of `features` (the output of the final MBConv block)
- ResNet50: `layer4[-1]` (last bottleneck block)

### Stage 7: Inference

```
single_image.jpg + checkpoints/best_*.pt
    --> infer.py
    --> stdout: prediction JSON
    --> output_dir/: overlay image
```

**Output format:**
```json
{
    "image": "path/to/image.jpg",
    "prediction": "Benign",
    "probabilities": {
        "Normal": 0.12,
        "Benign": 0.81,
        "Malignant": 0.07
    },
    "gradcam_overlay": "path/to/overlay.png"
}
```

### Stage 8: Demo (Optional Streamlit)

```
Browser --> app.py (Streamlit) --> src/models/ + checkpoint
    --> prediction + Grad-CAM overlay displayed in browser
```

---

## Recommended Directory Structure

```
AssureXRay/
|
|-- configs/                          # All configuration
|   |-- default.yaml                  # Default hyperparameters + paths
|   |-- experiment_efficientnet.yaml  # Experiment-specific overrides
|   |-- experiment_resnet.yaml        # Alternative backbone experiment
|
|-- scripts/                          # Pipeline entry-point scripts
|   |-- download.py                   # Stage 1: Fetch BTXRD
|   |-- audit.py                      # Stage 2: Data profiling
|   |-- split.py                      # Stage 3: Create train/val/test splits
|   |-- train.py                      # Stage 4: Model training
|   |-- eval.py                       # Stage 5: Evaluation + metrics
|   |-- gradcam.py                    # Stage 6: Explainability heatmaps
|   |-- infer.py                      # Stage 7: Single-image inference
|
|-- src/                              # Reusable library code
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- dataset.py                # BTXRDDataset(Dataset) class
|   |   |-- transforms.py            # Train/val/test transform pipelines
|   |   |-- split_utils.py           # Stratified + center split logic
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- classifier.py            # BoneTumorClassifier(nn.Module)
|   |   |-- factory.py               # create_model(config) -> nn.Module
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- metrics.py               # AUC, sensitivity, specificity, etc.
|   |   |-- visualization.py         # Confusion matrix, ROC, PR plots
|   |   |-- bootstrap.py             # Bootstrap CI computation
|   |
|   |-- explainability/
|   |   |-- __init__.py
|   |   |-- gradcam.py               # GradCAM class + overlay generation
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- config.py                 # Config loading + merging
|       |-- logging.py                # Logging setup
|       |-- reproducibility.py        # Seed setting, deterministic flags
|
|-- app/                              # Optional Streamlit demo
|   |-- app.py
|   |-- components/                   # Streamlit UI components
|
|-- data_raw/                         # Raw data (gitignored, immutable)
|   |-- images/
|   |-- annotations/
|   |-- dataset.csv
|
|-- data/                             # Processed/derived data (splits committed)
|   |-- splits/
|       |-- stratified_train.csv
|       |-- stratified_val.csv
|       |-- stratified_test.csv
|       |-- center_train.csv
|       |-- center_val.csv
|       |-- center_test.csv
|
|-- checkpoints/                      # Model checkpoints (gitignored)
|
|-- results/                          # Evaluation outputs (committed selectively)
|   |-- stratified/
|   |   |-- metrics.json
|   |   |-- confusion_matrix.png
|   |   |-- roc_curves.png
|   |   |-- pr_curves.png
|   |
|   |-- center_holdout/
|   |   |-- metrics.json
|   |   |-- confusion_matrix.png
|   |   |-- roc_curves.png
|   |   |-- pr_curves.png
|   |
|   |-- gradcam/
|       |-- (overlay PNGs)
|       |-- gradcam_summary.md
|
|-- docs/                             # Documentation deliverables
|   |-- dataset_spec.md
|   |-- data_audit_report.md
|   |-- model_card.md
|   |-- poc_report.md
|
|-- tests/                            # Unit and integration tests
|   |-- test_dataset.py
|   |-- test_model.py
|   |-- test_metrics.py
|   |-- test_transforms.py
|   |-- test_split.py
|
|-- notebooks/                        # Exploration notebooks (optional)
|   |-- 01_data_exploration.ipynb
|   |-- 02_training_analysis.ipynb
|
|-- .planning/                        # GSD planning artifacts
|-- .gitignore
|-- pyproject.toml                    # Project metadata + dependencies
|-- requirements.txt                  # Pinned dependencies (or use pyproject.toml)
|-- Makefile                          # Convenience targets for pipeline stages
|-- README.md
```

### What Gets Committed vs. Gitignored

| Path | Git Status | Reason |
|------|------------|--------|
| `configs/` | Committed | Reproducibility -- exact settings recorded |
| `scripts/` | Committed | Code |
| `src/` | Committed | Code |
| `data_raw/` | **Gitignored** | Large binary files, downloaded via script |
| `data/splits/` | Committed | Small CSVs, critical for reproducibility |
| `checkpoints/` | **Gitignored** | Large binary files |
| `results/` | Committed (selectively) | Metrics JSON and key plots committed; bulk Grad-CAM PNGs optionally gitignored |
| `docs/` | Committed | Deliverables |
| `tests/` | Committed | Code |
| `notebooks/` | Committed | Exploration record |

---

## Patterns to Follow

### Pattern 1: Config-Driven Pipeline

**What:** Every script reads a YAML config file. No magic numbers in code. All hyperparameters, paths, and experiment settings live in config.

**Why:** Reproducibility. Anyone can re-run the exact experiment by pointing to the same config. Different experiments are different config files, not code changes.

**Implementation:**

```yaml
# configs/default.yaml
data:
  raw_dir: data_raw
  splits_dir: data/splits
  image_size: 224
  num_workers: 4

model:
  backbone: efficientnet_b0
  pretrained: true
  num_classes: 3
  dropout: 0.2

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  early_stopping_patience: 7
  scheduler: cosine
  class_weights: auto   # computed from training set distribution

evaluation:
  bootstrap_iterations: 1000
  confidence_level: 0.95

gradcam:
  target_layer: auto    # auto-detect based on backbone
  examples_per_class: 3

paths:
  checkpoints_dir: checkpoints
  results_dir: results
  docs_dir: docs

seed: 42
device: auto   # auto-detect GPU/CPU
```

**Script pattern:**
```python
# scripts/train.py
import argparse
from src.utils.config import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", nargs="*")  # key=value overrides
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    # ... rest of training logic
```

### Pattern 2: Deterministic Reproducibility

**What:** Set all random seeds at the start of every script. Use deterministic CUDA operations where possible.

**Why:** Medical ML must be reproducible. "I got different results" is unacceptable for a PoC that informs clinical decisions.

**Implementation:**
```python
# src/utils/reproducibility.py
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Note:** `cudnn.deterministic = True` has a performance cost (~10-15% slower). This is acceptable for a PoC. For production, you would benchmark the tradeoff.

### Pattern 3: Checkpoint-as-Contract

**What:** Checkpoints contain everything needed to reproduce inference: model weights, config, class names, normalization stats, split strategy.

**Why:** Months later, you can load a checkpoint and know exactly what model it is, how it was trained, and what it expects as input. No guessing.

**Implementation:**
```python
# In train.py
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": epoch,
    "val_metric": best_val_metric,
    "config": config,                    # full config dict
    "class_names": ["Normal", "Benign", "Malignant"],
    "image_size": config["data"]["image_size"],
    "normalization": {
        "mean": [0.485, 0.456, 0.406],  # ImageNet defaults
        "std": [0.229, 0.224, 0.225],
    },
}, checkpoint_path)
```

### Pattern 4: Metrics-as-JSON

**What:** Every evaluation run writes a structured JSON file with all metrics. Plots are secondary artifacts.

**Why:** JSON is machine-readable. You can diff metrics across experiments, generate comparison tables, feed them into reports programmatically.

**Implementation:**
```json
{
    "split_strategy": "stratified",
    "checkpoint": "checkpoints/best_stratified.pt",
    "timestamp": "2026-02-19T14:30:00",
    "overall": {
        "accuracy": 0.87,
        "macro_auc": 0.93,
        "weighted_f1": 0.86
    },
    "per_class": {
        "Normal": {"sensitivity": 0.92, "specificity": 0.88, "auc": 0.95, "support": 282},
        "Benign": {"sensitivity": 0.84, "specificity": 0.91, "auc": 0.93, "support": 229},
        "Malignant": {"sensitivity": 0.71, "specificity": 0.97, "auc": 0.91, "support": 51}
    }
}
```

### Pattern 5: Separation of Concerns in Model Definition

**What:** The model class handles only the neural network architecture. Loss function, optimizer, and training loop live in the training script.

**Why:** The model class is reused by train, eval, gradcam, and infer scripts. Each has different needs. The model should not assume training context.

**Implementation:**
```python
# src/models/classifier.py
class BoneTumorClassifier(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0",
                 num_classes: int = 3,
                 pretrained: bool = True,
                 dropout: float = 0.2):
        super().__init__()
        self.backbone = self._create_backbone(backbone, pretrained)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._get_feature_dim(backbone), num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

    def get_gradcam_target_layer(self):
        """Return the target layer for Grad-CAM."""
        # Implementation depends on backbone
        ...
```

---

## Anti-Patterns to Avoid

### Anti-Pattern 1: Monolithic Training Script

**What:** Everything in one 500-line `train.py` -- dataset creation, model definition, training loop, evaluation, plotting, all inline.

**Why bad:** Cannot reuse dataset logic in eval, cannot reuse model in inference, cannot test components independently. Changes to evaluation require editing the training script.

**Instead:** Factor into library modules (`src/`) imported by thin scripts. Each script is <100 lines of orchestration glue.

### Anti-Pattern 2: Path Hardcoding

**What:** Paths like `/home/user/data/images/` scattered through code.

**Why bad:** Breaks on any other machine. Makes config management impossible.

**Instead:** All paths in config YAML, relative to project root. Scripts resolve paths from config + project root detection.

### Anti-Pattern 3: Training Data Information Leak into Transforms

**What:** Computing normalization statistics from the full dataset (including test set) or fitting augmentation parameters on test data.

**Why bad:** Information leakage. Test set statistics influence model behavior.

**Instead:** Use ImageNet normalization stats (standard for transfer learning) or compute stats from training set only. Apply the same normalization to val/test.

### Anti-Pattern 4: Grad-CAM as Afterthought

**What:** Building training and evaluation first, then bolting on Grad-CAM at the end as a separate, disconnected script.

**Why bad:** Grad-CAM requires access to intermediate feature maps. If the model architecture does not expose the right layers, retrofitting is painful. Also, Grad-CAM outputs should be part of the evaluation narrative, not a separate artifact.

**Instead:** Design the model class with `get_gradcam_target_layer()` from the start. Plan Grad-CAM output integration into the results directory structure from day one.

### Anti-Pattern 5: Overengineering the PoC

**What:** Adding MLflow tracking, Docker containers, CI/CD pipelines, multi-GPU support, mixed-precision training, and custom learning rate finders to a PoC.

**Why bad:** PoC goal is to validate feasibility. Infrastructure overhead delays the answer. Most of this infrastructure is irrelevant if the PoC shows the approach does not work.

**Instead:** Keep it minimal. Config YAML + CLI scripts + filesystem artifacts. Add infrastructure only if the PoC succeeds and moves toward production.

### Anti-Pattern 6: Evaluating Only on One Split

**What:** Running evaluation only on the stratified test set and declaring success.

**Why bad:** The stratified split may overestimate performance because images from the same center (and potentially the same patient) appear in both train and test. The center holdout split is the more clinically meaningful generalization test.

**Instead:** Always evaluate on both splits. Present them side-by-side. The gap between stratified and center-holdout performance is one of the most informative findings of this PoC.

---

## Suggested Build Order

The build order is dictated by **data dependencies** (what each stage needs to exist before it can run) and **validation dependencies** (what you need to know before investing effort in the next stage).

### Phase 1: Foundation (Data Pipeline + Project Scaffolding)

**Build:**
1. Project scaffolding (directory structure, pyproject.toml, .gitignore, configs/default.yaml)
2. `download.py` -- fetch and organize raw data
3. `audit.py` -- profile data quality
4. `docs/dataset_spec.md` -- document the dataset
5. `docs/data_audit_report.md` -- document audit findings

**Rationale:** You cannot build anything meaningful without data. The audit may reveal blocking issues (corrupt files, unexpected distributions, missing labels) that change downstream decisions. The dataset spec forces you to understand the data before writing code against it.

**Validation gate:** Data is downloaded, audited, and documented. No blocking quality issues. Class distribution matches expected (1879/1525/342).

### Phase 2: Data Loading + Splitting

**Build:**
1. `src/data/transforms.py` -- preprocessing and augmentation pipelines
2. `src/data/dataset.py` -- PyTorch Dataset class
3. `src/data/split_utils.py` -- split generation logic
4. `scripts/split.py` -- generate split CSVs
5. `src/utils/reproducibility.py` -- seed management
6. `src/utils/config.py` -- config loading

**Rationale:** The dataset class and splits must be correct before training begins. A bug here (leakage, wrong labels, incorrect transforms) invalidates all downstream results. Unit tests are especially important here.

**Validation gate:** Splits generated. Split statistics verified (class proportions preserved, no overlap between train/val/test). Dataset class loads images correctly. Transforms produce expected output shapes.

### Phase 3: Model + Training

**Build:**
1. `src/models/classifier.py` -- model definition (with `get_gradcam_target_layer()`)
2. `src/models/factory.py` -- model creation from config
3. `scripts/train.py` -- training loop with early stopping
4. Training on stratified split first (faster iteration)

**Rationale:** Model definition includes Grad-CAM hooks from the start (Pattern 4 avoidance). Train on the simpler stratified split first to validate the pipeline works end-to-end before running the center holdout experiment.

**Validation gate:** Training converges (loss decreases). Val metric improves then plateaus. Checkpoint saved correctly and loads without error.

### Phase 4: Evaluation

**Build:**
1. `src/evaluation/metrics.py` -- metric computation
2. `src/evaluation/visualization.py` -- plot generation
3. `scripts/eval.py` -- evaluation pipeline
4. Run eval on both split strategies
5. `src/evaluation/bootstrap.py` -- confidence intervals (optional but recommended)

**Rationale:** Evaluation must cover both splits. The comparison is a key deliverable. Bootstrap CIs add credibility for clinical audience.

**Validation gate:** Metrics computed for both splits. Results are plausible (not 99% accuracy, which would suggest leakage; not 33%, which would suggest random). Plots generated cleanly.

### Phase 5: Explainability + Inference

**Build:**
1. `src/explainability/gradcam.py` -- Grad-CAM class
2. `scripts/gradcam.py` -- batch heatmap generation for selected examples
3. `scripts/infer.py` -- single-image prediction pipeline
4. Compare Grad-CAM regions to LabelMe annotations (qualitative)

**Rationale:** Explainability comes after evaluation because you need to know which images are TP/FP/FN before selecting Grad-CAM examples. Inference wraps the same model loading + prediction logic used in eval, so it builds on that foundation.

**Validation gate:** Grad-CAM heatmaps visually highlight tumor regions (not background/edges). For FP/FN cases, heatmaps suggest plausible failure modes. Inference script works end-to-end on a single image.

### Phase 6: Documentation + Reporting

**Build:**
1. `docs/model_card.md` -- model architecture, training details, performance, limitations
2. `docs/poc_report.md` -- comprehensive findings, clinical relevance, next steps
3. Update `README.md` with project overview and usage instructions

**Rationale:** Documentation is the deliverable. It synthesizes everything from phases 1-5. It should be written after all results are available, not before.

**Validation gate:** Model card covers all required sections. PoC report answers the feasibility question with evidence. README allows someone new to reproduce the pipeline.

### Phase 7: Demo (Optional)

**Build:**
1. `app/app.py` -- Streamlit application
2. UI components for image upload, prediction display, Grad-CAM overlay

**Rationale:** The demo is a communication tool, not a technical requirement. It should only be built if the PoC results are worth demonstrating. It reuses inference logic from Phase 5.

**Validation gate:** Upload an image, see prediction + heatmap. Works on localhost.

### Build Order Dependency Graph

```
Phase 1 (Data)
    |
    v
Phase 2 (Loading + Splits)
    |
    v
Phase 3 (Model + Training)
    |
    v
Phase 4 (Evaluation)
    |
    +----> Phase 5 (Explainability + Inference)
    |          |
    v          v
Phase 6 (Documentation) <--- requires results from 4 + 5
    |
    v
Phase 7 (Demo) -- optional, can start after Phase 5
```

**Critical path:** Phases 1 -> 2 -> 3 -> 4 are strictly sequential. Phase 5 depends on Phase 4 (for image selection). Phase 6 depends on Phases 4 and 5 (for content). Phase 7 depends on Phase 5 only (reuses inference logic).

**Parallelization opportunity:** Within each phase, there is limited parallelization opportunity because this is a PoC with a single developer. However, documentation writing (Phase 6) can begin in draft form during Phase 4 while evaluation runs.

---

## Scalability Considerations

This is a PoC. Scalability is not a primary concern. However, note the following for potential future scaling:

| Concern | PoC (Current) | If Scaling to Production |
|---------|---------------|--------------------------|
| Data volume | 3,746 images, fits in memory | Would need streaming DataLoader, data versioning (DVC) |
| Training compute | Single GPU, ~30 min per run | Multi-GPU with DistributedDataParallel |
| Experiment tracking | Config YAML + filesystem | MLflow or Weights & Biases |
| Model serving | CLI script | FastAPI + Docker or TorchServe |
| Reproducibility | Seeds + config files | Docker containers + data versioning |
| CI/CD | Manual runs | GitHub Actions with test + train pipeline |

---

## Key Architecture Decisions for AssureXRay

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| Pipeline orchestration | None (script-per-stage) | PoC does not need Airflow/Prefect/DVC pipelines. Manual execution with Makefile targets is sufficient. |
| Config format | YAML | Human-readable, supports nesting, well-supported by Python (PyYAML/OmegaConf). JSON is too verbose for config. |
| Model framework | Pure PyTorch (no Lightning) | Lightning adds abstraction for multi-GPU, logging, etc. that a PoC does not need. Pure PyTorch is easier to debug and understand. |
| Image loading | torchvision + PIL | Standard, well-tested. No need for albumentations for basic augmentations. |
| Grad-CAM implementation | pytorch-grad-cam library (jacobgil) | Well-maintained, supports multiple architectures, handles hooks correctly. Do not reimplement from scratch. |
| Metrics computation | scikit-learn | Standard, well-tested. `roc_auc_score`, `classification_report`, `confusion_matrix` are exactly what is needed. |
| Plotting | matplotlib + seaborn | Standard for ML papers. No interactive plots needed for a PoC report. |
| Config loading | OmegaConf or plain PyYAML | OmegaConf is nicer (dot access, merging, CLI overrides) but PyYAML is zero-dependency. Either works. |
| Convenience runner | Makefile | `make download`, `make audit`, `make train`, etc. Simple, universal, no Python dependency. |

---

## Sources and Confidence Notes

| Claim | Confidence | Basis |
|-------|------------|-------|
| Script-per-stage is standard for ML PoC | HIGH | Widely adopted pattern in academic and industry ML projects |
| EfficientNet-B0 last MBConv block is Grad-CAM target | HIGH | Standard documented in pytorch-grad-cam library and EfficientNet papers |
| ImageNet normalization stats for transfer learning | HIGH | Universal standard in torchvision pretrained models |
| Config-driven YAML pattern | HIGH | Standard ML engineering practice |
| pytorch-grad-cam (jacobgil) is the go-to library | HIGH | Most popular Grad-CAM Python library; verified in training data up to 2025 |
| Pure PyTorch over Lightning for PoC | MEDIUM | Opinionated recommendation; Lightning is also valid but adds complexity |
| OmegaConf for config merging | MEDIUM | Popular but not universal; PyYAML is a simpler alternative |
| Bootstrap CI methodology | HIGH | Standard statistical practice for reporting ML model uncertainty |

**Verification gap:** Could not verify current versions of pytorch-grad-cam, OmegaConf, or latest torchvision model API changes via web sources. Version pinning should be verified during implementation.
