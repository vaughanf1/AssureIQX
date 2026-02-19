# Phase 1: Scaffold and Infrastructure - Research

**Researched:** 2026-02-19
**Domain:** Python ML project scaffolding, dependency management, configuration, Makefile automation
**Confidence:** HIGH (verified against current PyPI, well-established patterns)

## Summary

Phase 1 delivers the empty but fully configured project skeleton: directory structure, pinned dependencies, YAML config, Makefile, reproducibility utilities, and README. This is pure infrastructure -- no data, no model code, no training logic. The goal is that after this phase, `git clone && pip install -r requirements.txt && make` works cleanly and shows the developer what targets are available.

The most impactful finding from this research is a **Python version constraint change**. The original STACK.md recommended Python >=3.10. However, as of early 2026, several core dependencies have dropped Python 3.10 support: scikit-learn 1.8 requires >=3.11, pandas 3.0 requires >=3.11, and scipy 1.17 requires >=3.11. The project should target **Python >=3.11, <3.14** with 3.12 as the recommended version. This avoids needing to pin older package versions just for 3.10 compatibility.

The second important finding concerns **albumentations licensing**. The original albumentations library (MIT license) was archived in July 2025 with its last release being 2.0.8 (May 2025). Its successor, AlbumentationsX, uses AGPL/Commercial dual licensing. Since this is a non-commercial research PoC and the original MIT-licensed albumentations 2.0.8 still works, **use albumentations 2.0.8** (the archived MIT version) for this PoC. It is stable, compatible with current PyTorch, and avoids licensing concerns. If the project later needs maintained augmentation, torchvision transforms v2 covers most needs except CLAHE (which can be sourced from OpenCV directly or kornia).

**Primary recommendation:** Target Python 3.12. Pin all dependencies to specific versions verified against PyPI as of 2026-02-19. Use the archived MIT-licensed albumentations 2.0.8 rather than the AGPL AlbumentationsX successor.

## Standard Stack

### Core Dependencies (Verified against PyPI 2026-02-19)

| Library | Pin Version | Latest Available | Purpose | Python Req | Confidence |
|---------|-------------|------------------|---------|------------|------------|
| torch | 2.6.0 | 2.10.0 | Deep learning framework | >=3.10 | HIGH |
| torchvision | 0.21.0 | 0.25.0 | Transforms, pretrained models | >=3.10 | HIGH |
| timm | 1.0.15 | 1.0.24 | EfficientNet-B0 backbone | >=3.8 | HIGH |
| albumentations | 2.0.8 | 2.0.8 (archived) | Training augmentations (CLAHE) | >=3.9 | HIGH |
| grad-cam | 1.5.5 | 1.5.5 | Grad-CAM heatmaps | >=3.8 | HIGH |
| scikit-learn | 1.7.2 | 1.8.0 | Metrics (AUC, confusion matrix) | >=3.10 | HIGH |
| scipy | 1.15.2 | 1.17.0 | Bootstrap CIs | >=3.10 | HIGH |
| pandas | 2.2.3 | 3.0.1 | CSV/metadata handling | >=3.9 | HIGH |
| numpy | 2.2.3 | 2.4.2 | Array operations | >=3.10 | HIGH |
| matplotlib | 3.10.0 | 3.10.8 | Plotting | >=3.10 | HIGH |
| seaborn | 0.13.2 | 0.13.2 | Statistical visualization | >=3.8 | HIGH |
| tqdm | 4.67.1 | 4.67.3 | Progress bars | >=3.7 | HIGH |
| pyyaml | 6.0.2 | 6.0.3 | Config file parsing | >=3.8 | HIGH |
| Pillow | 11.1.0 | 12.1.1 | Image I/O (transitive) | >=3.9 | HIGH |

**Critical version decision: Why NOT pin to the absolute latest versions.**

The project REQUIREMENTS specify Python 3.10-3.12 support. However, the latest versions of scikit-learn (1.8), pandas (3.0), and scipy (1.17) have dropped Python 3.10 support (they require >=3.11). Rather than use bleeding-edge versions that constrain Python compatibility, we pin to the latest versions that support Python 3.10, keeping the original compatibility range while using modern, well-tested releases. This is the safer choice for a PoC.

**Alternative approach (recommended update to STACK.md):** Change the Python requirement to >=3.11, <3.14 and use the latest package versions. This is cleaner but narrows the supported Python range. The success criteria say "Python 3.10-3.12" so the conservative approach is to keep 3.10 support by pinning slightly older (but stable) versions.

### PyTorch + Torchvision Compatibility Matrix (Verified)

| PyTorch | Torchvision | Status |
|---------|-------------|--------|
| 2.10.0 | 0.25.0 | Latest stable (Jan 2026) |
| 2.9.0 | 0.24.0 | Previous stable |
| 2.8.0 | 0.22.1 | Two versions back |
| 2.6.0 | 0.21.0 | Recommended for broad compat |
| 2.5.0 | 0.20.0 | Minimum recommended |

**Source:** [PyTorch Versions Wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions), [PyPI torch](https://pypi.org/project/torch/), [PyPI torchvision](https://pypi.org/project/torchvision/)

**Recommendation:** Pin torch==2.6.0 and torchvision==0.21.0. This is a well-tested release pair that supports Python 3.10-3.13. Using torch 2.10.0 (latest) is also fine but is only 1 month old and may have fewer community battle-test miles. For a PoC, stability matters more than bleeding edge.

**IMPORTANT for requirements.txt:** PyTorch installation on systems with CUDA requires the `--index-url` or `--extra-index-url` flag pointing to `https://download.pytorch.org/whl/cu121` (or `cu124`). For CPU-only installs (Mac development), the default PyPI index works. The requirements.txt should include a comment documenting this, but the actual torch/torchvision lines should use version pins without the index URL (the user sets the index URL in their pip config or command line). Alternatively, include both a `requirements.txt` (CPU, default PyPI) and a note in README about CUDA installation.

### Albumentations Licensing Situation (Critical Finding)

| Library | Version | License | Status | Use For This Project? |
|---------|---------|---------|--------|-----------------------|
| albumentations | 2.0.8 | MIT | Archived (July 2025), last release May 2025 | YES -- stable, MIT, works with current PyTorch |
| albumentationsX | 2.0.13 | AGPL-3.0 / Commercial | Actively maintained | NO -- AGPL license is restrictive, 100% API compatible but licensing complicates distribution |

**Decision: Use albumentations 2.0.8 (MIT).** Rationale:
1. It is a direct drop-in -- same API, same imports as the planned code in ARCHITECTURE.md.
2. MIT license has no distribution restrictions -- important for a PoC that will be shared.
3. For a PoC scope, the library will not need bug fixes or new features -- 2.0.8 is stable.
4. If the project later needs actively maintained augmentation, torchvision transforms v2 covers most transforms except CLAHE. CLAHE can be sourced from OpenCV (`cv2.createCLAHE()`) directly with ~5 lines of code.

**Source:** [albumentations PyPI](https://pypi.org/project/albumentations/), [AlbumentationsX Blog Post](https://albumentations.ai/blog/2025/01-albumentationsx-dual-licensing/)

### Optional Dependencies

| Library | Pin Version | Purpose | When |
|---------|-------------|---------|------|
| streamlit | 1.54.0 | Demo app (Phase 8) | Only if demo is built |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| albumentations 2.0.8 (MIT archived) | AlbumentationsX (AGPL) | Active maintenance but AGPL license restricts distribution |
| albumentations 2.0.8 (MIT archived) | torchvision transforms v2 only | No CLAHE support natively; would need kornia or OpenCV for CLAHE |
| PyYAML | OmegaConf | OmegaConf adds dot-access and merge features but is an extra dependency; PyYAML + dict access is sufficient for PoC |
| requirements.txt | pyproject.toml | pyproject.toml is the modern standard but requirements.txt is explicitly required by DOCS-06 |

**Installation (for requirements.txt):**

```
# Core ML
torch==2.6.0
torchvision==0.21.0
timm==1.0.15

# Augmentation
albumentations==2.0.8

# Explainability
grad-cam==1.5.5

# Evaluation
scikit-learn==1.7.2
scipy==1.15.2

# Data handling
pandas==2.2.3
numpy==2.2.3

# Visualization
matplotlib==3.10.0
seaborn==0.13.2

# Utilities
tqdm==4.67.1
pyyaml==6.0.2
Pillow==11.1.0
```

## Architecture Patterns

### Recommended Project Structure

The directory structure from ARCHITECTURE.md is correct and does not need adjustments. Here it is annotated with what Phase 1 creates (everything is a placeholder):

```
AssureXRay/
|-- configs/
|   |-- default.yaml              # Phase 1: Full config with all hyperparams
|
|-- scripts/                       # Phase 1: Placeholder scripts with docstrings
|   |-- download.py                # Phase 2 implementation
|   |-- audit.py                   # Phase 2 implementation
|   |-- split.py                   # Phase 3 implementation
|   |-- train.py                   # Phase 4 implementation
|   |-- eval.py                    # Phase 5 implementation
|   |-- gradcam.py                 # Phase 6 implementation
|   |-- infer.py                   # Phase 6 implementation
|
|-- src/                           # Phase 1: __init__.py files + placeholder modules
|   |-- __init__.py
|   |-- data/
|   |   |-- __init__.py
|   |   |-- dataset.py             # Phase 3: BTXRDDataset class
|   |   |-- transforms.py          # Phase 3: Augmentation pipelines
|   |   |-- split_utils.py         # Phase 3: Split logic
|   |
|   |-- models/
|   |   |-- __init__.py
|   |   |-- classifier.py          # Phase 4: BoneTumorClassifier
|   |   |-- factory.py             # Phase 4: create_model()
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- metrics.py             # Phase 5: Metric computation
|   |   |-- visualization.py       # Phase 5: Plot generation
|   |   |-- bootstrap.py           # Phase 5: Bootstrap CIs
|   |
|   |-- explainability/
|   |   |-- __init__.py
|   |   |-- gradcam.py             # Phase 6: GradCAM wrapper
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- config.py              # Phase 1: Config loading (IMPLEMENTED)
|       |-- logging.py             # Phase 1: Logging setup (IMPLEMENTED)
|       |-- reproducibility.py     # Phase 1: Seed setting (IMPLEMENTED)
|
|-- app/                           # Phase 8: Streamlit demo
|   |-- app.py
|
|-- data_raw/                      # Gitignored, created by download script
|-- data/
|   |-- splits/                    # Phase 3: Split CSVs (committed)
|
|-- checkpoints/                   # Gitignored
|-- results/                       # Partially committed
|-- docs/                          # Documentation deliverables
|-- tests/                         # Unit tests
|-- notebooks/                     # Exploration notebooks
|
|-- .gitignore
|-- requirements.txt               # Phase 1: Pinned dependencies
|-- Makefile                       # Phase 1: Pipeline targets
|-- README.md                      # Phase 1: Setup instructions
```

### Pattern 1: Self-Documenting Makefile with Help Target

**What:** Every Makefile target has a `## description` comment. The default target (`make` with no args) prints formatted help.

**When to use:** Always -- this is the standard pattern for ML project Makefiles.

**Example:**
```makefile
.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: download
download: ## Download BTXRD dataset from figshare
	python scripts/download.py --config configs/default.yaml

.PHONY: audit
audit: ## Profile dataset quality and generate audit report
	python scripts/audit.py --config configs/default.yaml

.PHONY: split
split: ## Generate train/val/test split manifests
	python scripts/split.py --config configs/default.yaml

.PHONY: train
train: ## Train EfficientNet-B0 classifier
	python scripts/train.py --config configs/default.yaml

.PHONY: evaluate
evaluate: ## Evaluate model on both split strategies
	python scripts/eval.py --config configs/default.yaml

.PHONY: gradcam
gradcam: ## Generate Grad-CAM heatmaps for selected examples
	python scripts/gradcam.py --config configs/default.yaml

.PHONY: infer
infer: ## Run single-image inference
	python scripts/infer.py --config configs/default.yaml

.PHONY: report
report: ## Generate model card and PoC report
	@echo "Report generation is manual -- see docs/"

.PHONY: demo
demo: ## Launch Streamlit demo app
	streamlit run app/app.py

.PHONY: all
all: download audit split train evaluate gradcam report ## Run full pipeline
```

**Key details:**
- All targets are `.PHONY` because they do not produce files with the target name.
- The `help` target uses `grep` + `awk` to extract `## comment` descriptions.
- The `all` target chains the pipeline in dependency order.
- Each target passes `--config configs/default.yaml` for consistency.

**Source:** [Self-Documenting Makefiles](https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html), [Makefiles for ML Pipelines](https://robertmelton.com/posts/stop-worrying-and-learn-to-love-the-makefile/), [Made With ML Makefiles](https://madewithml.com/courses/mlops/makefile/)

### Pattern 2: YAML Config with Flat Load + argparse Override

**What:** Single YAML file loaded with `yaml.safe_load()`, returned as a nested dict. CLI overrides via argparse for key parameters.

**When to use:** Always for this PoC. No need for OmegaConf or Hydra.

**Example:**
```python
# src/utils/config.py
import yaml
from pathlib import Path
from typing import Any

def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load YAML config and apply CLI overrides.

    Args:
        config_path: Path to YAML config file.
        overrides: List of 'key.subkey=value' strings for CLI overrides.

    Returns:
        Config dictionary.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if overrides:
        for override in overrides:
            key, value = override.split("=", 1)
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d[k]
            # Attempt type coercion
            try:
                value = yaml.safe_load(value)
            except yaml.YAMLError:
                pass
            d[keys[-1]] = value

    return config
```

**Security note:** Always use `yaml.safe_load()`, never `yaml.load()`. The latter can execute arbitrary Python code from a YAML file.

### Pattern 3: Reproducibility Module

**What:** A single function that sets all random seeds for deterministic behavior.

**When to use:** Called at the start of every script.

**Example:**
```python
# src/utils/reproducibility.py
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds for reproducibility.

    Sets seeds for: random, numpy, torch (CPU + CUDA).
    Configures CuDNN for deterministic operation.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic CuDNN (trades ~10-15% speed for reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For PyTorch >= 2.0: enable deterministic algorithms globally
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
```

**Note:** `torch.use_deterministic_algorithms(True, warn_only=True)` is available in PyTorch >= 1.11. The `warn_only=True` flag logs a warning instead of raising an error when a non-deterministic operation is encountered. This is important because some PyTorch operations (e.g., `torch.nn.functional.interpolate` with certain modes) do not have deterministic implementations. The `CUBLAS_WORKSPACE_CONFIG` environment variable is needed for CUDA >= 10.2 to force deterministic cuBLAS behavior.

### Pattern 4: Placeholder Script Template

**What:** Each script in `scripts/` starts with a standardized template -- docstring, imports, argparse, config loading, seed setting.

**When to use:** For every placeholder script created in Phase 1.

**Example:**
```python
#!/usr/bin/env python3
"""Train EfficientNet-B0 classifier on BTXRD dataset.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --override training.lr=0.0001
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        help="Config overrides as key.subkey=value",
    )
    args = parser.parse_args()

    config = load_config(args.config, overrides=args.override)
    set_seed(config.get("seed", 42))

    # TODO: Implement in Phase N
    raise NotImplementedError("Training not yet implemented -- see Phase 4")


if __name__ == "__main__":
    main()
```

### Anti-Patterns to Avoid

- **sys.path hacks scattered everywhere:** Use a single consistent pattern at the top of each script (`PROJECT_ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(PROJECT_ROOT))`). An alternative is to make the project pip-installable via `pip install -e .` with a pyproject.toml, which avoids sys.path manipulation entirely. For a PoC, the sys.path approach is simpler and more transparent.

- **requirements.txt without pins:** Never use bare package names (`torch`, `pandas`). Always pin exact versions (`torch==2.6.0`). Unpinned dependencies break reproducibility when upstream packages release new versions.

- **Multiple YAML files from the start:** Start with a single `configs/default.yaml`. Experiment-specific overrides come later (Phase 4). Do not create experiment YAML files in Phase 1 -- they add confusion before there is anything to experiment with.

- **Mixing tabs and spaces in Makefile:** Makefile REQUIRES tabs for recipe lines. Spaces will silently fail. Configure your editor to use actual tabs in Makefiles.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| YAML config loading | Custom parser | `yaml.safe_load()` from PyYAML | Handles all YAML types, edge cases with anchors/aliases |
| Grad-CAM heatmaps | Custom gradient hooks | `grad-cam` library (pip install grad-cam) | Hook lifecycle, batch norm handling, overlay generation are tricky |
| Self-documenting Makefile help | Manual help text | grep+awk pattern on `## comments` | Auto-updates as targets change; never out of sync |
| .gitignore for Python/ML | Manual file list | Start from GitHub's Python.gitignore template | Covers 50+ patterns you would forget |
| Random seed management | Ad-hoc seed setting | Centralized `set_seed()` function | Must cover random, numpy, torch, cuda, cudnn -- easy to miss one |

**Key insight:** Phase 1 is infrastructure. Every minute spent building custom infrastructure is a minute not spent on the actual ML pipeline. Use established patterns exactly as documented.

## Common Pitfalls

### Pitfall 1: PyTorch/Torchvision Version Mismatch

**What goes wrong:** Installing `torch` and `torchvision` independently results in incompatible versions. Example: `torch==2.6.0` with `torchvision==0.25.0` causes import errors because torchvision 0.25.0 expects torch 2.10.0.

**Why it happens:** pip resolves dependencies independently. PyTorch's non-standard distribution (via pytorch.org index for CUDA builds) complicates resolution.

**How to avoid:** Always pin both torch AND torchvision as a matched pair in requirements.txt. Use the compatibility matrix above. In this project: `torch==2.6.0` + `torchvision==0.21.0`.

**Warning signs:** `ImportError` or `RuntimeError` on `import torchvision`. "torchvision requires torch X.Y but found Z.W" errors.

### Pitfall 2: Makefile Uses Spaces Instead of Tabs

**What goes wrong:** Makefile recipe lines use spaces instead of tabs. GNU Make requires literal tab characters for recipe lines. With spaces, you get `Makefile:N: *** missing separator. Stop.`

**Why it happens:** Many editors auto-convert tabs to spaces. Copy-pasting from web sources often converts tabs.

**How to avoid:** Configure your editor to use hard tabs in Makefiles (most editors support filetype-specific settings). After creating the Makefile, run `cat -A Makefile | head` to verify tab characters appear as `^I`.

**Warning signs:** `missing separator` error from make.

### Pitfall 3: Missing __init__.py Files

**What goes wrong:** Python cannot import modules from `src/` subdirectories because `__init__.py` files are missing. Error: `ModuleNotFoundError: No module named 'src.data'`.

**Why it happens:** Easy to forget when creating the directory structure. Python 3 supports namespace packages (no __init__.py needed) but only for packages in the Python path. Since `src/` is added via sys.path, regular packages (with __init__.py) are more reliable.

**How to avoid:** Create __init__.py in every directory under `src/`. They can be empty. The directory creation step in Phase 1 must include these files explicitly.

**Warning signs:** `ModuleNotFoundError` when importing from `src/`.

### Pitfall 4: .gitignore Missing Critical Patterns for ML Projects

**What goes wrong:** Large binary files (model checkpoints, raw images, JPEG datasets) get committed to git, bloating the repository from megabytes to gigabytes. Once committed, even deleting the files does not reduce repo size (git history retains them).

**Why it happens:** Standard Python .gitignore does not include ML-specific patterns like `*.pt`, `*.pth`, `data_raw/`, `checkpoints/`.

**How to avoid:** Start from GitHub's Python.gitignore and add ML-specific patterns. See the complete .gitignore template below.

**Warning signs:** `git status` shows unexpected large files. `git push` takes an unusually long time. GitHub warns about large files.

### Pitfall 5: YAML Config with Unsafe Load

**What goes wrong:** Using `yaml.load()` instead of `yaml.safe_load()` opens the door to arbitrary code execution from a YAML file. A crafted YAML file can execute shell commands on the host machine.

**Why it happens:** `yaml.load()` is the first function developers find in PyYAML docs. The security warning is easy to miss.

**How to avoid:** Always use `yaml.safe_load()`. Never use `yaml.load()` unless you have a specific reason AND use `yaml.SafeLoader` explicitly.

**Warning signs:** PyYAML deprecation warning: "calling yaml.load() without Loader=... is deprecated."

### Pitfall 6: requirements.txt Includes CUDA-Specific PyTorch URL

**What goes wrong:** `requirements.txt` includes `--extra-index-url https://download.pytorch.org/whl/cu121` which forces all users to download CUDA 12.1 wheels -- even on Mac (no CUDA) or systems with different CUDA versions.

**Why it happens:** PyTorch's installation instructions include the index URL, and developers copy it into requirements.txt.

**How to avoid:** Keep requirements.txt clean with just package pins. Document the CUDA installation in README.md. Users on CUDA systems add the index URL when running pip install. The CPU-only wheels on default PyPI work for development on Mac/CPU-only machines.

**Warning signs:** Mac users get `torch` installation errors or download 2GB CUDA wheels they do not need.

## Code Examples

### Complete .gitignore for ML Project

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
dist/
build/
*.egg

# Virtual environments
.venv/
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/

# ML-specific: Large binary files
data_raw/
checkpoints/
*.pt
*.pth
*.onnx
*.pkl

# ML-specific: Results (selectively commit -- keep metrics JSON, gitignore bulk images)
results/gradcam/*.png
results/*/training_log.csv

# Logs
logs/
*.log

# Secrets / environment
.env
.env.*

# Planning artifacts (optional -- remove if you want to commit these)
# .planning/
```

### Complete configs/default.yaml

```yaml
# AssureXRay Default Configuration
# All hyperparameters, paths, and experiment settings
# Override from CLI: python scripts/train.py --override training.lr=0.0001

data:
  raw_dir: data_raw
  splits_dir: data/splits
  image_size: 224
  num_workers: 4
  figshare_url: "https://figshare.com/ndownloader/articles/27865398"

model:
  backbone: efficientnet_b0
  pretrained: true
  num_classes: 3
  dropout: 0.2
  class_names:
    - Normal
    - Benign
    - Malignant

training:
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 50
  early_stopping_patience: 7
  scheduler: cosine
  class_weights: auto        # Computed from training set distribution
  loss_type: cross_entropy   # cross_entropy or focal
  split_strategy: stratified # stratified or center

evaluation:
  bootstrap_iterations: 1000
  confidence_level: 0.95
  split_strategies:
    - stratified
    - center

gradcam:
  target_layer: auto         # Auto-detect based on backbone
  examples_per_class: 3
  methods:
    - GradCAM

inference:
  default_checkpoint: checkpoints/best_stratified.pt

paths:
  checkpoints_dir: checkpoints
  results_dir: results
  docs_dir: docs
  logs_dir: logs

seed: 42
device: auto                 # auto-detect GPU/CPU/MPS
```

### requirements.txt Format

```
# AssureXRay Dependencies
# Python >=3.10, <3.13 (3.12 recommended)
#
# For GPU (CUDA 12.1):
#   pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
#
# For CPU only (Mac/development):
#   pip install -r requirements.txt

# Core ML framework (must be a matched pair)
torch==2.6.0
torchvision==0.21.0

# Model zoo
timm==1.0.15

# Augmentation (MIT-licensed archived version -- stable, no active maintenance)
albumentations==2.0.8

# Explainability
grad-cam==1.5.5

# Evaluation
scikit-learn==1.7.2
scipy==1.15.2

# Data handling
pandas==2.2.3
numpy==2.2.3
Pillow==11.1.0

# Visualization
matplotlib==3.10.0
seaborn==0.13.2

# Utilities
tqdm==4.67.1
pyyaml==6.0.2
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `torch.cuda.amp` for mixed precision | `torch.amp` (device-agnostic) | PyTorch 2.4+ | Use `torch.amp.autocast('cuda')` not `torch.cuda.amp.autocast()` |
| `torchvision.transforms` (v1) | `torchvision.transforms.v2` | torchvision 0.17+ (stable) | V2 supports tensors and PIL uniformly; V1 is legacy |
| albumentations (MIT, maintained) | albumentations archived; AlbumentationsX (AGPL) | July 2025 | Use archived 2.0.8 for MIT license; or switch to torchvision v2 + OpenCV CLAHE |
| `setup.py` + `requirements.txt` | `pyproject.toml` + `requirements.txt` | PEP 621, widely adopted 2024+ | pyproject.toml for metadata; requirements.txt for pinned installs |
| Python 3.10 as ML standard | Python 3.11-3.12 becoming standard | Late 2025 (scipy, sklearn, pandas drop 3.10) | New projects should target 3.11+ |

**Deprecated/outdated:**
- `torch.cuda.amp.autocast()`: Use `torch.amp.autocast('cuda')` instead (PyTorch 2.4+)
- `torchvision.transforms` (v1 API): Use `torchvision.transforms.v2` instead
- `yaml.load()` without Loader: Use `yaml.safe_load()` always
- albumentations (original) for new development: Archived July 2025, but still usable for existing projects

## pyproject.toml vs requirements.txt

**The project requirement DOCS-06 specifies `requirements.txt`.** This is the correct choice for this PoC:

| Factor | requirements.txt | pyproject.toml |
|--------|------------------|----------------|
| Explicit version pins | Yes (`torch==2.6.0`) | Typically uses ranges (`torch>=2.6`) |
| Reproducibility | Exact -- same versions on every install | Depends on lock file generation |
| Familiarity for ML teams | Universal | Gaining adoption but not universal |
| pip install simplicity | `pip install -r requirements.txt` | `pip install .` or `pip install -e .` |
| Metadata/packaging | No project metadata | Full project metadata |

**Recommendation:** Create `requirements.txt` (required by DOCS-06) AND a minimal `pyproject.toml` for project metadata. The pyproject.toml is not strictly needed but provides:
- A `[project]` section with name, version, description
- A `[tool.pytest]` section for test configuration (useful in later phases)

The `requirements.txt` is the source of truth for dependencies. The `pyproject.toml` is supplementary metadata.

## .gitignore Considerations for ML Projects

Beyond the standard Python .gitignore, ML projects must specifically address:

| Pattern | Why |
|---------|-----|
| `data_raw/` | Raw dataset (potentially GB of images) |
| `checkpoints/` | Model weights (50-200MB each) |
| `*.pt`, `*.pth` | PyTorch model files anywhere in the tree |
| `*.onnx` | Exported model files |
| `results/gradcam/*.png` | Bulk generated Grad-CAM overlays |
| `.ipynb_checkpoints/` | Jupyter auto-saves |
| `logs/` | Training logs (can regenerate) |
| `*.pkl` | Pickled Python objects |

**What to commit:**
| Pattern | Why |
|---------|-----|
| `data/splits/*.csv` | Critical for reproducibility (small files) |
| `results/*/metrics.json` | Key evaluation outputs (small) |
| `results/*/confusion_matrix.png` | Key visualizations (small) |
| `configs/*.yaml` | Experiment settings |
| `docs/*.md` | Documentation deliverables |

## Open Questions

1. **Exact torch/torchvision pin vs range:** The research recommends `torch==2.6.0` for stability, but the success criteria say "Python 3.10-3.12." If the team is comfortable with Python 3.11+ only, pinning to `torch==2.10.0` + `torchvision==0.25.0` and latest of all packages is cleaner. This is a decision point.

2. **pip install -e . vs sys.path hack:** The placeholder scripts use `sys.path.insert(0, str(PROJECT_ROOT))` which works but is fragile. An alternative is `pip install -e .` with a pyproject.toml that declares the `src/` package. This is cleaner but adds a setup step. For a PoC, the sys.path approach is fine; the pyproject.toml approach is better if the project grows.

3. **albumentations longevity:** The archived 2.0.8 works today. If a bug is discovered during later phases, there will be no upstream fix. The fallback plan is to replace the affected transform with a torchvision or OpenCV equivalent. CLAHE specifically can be replaced with `cv2.createCLAHE()` which is zero-dependency (OpenCV is already a transitive dependency of albumentations).

## Sources

### Primary (HIGH confidence)
- [PyPI torch 2.10.0](https://pypi.org/project/torch/) - Latest version verified
- [PyPI torchvision 0.25.0](https://pypi.org/project/torchvision/) - Latest version verified
- [PyPI timm 1.0.24](https://pypi.org/project/timm/) - Latest version verified
- [PyPI albumentations 2.0.8](https://pypi.org/project/albumentations/) - Latest (archived) version verified
- [PyPI grad-cam 1.5.5](https://pypi.org/project/grad-cam/) - Latest version verified
- [PyPI scikit-learn 1.8.0](https://pypi.org/project/scikit-learn/) - Latest version verified (requires Python >=3.11)
- [PyPI scipy 1.17.0](https://pypi.org/project/scipy/) - Latest version verified (requires Python >=3.11)
- [PyPI pandas 3.0.1](https://pypi.org/project/pandas/) - Latest version verified (requires Python >=3.11)
- [PyTorch Versions Wiki](https://github.com/pytorch/pytorch/wiki/PyTorch-Versions) - Version compatibility matrix
- [AlbumentationsX licensing blog](https://albumentations.ai/blog/2025/01-albumentationsx-dual-licensing/) - Licensing change details

### Secondary (MEDIUM confidence)
- [Self-Documenting Makefiles](https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html) - Makefile help pattern
- [Made With ML Makefiles](https://madewithml.com/courses/mlops/makefile/) - ML-specific Makefile patterns
- [Makefiles for ML Pipelines](https://robertmelton.com/posts/stop-worrying-and-learn-to-love-the-makefile/) - Pipeline automation patterns
- [Python .gitignore Clean Repository](https://www.pythoncentral.io/python-gitignore-clean-repository-management/) - .gitignore best practices
- [pyproject.toml vs requirements.txt](https://pydevtools.com/handbook/explanation/pyproject-vs-requirements/) - Modern dependency management

### Tertiary (LOW confidence)
- Specific version pins (e.g., `torch==2.6.0`) are the researcher's recommendation for stability based on the compatibility constraints. They have not been tested together in this specific combination. The planner should note that the first task should include a `pip install -r requirements.txt` smoke test.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - All versions verified against PyPI on 2026-02-19
- Architecture: HIGH - Directory structure from existing ARCHITECTURE.md, confirmed with standard patterns
- Pitfalls: HIGH - Well-known issues with PyTorch versioning, Makefile syntax, and ML .gitignore

**Research date:** 2026-02-19
**Valid until:** 2026-04-19 (60 days -- package versions are stable; albumentations licensing situation is settled)
