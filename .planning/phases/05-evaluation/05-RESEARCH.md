# Phase 5: Evaluation - Research

**Researched:** 2026-02-20
**Domain:** Multiclass classification evaluation metrics, bootstrap confidence intervals, ROC/PR curve visualization, clinical metric reporting
**Confidence:** HIGH

## Summary

Phase 5 implements comprehensive model evaluation for the BTXRD bone tumor classifier, producing clinically relevant metrics, publication-quality visualizations, and side-by-side comparison of both split strategies. The three stub modules (`src/evaluation/metrics.py`, `src/evaluation/bootstrap.py`, `src/evaluation/visualization.py`) and the placeholder `scripts/eval.py` must be completed to fulfill requirements EVAL-01 through EVAL-08.

All required libraries are already installed and pinned: scikit-learn 1.7.2 provides `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `confusion_matrix`, `multilabel_confusion_matrix`, and `classification_report`; matplotlib 3.10.0 and seaborn 0.13.2 handle all plotting; numpy 2.2.3 and scipy 1.15.2 support bootstrap resampling; torch 2.6.0 provides model inference. No new dependencies are needed.

The evaluation pipeline must: (1) load a trained checkpoint and run inference on the test set to collect predictions and softmax probabilities, (2) compute all metrics from those arrays (no re-running inference per metric), (3) generate all plots and save to the correct results subdirectory, (4) produce JSON files for bootstrap CIs, (5) generate a comparison table across both split strategies, and (6) include a comparison against the BTXRD paper's YOLOv8s-cls baseline. Key challenge: the stratified test set has only 51 Malignant samples, which means per-class bootstrap CIs for Malignant sensitivity will have wide intervals. This is a factual limitation, not a bug.

**Primary recommendation:** Structure the evaluation as a three-layer architecture: (1) `metrics.py` computes all scalar metrics from prediction arrays, (2) `visualization.py` generates all plots, (3) `bootstrap.py` handles confidence interval estimation. The `scripts/eval.py` orchestrates: load model, run inference once, pass arrays to all three modules, save outputs. Run both splits sequentially, then generate the comparison table.

## Standard Stack

### Core (all already in requirements.txt)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| scikit-learn | 1.7.2 | `roc_curve`, `roc_auc_score`, `precision_recall_curve`, `average_precision_score`, `confusion_matrix`, `multilabel_confusion_matrix`, `classification_report`, `LabelBinarizer` | De-facto standard for ML evaluation metrics; handles multiclass OvR natively |
| matplotlib | 3.10.0 | ROC curves, PR curves, base plotting infrastructure | Publication-quality plotting; already used in Phase 4 |
| seaborn | 0.13.2 | Confusion matrix heatmaps via `sns.heatmap()` | Better default styling than raw matplotlib for annotated heatmaps |
| numpy | 2.2.3 | Array operations, bootstrap resampling via `np.random.choice` | Foundation for all numerical work |
| scipy | 1.15.2 | Not strictly required (numpy suffices for percentile-based CIs) but available | Could be used for DeLong AUC test if needed |
| torch | 2.6.0 | Model loading, inference, softmax computation via `F.softmax` | Core framework; already used throughout |
| json (stdlib) | N/A | Saving bootstrap CI results and comparison tables | Lightweight, human-readable output |
| pandas | 2.2.3 | Comparison table formatting, CSV reading for test manifests | Already used for dataset loading |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| csv (stdlib) | N/A | Writing classification reports as CSV | Alternative to JSON for tabular data |
| pathlib (stdlib) | N/A | Path construction for output directories | Consistent with existing codebase |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Manual bootstrap loop | `confidenceinterval` PyPI package | Extra dependency; our 1000-iteration loop is ~30 lines, not worth adding a dependency |
| `seaborn.heatmap` for confusion matrix | `sklearn.metrics.ConfusionMatrixDisplay` | ConfusionMatrixDisplay is simpler but less flexible; seaborn allows dual annotation (counts + percentages), better colormap control, and the project already has seaborn |
| Manual OvR ROC curve computation | `sklearn.metrics.RocCurveDisplay.from_predictions` | Display helper is convenient for single plots, but we need per-class curves + macro on one figure, which requires manual composition |
| `scipy.stats.bootstrap` | Manual `np.random.choice` loop | scipy.stats.bootstrap is cleaner but our use case (multiple metrics per bootstrap sample) is simpler with a manual loop |

**Installation:** No new packages needed. All are in `requirements.txt`.

## Architecture Patterns

### Recommended Project Structure

```
src/evaluation/
  __init__.py         # Exports (already exists)
  metrics.py          # compute_metrics() -> dict of all scalar metrics
  bootstrap.py        # bootstrap_ci() -> dict of CIs for AUC + sensitivity
  visualization.py    # plot_roc_curves(), plot_pr_curves(), plot_confusion_matrix()

scripts/
  eval.py             # CLI entry point: load model -> inference -> metrics + plots + CIs

results/
  stratified/
    roc_curves.png
    pr_curves.png
    confusion_matrix.png
    confusion_matrix_normalized.png
    classification_report.json
    bootstrap_ci.json
    metrics_summary.json
  center_holdout/
    roc_curves.png
    pr_curves.png
    confusion_matrix.png
    confusion_matrix_normalized.png
    classification_report.json
    bootstrap_ci.json
    metrics_summary.json
  comparison_table.json
  comparison_table.csv
```

### Pattern 1: Inference-First Architecture

**What:** Run model inference once on the entire test set, collecting all predictions, softmax probabilities, and true labels into numpy arrays. All downstream metrics and visualizations consume these arrays -- no re-inference.

**When to use:** Always. This is the standard evaluation pipeline pattern.

**Example:**
```python
# Source: Standard PyTorch evaluation pattern
import torch
import torch.nn.functional as F
import numpy as np

def run_inference(model, dataloader, device):
    """Run inference on full test set, returning predictions and probabilities.

    Returns:
        y_true: np.ndarray of shape (N,) -- true labels (integer)
        y_pred: np.ndarray of shape (N,) -- predicted labels (integer)
        y_prob: np.ndarray of shape (N, num_classes) -- softmax probabilities
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)
    return y_true, y_pred, y_prob
```

**Critical detail:** Use `F.softmax(logits, dim=1)` to convert raw logits to probabilities. The model outputs logits (no softmax applied internally, as documented in `BTXRDClassifier`). scikit-learn's `roc_auc_score` with `multi_class='ovr'` requires probability estimates that sum to 1 across classes.

### Pattern 2: One-vs-Rest ROC Curves with Macro AUC

**What:** Compute per-class ROC curves using binarized labels and per-class probability scores, then compute macro-average AUC by interpolating all curves to a common FPR grid.

**When to use:** Required for EVAL-01.

**Example:**
```python
# Source: scikit-learn 1.7.2 documentation (verified via WebFetch)
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import numpy as np

def compute_roc_curves(y_true, y_prob, class_names):
    """Compute per-class and macro-average ROC curves.

    Args:
        y_true: (N,) integer labels
        y_prob: (N, C) softmax probabilities
        class_names: list of class name strings

    Returns:
        dict with fpr, tpr, roc_auc per class + macro
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr, tpr, roc_auc = {}, {}, {}

    # Per-class ROC
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC (interpolate to common grid)
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Can also use the built-in for verification:
    # macro_auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')

    return {"fpr": fpr, "tpr": tpr, "roc_auc": roc_auc}
```

**Verified facts (HIGH confidence):**
- `roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')` computes macro-average AUC for multiclass OvR
- `y_prob` must be shape `(N, C)` with probabilities summing to 1 per row
- `label_binarize(y_true, classes=[0, 1, 2])` produces `(N, 3)` binary matrix
- `roc_curve` returns `(fpr, tpr, thresholds)` for binary classification; use with binarized columns for OvR

### Pattern 3: One-vs-Rest PR Curves with Average Precision

**What:** Compute per-class precision-recall curves using binarized labels. scikit-learn's `precision_recall_curve` does not natively support multiclass, so manual OvR binarization is required.

**When to use:** Required for EVAL-02.

**Example:**
```python
# Source: scikit-learn 1.7.2 documentation (verified via WebFetch)
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

def compute_pr_curves(y_true, y_prob, class_names):
    """Compute per-class precision-recall curves and average precision.

    Returns:
        dict with precision, recall, ap per class
    """
    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    precision, recall, ap = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(
            y_bin[:, i], y_prob[:, i]
        )
        ap[i] = average_precision_score(y_bin[:, i], y_prob[:, i])

    return {"precision": precision, "recall": recall, "ap": ap}
```

**Verified facts (HIGH confidence):**
- `precision_recall_curve` does NOT support `multi_class` parameter -- must use manual OvR via `label_binarize`
- `average_precision_score` computes AP as the weighted mean of precisions at each threshold, equivalent to area under the PR curve
- PR curves are more informative than ROC for imbalanced classes (Malignant with only 51/107 test samples)

### Pattern 4: Specificity from Multilabel Confusion Matrix

**What:** Compute per-class specificity (TNR = TN / (TN + FP)) using scikit-learn's `multilabel_confusion_matrix`, which provides per-class 2x2 confusion matrices.

**When to use:** Required for EVAL-03.

**Example:**
```python
# Source: scikit-learn 1.7.2 multilabel_confusion_matrix docs (verified)
from sklearn.metrics import multilabel_confusion_matrix

def compute_sensitivity_specificity(y_true, y_pred, num_classes=3):
    """Compute per-class sensitivity and specificity.

    Uses multilabel_confusion_matrix which produces a (C, 2, 2) array
    where for each class: [[TN, FP], [FN, TP]].

    Returns:
        sensitivity: np.ndarray of shape (C,)
        specificity: np.ndarray of shape (C,)
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]

    sensitivity = tp / (tp + fn)  # recall / TPR
    specificity = tn / (tn + fp)  # TNR

    return sensitivity, specificity
```

**Verified facts (HIGH confidence):**
- `multilabel_confusion_matrix` with multiclass data produces `(n_classes, 2, 2)` array
- Layout per class: `[[TN, FP], [FN, TP]]`
- Sensitivity = TP / (TP + FN) = recall (identical to sklearn's recall_score)
- Specificity = TN / (TN + FP) -- not directly available in sklearn, must compute manually

### Pattern 5: Confusion Matrix Heatmaps (Dual: Absolute + Normalized)

**What:** Generate two confusion matrix plots: one with absolute counts and one row-normalized (showing per-class recall rates). Use seaborn for better styling.

**When to use:** Required for EVAL-04.

**Example:**
```python
# Source: seaborn 0.13.2 + scikit-learn 1.7.2 (verified)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, output_path, normalize=False):
    """Plot confusion matrix as annotated heatmap.

    Args:
        normalize: If True, normalize rows to show recall per class.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Confusion Matrix (Row-Normalized)'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix (Absolute Counts)'

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
        ax=ax,
    )
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

**Verified facts (HIGH confidence):**
- `confusion_matrix(y_true, y_pred)` returns `(C, C)` ndarray where `cm[i][j]` = count of true class `i` predicted as class `j`
- Row normalization: `cm / cm.sum(axis=1)[:, np.newaxis]` -- diagonal values become per-class recall
- seaborn `heatmap(annot=True, fmt='d')` for integer annotations; `fmt='.2f'` for normalized
- `cmap='Blues'` is the standard colormap for confusion matrices

### Pattern 6: Bootstrap Confidence Intervals

**What:** Resample the test set with replacement 1000 times, compute the metric of interest on each bootstrap sample, then take the 2.5th and 97.5th percentiles for a 95% CI.

**When to use:** Required for EVAL-07.

**Example:**
```python
# Source: Standard bootstrap methodology (verified against multiple sources)
import numpy as np
from sklearn.metrics import roc_auc_score

def bootstrap_metric(y_true, y_prob, y_pred, metric_fn, n_iterations=1000,
                     confidence_level=0.95, seed=42):
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: (N,) true labels
        y_prob: (N, C) softmax probabilities
        y_pred: (N,) predicted labels
        metric_fn: callable(y_true, y_prob, y_pred) -> float
        n_iterations: number of bootstrap samples
        confidence_level: CI level (0.95 for 95%)
        seed: random seed for reproducibility

    Returns:
        dict with 'mean', 'lower', 'upper', 'std'
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_iterations):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_true = y_true[indices]
        boot_prob = y_prob[indices]
        boot_pred = y_pred[indices]

        # Skip if a class is missing from bootstrap sample
        if len(np.unique(boot_true)) < y_prob.shape[1]:
            continue

        score = metric_fn(boot_true, boot_prob, boot_pred)
        scores.append(score)

    scores = np.array(scores)
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(scores, alpha * 100)
    upper = np.percentile(scores, (1 - alpha) * 100)

    return {
        'mean': float(np.mean(scores)),
        'lower': float(lower),
        'upper': float(upper),
        'std': float(np.std(scores)),
        'n_valid': len(scores),
    }
```

**Critical implementation details:**
- **Seed the RNG:** Use `np.random.RandomState(seed)` (not global `np.random.seed`) for isolated reproducibility
- **Skip degenerate samples:** When a bootstrap sample misses an entire class, `roc_auc_score` with `multi_class='ovr'` will fail. Skip these samples and report `n_valid` count.
- **Stratified test set warning:** With only 51 Malignant samples in the stratified test, approximately 1.6% of bootstrap samples will have 0 Malignant samples (probability = (1 - 51/564)^564 per sample, but since we're sampling with replacement, missing a class with 51/564 prevalence happens occasionally). Report `n_valid` to be transparent.
- **Use percentile method:** The BCa (bias-corrected and accelerated) method is more sophisticated but percentile is standard for medical imaging papers and sufficient for a PoC.

### Pattern 7: Comparison Table (Both Splits + Baseline)

**What:** After evaluating both splits, produce a side-by-side comparison table showing the generalization gap. Include BTXRD paper's YOLOv8s-cls baseline with appropriate caveats.

**When to use:** Required for EVAL-06 and EVAL-08.

**Example:**
```python
# Output format for comparison_table.json
{
    "stratified": {
        "macro_auc": 0.85,
        "per_class_auc": {"Normal": 0.90, "Benign": 0.82, "Malignant": 0.83},
        "per_class_sensitivity": {"Normal": 0.64, "Benign": 0.80, "Malignant": 0.67},
        "per_class_specificity": {"Normal": 0.88, "Benign": 0.75, "Malignant": 0.95},
        "accuracy": 0.71,
        "macro_f1": 0.68,
        ...
    },
    "center_holdout": {
        ...  # same structure
    },
    "btxrd_baseline": {
        "model": "YOLOv8s-cls",
        "split": "Random 80/20 (no patient grouping)",
        "image_size": 600,
        "epochs": 300,
        "per_class_precision": {"Normal": 0.913, "Benign": 0.881, "Malignant": 0.734},
        "per_class_recall": {"Normal": 0.898, "Benign": 0.875, "Malignant": 0.839},
        "caveats": [
            "Random 80/20 split without patient-level grouping (potential leakage)",
            "Validation set used for reporting (no separate test set)",
            "Different image size (600 vs 224)",
            "Different architecture (YOLOv8s-cls vs EfficientNet-B0)",
            "Different training duration (300 epochs vs ~50 with early stopping)"
        ]
    },
    "generalization_gap": {
        "description": "Difference between stratified and center-holdout performance",
        "macro_auc_gap": -0.05,
        "malignant_sensitivity_gap": -0.04,
        ...
    }
}
```

### Anti-Patterns to Avoid

- **Recomputing softmax probabilities inconsistently:** Always use `F.softmax(logits, dim=1)` on the raw model output. Do NOT use `torch.sigmoid` or apply softmax twice. The BTXRDClassifier outputs raw logits.
- **Using `roc_auc_score` without specifying `multi_class`:** Default is `multi_class='raise'` which will error on 3-class data. Always pass `multi_class='ovr'`.
- **Reporting only accuracy as the headline metric:** For medical imaging with class imbalance, sensitivity (recall) for the clinically important class (Malignant) is the headline metric. Overall accuracy is misleading when Normal is 50% of the data.
- **Bootstrap without class checking:** If a bootstrap sample happens to exclude all Malignant samples, per-class metrics will fail silently or produce NaN. Always check for class presence.
- **Hardcoding class count or names:** Use `class_names` from the checkpoint (`ckpt['class_names']`) and derive `num_classes` from it. This keeps evaluation consistent with training.
- **Plotting without `plt.close(fig)`:** Memory leak in batch evaluation. Always close figures after saving.
- **Using `model.train()` during evaluation:** Must use `model.eval()` to disable dropout and use running batch norm statistics. Already enforced in the inference function pattern.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| ROC AUC computation | Manual integration of FPR/TPR | `sklearn.metrics.roc_auc_score(y_true, y_prob, multi_class='ovr')` | Handles interpolation, edge cases, multiclass OvR correctly |
| Classification report | Manual precision/recall/F1 counting | `sklearn.metrics.classification_report(y_true, y_pred, target_names=class_names, output_dict=True)` | Handles macro/weighted averages, support counts, zero-division |
| Confusion matrix | Manual counting loop | `sklearn.metrics.confusion_matrix(y_true, y_pred)` | Handles label ordering, missing predictions correctly |
| Per-class specificity extraction | Manual TN/FP counting from full confusion matrix | `sklearn.metrics.multilabel_confusion_matrix(y_true, y_pred)` -> extract TN, FP per class | Produces per-class (2,2) matrices, avoids off-by-one errors in multi-class indexing |
| Label binarization for OvR | Manual one-hot encoding | `sklearn.preprocessing.label_binarize(y_true, classes=[0,1,2])` | Handles edge cases, ordering, consistent with sklearn metric functions |
| Heatmap plotting | Manual matplotlib imshow + text | `seaborn.heatmap(cm, annot=True, fmt='d', cmap='Blues')` | Handles colorbar, annotation formatting, axis labels, styling |
| Softmax from logits | Manual exp + normalize | `torch.nn.functional.softmax(logits, dim=1)` | Numerically stable (log-sum-exp trick internally) |

**Key insight:** The evaluation phase is almost entirely composed of well-solved problems. The only custom code needed is: (1) the inference loop gluing PyTorch model to numpy arrays, (2) the bootstrap loop applying metric functions to resampled data, and (3) the orchestration in `scripts/eval.py` that ties everything together.

## Common Pitfalls

### Pitfall 1: Bootstrap Fails on Missing Classes

**What goes wrong:** With only 51 Malignant samples in the stratified test set, some bootstrap samples will have 0 Malignant samples. `roc_auc_score` will raise `ValueError: Only one class present in y_true`.
**Why it happens:** Bootstrap sampling with replacement from a small class can produce all-zero samples for that class. With 51/564 Malignant prevalence, the probability of 0 Malignant in a bootstrap sample is approximately (1 - 51/564)^564 = ~2.7e-43 for a single sample (this is negligible for the FULL dataset), but `label_binarize` + per-class metrics can still fail if the binary column for a class is all zeros.
**How to avoid:** Wrap metric computation in a try/except, or check `len(np.unique(boot_true)) == num_classes` before computing. Count and report `n_valid` iterations. For per-class sensitivity bootstrap, check that the specific class exists in the bootstrap sample.
**Warning signs:** `n_valid` significantly less than `n_iterations`; `ValueError` from sklearn.

### Pitfall 2: Softmax Applied Twice

**What goes wrong:** If the model applies softmax internally AND the eval script applies `F.softmax()`, the probabilities will be "double-softmaxed" -- compressed toward uniform distribution, degrading AUC and all probability-based metrics.
**Why it happens:** Confusion about whether the model outputs logits or probabilities.
**How to avoid:** The `BTXRDClassifier` is documented to output raw logits (no softmax). `F.softmax(logits, dim=1)` must be applied exactly once in the inference function. Verify by checking that `y_prob.sum(axis=1)` is approximately 1.0 for all samples.
**Warning signs:** All class probabilities close to 0.33 (uniform); AUC near 0.5 despite trained model.

### Pitfall 3: Wrong Label Ordering in Metrics

**What goes wrong:** If `label_binarize` uses a different class ordering than the model's softmax output, ROC/PR curves will be computed against the wrong probability columns. E.g., if binarized column 0 = Normal but `y_prob[:, 0]` = Benign probability.
**Why it happens:** Mismatch between `CLASS_TO_IDX` mapping (Normal=0, Benign=1, Malignant=2) and the order of `roc_curve`/`label_binarize`.
**How to avoid:** Always use the same class-to-index mapping throughout. The dataset uses `{Normal: 0, Benign: 1, Malignant: 2}`. Pass `classes=[0, 1, 2]` to `label_binarize`. The model's output column order matches training label indices (position 0 = Normal logit, etc.).
**Warning signs:** Per-class AUC values that don't match expected performance (e.g., Normal AUC < 0.5 when Normal recall is 0.64).

### Pitfall 4: Results Directory Naming Inconsistency

**What goes wrong:** Evaluation writes to `results/center/` but the established convention from Phase 4 is `results/center_holdout/`. The comparison table can't find results from both splits.
**Why it happens:** The split strategy is called "center" in the config but "center_holdout" in the directory naming (from Phase 4).
**How to avoid:** Use the same mapping as `scripts/train.py`: `"stratified" -> "stratified"`, `"center" -> "center_holdout"` for directory names. This is already established in the training script.
**Warning signs:** Empty results directories; FileNotFoundError when generating comparison table.

### Pitfall 5: Checkpoint Loading Without Model Creation

**What goes wrong:** Attempting to load `model_state_dict` into a model created with different architecture parameters (e.g., wrong `num_classes` or `backbone`).
**Why it happens:** Creating the model with default parameters instead of using the config stored in the checkpoint.
**How to avoid:** Load the checkpoint first, extract `config` from it, then use `create_model(checkpoint['config'])` to ensure architecture matches. Then load `model.load_state_dict(checkpoint['model_state_dict'])`.
**Warning signs:** `RuntimeError: Error(s) in loading state_dict: size mismatch`.

### Pitfall 6: Reporting Validation Metrics Instead of Test Metrics

**What goes wrong:** Evaluating on the validation set instead of the test set, or mixing up val/test file paths.
**Why it happens:** The training script evaluates on val set; copying that pattern without changing to test CSV.
**How to avoid:** Explicitly use `{split_prefix}_test.csv` for evaluation, not `{split_prefix}_val.csv`. Test CSVs are: `data/splits/stratified_test.csv` (564 samples) and `data/splits/center_test.csv` (808 samples).
**Warning signs:** Metrics suspiciously close to training validation metrics; sample counts matching val set sizes.

### Pitfall 7: matplotlib Backend on Headless Systems

**What goes wrong:** `plt.show()` or default backend tries to open a GUI window, causing crash in CI or headless environments.
**Why it happens:** matplotlib defaults to interactive backend.
**How to avoid:** Set `matplotlib.use('Agg')` before any pyplot import, as already done in `scripts/train.py`. Never call `plt.show()` in the eval script -- only `fig.savefig()` + `plt.close(fig)`.
**Warning signs:** `_tkinter.TclError: no display name`.

## Code Examples

### Complete Inference Function

```python
# Source: Verified PyTorch 2.6.0 pattern
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def run_inference(model, dataloader, device):
    """Run inference on full dataset, return arrays for metric computation.

    Args:
        model: Trained BTXRDClassifier in eval mode.
        dataloader: Test DataLoader (shuffle=False, drop_last=False).
        device: torch.device to run inference on.

    Returns:
        y_true: np.ndarray shape (N,) int -- true class indices
        y_pred: np.ndarray shape (N,) int -- predicted class indices
        y_prob: np.ndarray shape (N, C) float -- softmax probabilities
    """
    model.eval()
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Inference"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_labels.extend(labels.numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.array(all_labels, dtype=np.int64)
    y_prob = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    return y_true, y_pred, y_prob
```

### Complete Metrics Computation

```python
# Source: scikit-learn 1.7.2 (verified via official docs WebFetch)
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix, multilabel_confusion_matrix,
    classification_report, recall_score,
)
from sklearn.preprocessing import label_binarize
import numpy as np

def compute_all_metrics(y_true, y_pred, y_prob, class_names):
    """Compute all evaluation metrics from prediction arrays.

    Args:
        y_true: (N,) true labels
        y_pred: (N,) predicted labels
        y_prob: (N, C) softmax probabilities
        class_names: ["Normal", "Benign", "Malignant"]

    Returns:
        dict containing all metrics needed for EVAL-01 through EVAL-05
    """
    num_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    # EVAL-01: ROC AUC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = float(auc(fpr[i], tpr[i]))
    macro_auc = float(roc_auc_score(
        y_true, y_prob, multi_class='ovr', average='macro'
    ))

    # EVAL-02: PR AUC
    precision_curves, recall_curves, ap = {}, {}, {}
    for i in range(num_classes):
        precision_curves[i], recall_curves[i], _ = precision_recall_curve(
            y_bin[:, i], y_prob[:, i]
        )
        ap[i] = float(average_precision_score(y_bin[:, i], y_prob[:, i]))

    # EVAL-03: Sensitivity and specificity
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # EVAL-04: Confusion matrix (raw)
    cm = confusion_matrix(y_true, y_pred)

    # EVAL-05: Classification report
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        # ROC data (for plotting)
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "roc_auc": {class_names[i]: roc_auc[i] for i in range(num_classes)},
        "macro_auc": macro_auc,
        # PR data (for plotting)
        "pr_precision": precision_curves,
        "pr_recall": recall_curves,
        "average_precision": {class_names[i]: ap[i] for i in range(num_classes)},
        # Sensitivity / specificity
        "sensitivity": {class_names[i]: float(sensitivity[i]) for i in range(num_classes)},
        "specificity": {class_names[i]: float(specificity[i]) for i in range(num_classes)},
        # Confusion matrix (for plotting)
        "confusion_matrix": cm,
        # Classification report
        "classification_report": report,
        # Headline metric
        "malignant_sensitivity": float(sensitivity[2]),  # Index 2 = Malignant
        "accuracy": float(report["accuracy"]),
    }
```

### Bootstrap CI for AUC and Sensitivity

```python
# Source: Standard bootstrap methodology
import numpy as np
from sklearn.metrics import roc_auc_score, recall_score

def bootstrap_confidence_intervals(y_true, y_pred, y_prob, class_names,
                                     n_iterations=1000, confidence_level=0.95,
                                     seed=42):
    """Compute bootstrap 95% CIs for macro AUC and per-class sensitivity.

    Returns:
        dict with CIs for each metric
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    num_classes = len(class_names)
    alpha = (1 - confidence_level) / 2

    # Collectors
    auc_scores = []
    sensitivity_scores = {i: [] for i in range(num_classes)}

    for _ in range(n_iterations):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        bt = y_true[idx]
        bp = y_pred[idx]
        bprob = y_prob[idx]

        # Check all classes present
        if len(np.unique(bt)) < num_classes:
            continue

        # Macro AUC
        try:
            auc_val = roc_auc_score(bt, bprob, multi_class='ovr', average='macro')
            auc_scores.append(auc_val)
        except ValueError:
            continue

        # Per-class sensitivity
        per_class = recall_score(bt, bp, average=None,
                                 labels=list(range(num_classes)),
                                 zero_division=0)
        for i in range(num_classes):
            sensitivity_scores[i].append(per_class[i])

    # Compute CIs
    results = {}

    auc_arr = np.array(auc_scores)
    results["macro_auc"] = {
        "mean": float(np.mean(auc_arr)),
        "ci_lower": float(np.percentile(auc_arr, alpha * 100)),
        "ci_upper": float(np.percentile(auc_arr, (1 - alpha) * 100)),
        "n_valid": len(auc_scores),
    }

    for i in range(num_classes):
        arr = np.array(sensitivity_scores[i])
        results[f"sensitivity_{class_names[i]}"] = {
            "mean": float(np.mean(arr)),
            "ci_lower": float(np.percentile(arr, alpha * 100)),
            "ci_upper": float(np.percentile(arr, (1 - alpha) * 100)),
            "n_valid": len(arr),
        }

    return results
```

### Confusion Matrix Heatmap (Dual Plot)

```python
# Source: seaborn 0.13.2 + matplotlib 3.10.0 (verified)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confusion_matrices(cm, class_names, output_dir):
    """Save both absolute and row-normalized confusion matrix heatmaps.

    Args:
        cm: (C, C) confusion matrix from sklearn
        class_names: list of class name strings
        output_dir: Path to save PNG files
    """
    # Absolute counts
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (Absolute Counts)')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix.png', dpi=150)
    plt.close(fig)

    # Row-normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', square=True,
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, vmin=0, vmax=1, ax=ax)
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.set_title('Confusion Matrix (Row-Normalized)')
    fig.tight_layout()
    fig.savefig(output_dir / 'confusion_matrix_normalized.png', dpi=150)
    plt.close(fig)
```

### ROC Curve Multi-Class Plot

```python
# Source: scikit-learn multiclass ROC example (verified via WebFetch)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_roc_curves(metrics, class_names, output_path):
    """Plot one-vs-rest ROC curves + macro-average on single figure.

    Args:
        metrics: dict from compute_all_metrics() with roc_fpr, roc_tpr, roc_auc
        output_path: Path to save PNG
    """
    colors = ['#2196F3', '#FF9800', '#F44336']  # Blue, Orange, Red
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Per-class curves
    for i in range(num_classes):
        ax.plot(
            metrics["roc_fpr"][i], metrics["roc_tpr"][i],
            color=colors[i], linewidth=2,
            label=f'{class_names[i]} (AUC = {metrics["roc_auc"][class_names[i]]:.3f})'
        )

    # Macro-average (if computed with interpolation)
    # Compute macro from per-class curves
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)
    for i in range(num_classes):
        mean_tpr += np.interp(fpr_grid, metrics["roc_fpr"][i], metrics["roc_tpr"][i])
    mean_tpr /= num_classes
    ax.plot(fpr_grid, mean_tpr, color='navy', linewidth=2, linestyle='--',
            label=f'Macro-average (AUC = {metrics["macro_auc"]:.3f})')

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
```

### Model Loading for Evaluation

```python
# Source: Verified against existing factory.py and checkpoint format
import torch
from src.models.factory import create_model, load_checkpoint, get_device
from src.data.dataset import BTXRDDataset, create_dataloader
from src.data.transforms import get_test_transforms

def load_model_for_eval(checkpoint_path, device):
    """Load trained model from checkpoint for evaluation.

    Uses config stored IN the checkpoint to ensure model architecture matches.

    Returns:
        model: BTXRDClassifier in eval mode on device
        config: training config dict from checkpoint
        class_names: list of class names
    """
    ckpt = load_checkpoint(checkpoint_path, device="cpu")
    config = ckpt["config"]
    class_names = ckpt["class_names"]

    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, config, class_names
```

## BTXRD Paper Baseline (EVAL-08)

### Verified Baseline Results

From the BTXRD paper (Yao et al., Scientific Data 2025, PMC11739492):

**Model:** YOLOv8s-cls
**Split:** Random 80/20 train/validation (no separate test set, no patient grouping)
**Image size:** 600 pixels
**Training:** 300 epochs with official pretrained weights
**Hardware:** NVIDIA GeForce RTX 4090 (24GB VRAM)

| Class | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Normal | 0.913 | 0.898 | 0.904 |
| Benign | 0.881 | 0.875 | 0.899 |
| Malignant | 0.734 | 0.839 | 0.965 |

**Important caveats for comparison (MUST be documented):**

1. **Split methodology:** Paper used random 80/20 split without patient-level grouping. Their "validation" set is what we'd call a test set, but with potential same-patient leakage inflating numbers.
2. **No separate test set:** Paper evaluates on the validation set used during training, not a held-out test set.
3. **Image size:** Paper used 600px input; we use 224px (standard for EfficientNet-B0).
4. **Architecture:** YOLOv8s-cls is a classification-adapted YOLO model; EfficientNet-B0 is a pure classification model.
5. **Training duration:** Paper trained for 300 epochs; we use ~50 epochs with early stopping.
6. **Metric differences:** Paper reports mAP@0.5 which is a detection metric adapted for classification; we report standard classification metrics (AUC, F1, etc.).
7. **No confidence intervals:** Paper does not report bootstrap CIs.

**Recommendation for comparison table:** Present the baseline numbers alongside our results with a prominent caveat section explaining why direct comparison is imperfect. Frame it as context, not as a head-to-head benchmark.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Accuracy as headline metric | Sensitivity/specificity as headline, AUC for discrimination | Best practice in medical imaging literature ~2020+ | Accuracy is misleading with class imbalance; sensitivity for clinically important class (Malignant) is what matters |
| Point estimates only | Bootstrap CIs as standard | Became standard ~2019+ in clinical ML papers | Confidence intervals are now expected in medical imaging publications |
| Single test set evaluation | Multiple split strategies to test generalization | Current best practice for multi-center datasets | Center-holdout reveals domain shift that stratified split conceals |
| Manual confusion matrix | `ConfusionMatrixDisplay` or seaborn heatmap | sklearn 1.0+ (2021) | Better visualization, less boilerplate |
| `sklearn.metrics.roc_curve` + manual plot | Still the standard for multiclass OvR | Unchanged | `RocCurveDisplay` helper exists but manual composition gives more control for multi-class |

**Deprecated/outdated:**
- Using `micro` averaging for AUC in imbalanced settings: `macro` is preferred to give equal weight to each class regardless of support
- Reporting only accuracy for multi-class medical classifiers: sensitivity and specificity per class are required

## Open Questions

1. **Should we also evaluate on the validation set for reference?**
   - What we know: The primary evaluation must be on the TEST set. But validation metrics from training are already logged in `results/{split}/training_log.csv`.
   - What's unclear: Whether to also run the full evaluation pipeline (ROC/PR/CM/bootstrap) on the validation set.
   - Recommendation: Evaluate on TEST SET ONLY for the formal evaluation. The training log already has per-epoch val metrics. Running full eval on val would be redundant and could create confusion about which numbers are "official."

2. **How to handle NaN/inf in bootstrap when class is missing?**
   - What we know: With 51 Malignant test samples (stratified), most bootstrap samples will include at least 1 Malignant. But edge cases can occur.
   - What's unclear: Whether to use try/except or pre-check for class presence.
   - Recommendation: Pre-check with `len(np.unique(boot_true)) < num_classes` and skip the iteration. Report `n_valid` to be transparent. This is cleaner than catching exceptions.

3. **Should the comparison table be JSON, CSV, or both?**
   - What we know: JSON is more structured and can represent nested data. CSV is easier to view.
   - What's unclear: Which format downstream phases (report, docs) will consume.
   - Recommendation: Save both. JSON for programmatic consumption (Phase 7 report generation). CSV for quick human review. The JSON is the source of truth.

4. **Center 3 Normal class warning**
   - What we know: Center 3 has only 27 Normal images. In the center-holdout test set, these 27 images are part of the 286 Normal test samples (along with Center 2's 259 Normal images).
   - What's unclear: Whether to break down center-holdout test metrics by source center.
   - Recommendation: Report aggregate center-holdout metrics as required. A per-center breakdown is valuable context but may be better suited for the Phase 7 report narrative rather than the eval script output. If time permits, add a per-center breakdown, but it is not required by EVAL-01 through EVAL-08.

## Test Set Statistics

For reference when interpreting evaluation results:

| Metric | Stratified Test | Center-Holdout Test |
|--------|-----------------|---------------------|
| Total samples | 564 | 808 |
| Normal | 282 (50.0%) | 286 (35.4%) |
| Benign | 231 (41.0%) | 415 (51.4%) |
| Malignant | 51 (9.0%) | 107 (13.2%) |
| Source | All centers (random stratified) | Centers 2 + 3 only |

**Impact on metrics:**
- Stratified test: Only 51 Malignant samples means wider bootstrap CIs for Malignant metrics
- Center-holdout test: Domain shift from training (Center 1) to test (Centers 2+3) means lower expected performance
- Both: Malignant is minority class in both sets, making sensitivity the critical metric

## Sources

### Primary (HIGH confidence)

- **scikit-learn 1.7.2 roc_auc_score docs** (WebFetched): API signature, multi_class='ovr' parameter, y_score format requirements -- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- **scikit-learn 1.7.2 multiclass ROC example** (WebFetched): Complete code pattern for OvR ROC curves with macro averaging -- https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
- **scikit-learn 1.7.2 precision_recall_curve docs** (WebFetched): Manual OvR required for multiclass, average_precision_score usage -- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html
- **scikit-learn 1.7.2 confusion_matrix docs** (WebFetched): normalize parameter ('true', 'pred', 'all'), ConfusionMatrixDisplay usage -- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
- **scikit-learn 1.7.2 classification_report docs** (WebFetched): output_dict=True format, target_names parameter -- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
- **scikit-learn 1.7.2 multilabel_confusion_matrix docs** (WebSearched + verified): Per-class (2,2) matrices for specificity computation -- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.multilabel_confusion_matrix.html
- **BTXRD paper baseline results** (WebFetched from PMC): YOLOv8s-cls per-class precision/recall/mAP, training configuration -- https://pmc.ncbi.nlm.nih.gov/articles/PMC11739492/
- **seaborn 0.13.2 heatmap docs** (WebFetched): annot, fmt, cmap, square parameters for confusion matrix visualization -- https://seaborn.pydata.org/generated/seaborn.heatmap.html
- **Existing codebase** (Read directly): `src/models/factory.py` (load_checkpoint, create_model, get_device), `src/data/dataset.py` (BTXRDDataset, create_dataloader), `src/data/transforms.py` (get_test_transforms), `src/models/classifier.py` (BTXRDClassifier outputs raw logits), checkpoint format verified

### Secondary (MEDIUM confidence)

- **Medical imaging evaluation best practices**: Sensitivity/specificity as standard metrics, bootstrap CIs expected in clinical ML papers -- https://www.sciencedirect.com/science/article/pii/S3050577125000283
- **Bootstrap methodology**: Percentile method for 95% CIs, handling degenerate samples -- https://machinelearningmastery.com/calculate-bootstrap-confidence-intervals-machine-learning-results-python/

### Tertiary (LOW confidence)

- None -- all findings verified with official documentation or codebase inspection

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- All libraries installed, APIs verified via official documentation WebFetch; no dependency gaps
- Architecture: HIGH -- Inference pattern verified against existing model/dataset code; checkpoint format inspected; all function signatures confirmed
- Pitfalls: HIGH -- Pitfalls derived from verified API behavior (multiclass roc_auc_score requires multi_class param, softmax must be applied once, label ordering confirmed in codebase, bootstrap class-missing edge case documented in sklearn)
- BTXRD baseline: HIGH -- Results extracted directly from PMC full-text article with exact numbers

**Research date:** 2026-02-20
**Valid until:** 2026-03-20 (stable domain; scikit-learn 1.7.2, matplotlib 3.10.0, seaborn 0.13.2 are all pinned)
