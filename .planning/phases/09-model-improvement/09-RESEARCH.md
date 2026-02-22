# Phase 9: Model Improvement - Research

**Researched:** 2026-02-22
**Domain:** Annotation-guided training, architecture upgrades, medical image classification
**Confidence:** MEDIUM (verified against codebase + timm library; paper methodology partially verified)

## Summary

Phase 9 aims to close the accuracy gap between our EfficientNet-B0 baseline (67.9% stratified accuracy, 47.2% center-holdout) and the BTXRD paper's YOLOv8s-cls results (Normal 91.3%, Benign 88.1%, Malignant 73.4% precision). The gap is partly explained by differences in split strategy (our stratified/center-holdout vs their random 80/20), image size (224px vs 600px), architecture, and training duration (5 epochs vs 300 epochs).

Three improvement axes are available: (1) annotation-guided training using LabelMe masks to focus model attention on tumor regions, (2) architecture upgrades from EfficientNet-B0 to B2/B3 and ResNet-50-CBAM, and (3) training recipe improvements (larger image size, longer training, label smoothing, warmup). The codebase already has `BTXRDAnnotatedDataset` implementing ROI-guided dimming and config changes staged for B3 at 380px -- these just need to be trained and evaluated.

**Critical discovery:** The current config (`default.yaml`) has already been modified (uncommitted) to EfficientNet-B3 at 380px with annotation-guided training, but the existing checkpoints were trained with EfficientNet-B0 at 224px without annotations. Phase 9 needs to systematically train and evaluate multiple configurations.

**Primary recommendation:** Run a structured experiment grid: (1) retrain EfficientNet-B0 at 224px as a clean baseline, (2) train EfficientNet-B3 at 300px with annotation-guided dataset, (3) train ResNet-50-CBAM at 224px with annotation-guided dataset, (4) evaluate all on both split strategies using the existing Phase 5 pipeline, and (5) select the best model as the new default.

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| timm | 1.0.15 | Model creation (EfficientNet-B0/B2/B3, ResNet-50+CBAM) | Already installed; supports all target architectures including CBAM via `block_args` |
| PyTorch | 2.6.0 | Training framework | Already installed; MPS backend available |
| albumentations | 2.0.8 | Image augmentation | Already installed; handles mask co-transforms |
| pytorch-grad-cam | (installed) | Explainability | Already installed; works with all architectures via `gradcam_target_layer` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| torch.amp | built-in | Mixed precision training | If VRAM is tight with larger models/image sizes |
| pandas | (installed) | Experiment results tracking | Compare metrics across runs |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| ResNet-50-CBAM via timm block_args | Standalone CBAM implementation (luuuyi/CBAM.PyTorch) | timm's built-in CBAM is verified working and uses pretrained ResNet-50 weights; standalone requires manual weight loading |
| EfficientNet-B3 | EfficientNetV2-S | V2 is newer/faster but less comparable to paper's EfficientNet references; V2 available in timm (`efficientnetv2_rw_s`) |
| Multi-task learning (cls + seg) | ROI-guided dimming (already implemented) | Multi-task adds significant complexity; dimming approach already coded and simpler |

## Architecture Patterns

### Recommended Project Structure
```
configs/
├── default.yaml             # Default config (will point to best model after Phase 9)
├── experiment_b0_baseline.yaml   # Clean B0 baseline config
├── experiment_b3_annotated.yaml  # B3 + annotation-guided config
├── experiment_resnet50_cbam.yaml # ResNet-50-CBAM config
src/
├── models/
│   ├── classifier.py        # BTXRDClassifier (UPDATE: handle ResNet gradcam_target_layer)
│   └── factory.py           # create_model (UPDATE: pass block_args for CBAM)
├── data/
│   ├── dataset.py           # BTXRDAnnotatedDataset (EXISTING: already implements ROI dimming)
│   └── transforms.py        # get_train_transforms (UPDATE: parameterize image_size)
scripts/
├── train.py                 # Training loop (EXISTING: already supports annotated dataset)
├── eval.py                  # Evaluation pipeline (EXISTING: reuse as-is)
├── gradcam.py               # Grad-CAM generation (EXISTING: reuse as-is)
└── benchmark.py             # NEW: orchestrate experiment grid and compare results
results/
├── experiments/             # NEW: per-experiment results subdirectories
│   ├── b0_224_baseline/
│   ├── b3_300_annotated/
│   └── resnet50_cbam_224/
```

### Pattern 1: Experiment Config Overrides
**What:** Use the existing `--override` CLI mechanism to parameterize experiments without duplicating config files.
**When to use:** Running the experiment grid.
**Example:**
```bash
# Train B3 with annotations at 300px
python scripts/train.py --config configs/default.yaml \
  --override model.backbone=efficientnet_b3 \
  --override data.image_size=300 \
  --override training.batch_size=16 \
  --override model.dropout=0.3

# Train ResNet-50-CBAM at 224px
python scripts/train.py --config configs/default.yaml \
  --override model.backbone=resnet50 \
  --override data.image_size=224 \
  --override training.batch_size=16
```

### Pattern 2: ResNet-50-CBAM via timm block_args
**What:** Use timm's attention factory to add CBAM attention to ResNet-50.
**When to use:** Creating ResNet-50-CBAM model.
**Example (verified working in timm 1.0.15):**
```python
import timm

# CBAM is available in timm's attention factory
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=3,
    drop_rate=0.3,
    block_args=dict(attn_layer='cbam'),
)
# Result: 26.0M params, 2048 features, 48 CBAM modules
```

### Pattern 3: Grad-CAM Target Layer per Architecture
**What:** Different architectures need different Grad-CAM target layers.
**When to use:** Updating `BTXRDClassifier.gradcam_target_layer` property.
**Example:**
```python
@property
def gradcam_target_layer(self) -> nn.Module:
    # EfficientNet family: model.bn2 (BatchNormAct2d before global avg pool)
    if hasattr(self.model, 'bn2'):
        return self.model.bn2
    # ResNet family: layer4[-1] (last bottleneck block)
    if hasattr(self.model, 'layer4'):
        return self.model.layer4[-1]
    # Fallback
    for module in reversed(list(self.model.modules())):
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
            return module
    raise AttributeError("Cannot find Grad-CAM target layer")
```

### Pattern 4: Annotation-Guided ROI Dimming (Already Implemented)
**What:** `BTXRDAnnotatedDataset` creates additional training samples where non-tumor regions are dimmed by 50%.
**When to use:** Training with annotation guidance.
**Key behavior:**
- Doubles tumor class samples (Benign + Malignant) with ROI-guided versions
- Normal images pass through unchanged (no annotations exist)
- Effectively addresses class imbalance while guiding attention
- `dim_factor=0.5` dims non-ROI regions to 50% brightness
- Already integrated into `scripts/train.py` when `data.annotations_dir` is set in config

### Anti-Patterns to Avoid
- **Training only one split strategy:** Always train both stratified AND center-holdout to measure generalization gap.
- **Changing image size without adjusting batch size:** Larger images (300px vs 224px) consume ~1.8x more activation memory per sample. Reduce batch size accordingly.
- **Comparing models trained for different durations:** Use early stopping with same patience across all experiments for fair comparison.
- **Overwriting checkpoints:** Use experiment-specific checkpoint names (e.g., `best_stratified_b3.pt`) to preserve all results.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| CBAM attention module | Custom CBAM implementation | `timm.create_model('resnet50', block_args=dict(attn_layer='cbam'))` | timm's CBAM is verified working (48 modules injected), uses pretrained weights correctly |
| Annotation mask loading | New mask loader | `load_annotation_mask_raw()` in `src/data/dataset.py` | Already handles polygon + rectangle LabelMe shapes |
| ROI-guided training samples | Custom augmentation | `BTXRDAnnotatedDataset` in `src/data/dataset.py` | Already implemented with configurable dim_factor |
| Model evaluation pipeline | Custom eval script | `scripts/eval.py` | Full pipeline: inference, metrics, ROC/PR curves, confusion matrix, bootstrap CIs, comparison table |
| Grad-CAM visualization | Custom implementation | `src/explainability/gradcam.py` + `scripts/gradcam.py` | Handles all architectures via gradcam_target_layer property |
| Class weight computation | Manual weight calculation | `compute_class_weights()` in `src/models/factory.py` | Inverse-frequency weighting matching sklearn's balanced mode |

**Key insight:** The codebase already has ~80% of what Phase 9 needs. The main work is: (1) updating `classifier.py` and `factory.py` to handle ResNet+CBAM, (2) creating a benchmark orchestration script, and (3) running experiments.

## Common Pitfalls

### Pitfall 1: Unfair Architecture Comparison
**What goes wrong:** Comparing models trained with different hyperparameters, image sizes, or training durations.
**Why it happens:** Each architecture has different optimal settings (B3 wants 288-300px, ResNet wants 224px).
**How to avoid:** Fix what should be fixed (same splits, same augmentation pipeline, same early stopping patience, same seed). Allow architecture-specific settings for: image size (use each model's default), batch size (to fit memory), and dropout rate.
**Warning signs:** One model trains for 5 epochs while another trains for 50.

### Pitfall 2: Image Size Mismatch After Architecture Change
**What goes wrong:** Training EfficientNet-B3 at 224px instead of its native 288px, or ResNet-50 at 300px unnecessarily.
**Why it happens:** Config image_size is global, not per-model.
**How to avoid:** Use native image sizes from timm defaults: B0=224, B2=260, B3=288 (or 300 as common round number), ResNet-50=224. Pass via `--override data.image_size=N`.
**Warning signs:** Accuracy worse than smaller model despite more parameters.

### Pitfall 3: Grad-CAM Target Layer Wrong for ResNet
**What goes wrong:** Using `model.bn2` (EfficientNet-specific) for ResNet, getting an error or meaningless heatmaps.
**Why it happens:** Current `gradcam_target_layer` property has a fallback but it may select the wrong layer.
**How to avoid:** Add explicit ResNet layer4 detection (see Pattern 3 above). For ResNet-50-CBAM, `layer4[-1]` children are: `['conv1', 'bn1', 'act1', 'conv2', 'bn2', 'drop_block', 'act2', 'aa', 'conv3', 'bn3', 'se', 'act3']`. The `se` child is the CBAM module.
**Warning signs:** Grad-CAM heatmaps are all zeros or uniformly bright.

### Pitfall 4: CBAM Weights Not Pretrained
**What goes wrong:** Loading ImageNet-pretrained ResNet-50 but CBAM modules have random weights.
**Why it happens:** `timm.create_model('resnet50', pretrained=True, block_args=dict(attn_layer='cbam'))` loads pretrained ResNet-50 weights but CBAM layers were not in the original model.
**How to avoid:** This is expected behavior -- CBAM layers will be randomly initialized and will train from scratch while the ResNet backbone retains pretrained features. The attention modules are small (48 CBAM modules add ~2.5M parameters to the 23.5M base). Fine-tuning the full model will train both backbone and CBAM jointly.
**Warning signs:** Not a real problem; just be aware initial attention maps will be noisy.

### Pitfall 5: Memory Pressure with Larger Models + Larger Images
**What goes wrong:** OOM errors with EfficientNet-B3 at 380px.
**Why it happens:** Activation memory scales with image resolution squared and model depth.
**How to avoid:** Use batch_size=8 for B3 at 300px, batch_size=4 for B3 at 380px if needed. Consider `torch.amp.autocast` for mixed precision on MPS. Start conservative and increase.
**Warning signs:** Training crashes mid-epoch with memory errors.

### Pitfall 6: Annotation-Guided Dataset Changes Class Weights
**What goes wrong:** `BTXRDAnnotatedDataset` doubles tumor samples, changing the effective class distribution.
**Why it happens:** ROI-guided duplicates add ~1,300 extra tumor samples to training set.
**How to avoid:** Recompute class weights from `train_dataset.labels` (already done in `train.py`). The `compute_class_weights()` function uses all labels including duplicates, which will reduce tumor class weights appropriately. This is correct behavior -- the duplicates provide variety (dimmed vs original) so the model sees both perspectives.
**Warning signs:** Class weights look different from non-annotated training -- this is expected, not a bug.

### Pitfall 7: Overwriting Existing Baseline Checkpoints
**What goes wrong:** Running new training overwrites `best_stratified.pt` and `best_center.pt`, losing the original B0 baseline.
**Why it happens:** Checkpoint naming uses `best_{split_prefix}.pt` without model name.
**How to avoid:** Either (a) back up existing checkpoints before running new experiments, or (b) modify checkpoint naming to include model identifier, or (c) use separate results directories per experiment.
**Warning signs:** Old metrics and new metrics can't be compared because old checkpoints are gone.

## Code Examples

### Creating EfficientNet-B3 with timm (Verified)
```python
# Source: Verified locally with timm 1.0.15
import timm
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=3, drop_rate=0.3)
# params=10.7M, features=1536, default input=(3, 288, 288)
# Pretrained weights: 'efficientnet_b3.ra2_in1k'
```

### Creating ResNet-50-CBAM with timm (Verified)
```python
# Source: Verified locally with timm 1.0.15
import timm
model = timm.create_model(
    'resnet50',
    pretrained=True,
    num_classes=3,
    drop_rate=0.3,
    block_args=dict(attn_layer='cbam'),
)
# params=26.0M, features=2048, 48 CBAM modules injected
# Grad-CAM target: model.layer4[-1] (Bottleneck with CBAM in .se position)
```

### Updating factory.py for CBAM Support
```python
# In src/models/factory.py, update create_model():
def create_model(config: dict) -> BTXRDClassifier:
    model_cfg = config["model"]
    backbone = model_cfg["backbone"]

    # Build kwargs for timm.create_model
    kwargs = {}
    if model_cfg.get("attn_layer"):
        kwargs["block_args"] = dict(attn_layer=model_cfg["attn_layer"])

    return BTXRDClassifier(
        backbone=backbone,
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        drop_rate=model_cfg.get("dropout", 0.2),
        **kwargs,
    )
```

### Updated Grad-CAM Target Layer for ResNet (Verified)
```python
@property
def gradcam_target_layer(self) -> nn.Module:
    """Return the layer to use for Grad-CAM visualization.

    Architecture-specific target layers:
    - EfficientNet: model.bn2 (final BatchNormAct2d before GAP)
    - ResNet: model.layer4[-1] (last bottleneck block)
    """
    if hasattr(self.model, "bn2"):
        return self.model.bn2  # EfficientNet family
    if hasattr(self.model, "layer4"):
        return self.model.layer4[-1]  # ResNet family
    # Fallback: walk backwards
    for module in reversed(list(self.model.modules())):
        if isinstance(module, (nn.BatchNorm2d, nn.Conv2d)):
            return module
    raise AttributeError("Cannot find Grad-CAM target layer")
```

### Running an Experiment (Using Existing CLI)
```bash
# Experiment 1: Clean B0 baseline (reproduce original results)
python scripts/train.py --config configs/default.yaml \
  --override model.backbone=efficientnet_b0 \
  --override data.image_size=224 \
  --override training.batch_size=32 \
  --override model.dropout=0.2 \
  --override training.split_strategy=stratified

# Experiment 2: B3 with annotations
python scripts/train.py --config configs/default.yaml \
  --override model.backbone=efficientnet_b3 \
  --override data.image_size=300 \
  --override training.batch_size=16 \
  --override model.dropout=0.3 \
  --override training.split_strategy=stratified

# Experiment 3: ResNet-50-CBAM with annotations
python scripts/train.py --config configs/default.yaml \
  --override model.backbone=resnet50 \
  --override model.attn_layer=cbam \
  --override data.image_size=224 \
  --override training.batch_size=16 \
  --override model.dropout=0.3 \
  --override training.split_strategy=stratified
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| EfficientNet-B0 @ 224px | EfficientNet-B3 @ 288-300px | Phase 9 | ~2.7x more params (10.7M vs 4.0M), ~1.7x larger input |
| Standard dataset (no annotations) | BTXRDAnnotatedDataset (ROI dimming) | Already implemented | Doubles tumor samples, guides attention to annotated regions |
| No attention mechanism | ResNet-50-CBAM (channel + spatial attention) | Phase 9 | Explicit attention guidance; 2025 paper achieved 97.4% accuracy on BTXRD with 9-class task |
| Manual hyperparameter selection | Systematic experiment grid | Phase 9 | Fair comparison across architectures |

**Key context -- BTXRD paper methodology (Yao et al. 2025):**
- Used YOLOv8s-cls with random 80/20 split (NO patient grouping -- potential leakage)
- Image size: 600px
- Training: 300 epochs with "standard hyperparameters and official pre-trained weights"
- No annotations used for classification (only for detection/segmentation tasks)
- Results likely inflated due to random split without patient grouping

**Key context -- ResNet-50-CBAM paper (Ulster University, 2025):**
- Used BTXRD dataset with BM3D filtering and SMOTE oversampling
- 9-class subtypes (not 3-class)
- Achieved 97.41% accuracy, F1=0.9759, AUC-ROC=0.984
- Compared against: ResNet50, EfficientNet-B3, YOLOv8s-cls, DS-Net
- Full details behind paywall (Biomedical Signal Processing and Control)

**Deprecated/outdated:**
- "Bone-CNN" is NOT a specific published architecture. Web searches reveal it as a generic term for CNNs applied to bone classification, not a reproducible model. **Do not attempt to implement "Bone-CNN".**

## Open Questions

1. **What is the optimal dim_factor for ROI-guided training?**
   - What we know: Currently 0.5 (50% dimming of non-ROI regions). This is a reasonable starting point.
   - What's unclear: Whether stronger dimming (0.3) or weaker (0.7) would work better.
   - Recommendation: Keep 0.5 for now; could be a future hyperparameter sweep.

2. **Should we use attention guidance loss in addition to ROI dimming?**
   - What we know: Research (Li et al. CVPR 2018 "Tell Me Where to Look", IEEE JBHI 2022 attention guidance) shows combining classification cross-entropy with an auxiliary attention guidance loss (binary cross-entropy on attention maps vs annotation masks) can improve performance by 5-10%.
   - What's unclear: Implementation complexity is moderate -- requires extracting intermediate attention maps and computing auxiliary loss.
   - Recommendation: **Defer for now.** ROI dimming is simpler and already implemented. If results are still insufficient after architecture upgrades, attention guidance loss is the next lever to pull.

3. **Should we use 300px or 380px for EfficientNet-B3?**
   - What we know: B3's default input is 288x288. The config currently has 380px.
   - What's unclear: Whether the additional resolution helps or just wastes memory. The BTXRD paper used 600px but with YOLOv8.
   - Recommendation: Use 300px (close to native, round number, good memory tradeoff). 380px can be tested if 300px results are promising.

4. **Memory constraints on MPS (Apple Silicon)?**
   - What we know: System has MPS (no CUDA). MPS memory is shared with system RAM. Mixed precision via `torch.amp.autocast(device_type='mps', dtype=torch.bfloat16)` is available in PyTorch 2.6.
   - What's unclear: Exact batch sizes that will fit for B3 at 300px on this specific hardware.
   - Recommendation: Start with batch_size=16 for B3 at 300px. Fall back to 8 if OOM.

5. **How to handle checkpoint naming for multiple experiments?**
   - What we know: Current naming is `best_{split_prefix}.pt` which will overwrite across experiments.
   - What's unclear: Best naming convention that doesn't break existing eval/gradcam scripts.
   - Recommendation: Add model identifier to checkpoint name: `best_{split_prefix}_{backbone}.pt`. Update eval script to accept checkpoint path override.

6. **How does `block_args` integrate with `BTXRDClassifier`?**
   - What we know: `BTXRDClassifier.__init__` currently passes backbone, pretrained, num_classes, and drop_rate to `timm.create_model`. It doesn't support `block_args`.
   - What's unclear: N/A -- the fix is straightforward.
   - Recommendation: Add `**kwargs` to `BTXRDClassifier.__init__` and pass through to `timm.create_model`.

## Verified Architecture Details

### EfficientNet Family (Verified in timm 1.0.15)
| Model | Params | Features | Default Input | Pretrained |
|-------|--------|----------|---------------|------------|
| efficientnet_b0 | 4.0M | 1280 | 224x224 | `efficientnet_b0.ra_in1k` |
| efficientnet_b2 | 7.7M | 1408 | 256x256 | `efficientnet_b2.ra_in1k` |
| efficientnet_b3 | 10.7M | 1536 | 288x288 | `efficientnet_b3.ra2_in1k` |

### ResNet-50 Variants (Verified in timm 1.0.15)
| Model | Params | Features | Default Input | Notes |
|-------|--------|----------|---------------|-------|
| resnet50 (standard) | 23.5M | 2048 | 224x224 | No attention |
| resnet50 + CBAM | 26.0M | 2048 | 224x224 | 48 CBAM modules, `block_args=dict(attn_layer='cbam')` |
| seresnet50 | 26.0M | 2048 | 224x224 | SE attention (alternative to CBAM) |

### Grad-CAM Target Layers (Verified)
| Architecture | Target Layer | Python Expression |
|-------------|-------------|-------------------|
| EfficientNet-B0/B2/B3 | `model.bn2` (BatchNormAct2d) | `model.model.bn2` |
| ResNet-50 (plain) | `model.layer4[-1]` (Bottleneck) | `model.model.layer4[-1]` |
| ResNet-50-CBAM | `model.layer4[-1]` (Bottleneck with CBAM in .se) | `model.model.layer4[-1]` |

### Annotation Coverage (Verified)
| Split | Total | With Annotations | Normal (no ann) | Benign (all ann) | Malignant (all ann) |
|-------|-------|-----------------|-----------------|------------------|---------------------|
| stratified_train | 2,621 | 1,306 | 1,315 | 1,066 | 240 |
| stratified_val | 561 | 279 | 282 | 228 | 51 |
| stratified_test | 564 | 282 | 282 | 231 | 51 |
| center_train | 2,499 | 1,144 | 1,355 | 944 | 200 |
| center_val | 439 | 201 | 238 | 166 | 35 |
| center_test | 808 | 522 | 286 | 415 | 107 |

### Current Baseline vs Paper (for context)
| Metric | Our Stratified | Our Center-Holdout | Paper (random 80/20) |
|--------|---------------|-------------------|---------------------|
| Accuracy | 67.9% | 47.2% | ~90% (estimated from per-class) |
| Normal precision | 83.4% | 57.7% | 91.3% |
| Benign precision | 59.2% | 53.6% | 88.1% |
| Malignant precision | 58.5% | 21.7% | 73.4% |
| Malignant recall | 60.8% | 36.4% | 83.9% |
| Architecture | EfficientNet-B0 | EfficientNet-B0 | YOLOv8s-cls |
| Image size | 224px | 224px | 600px |
| Epochs trained | 5 (early stop) | ~5 (early stop) | 300 |
| Annotations in training | No | No | No |

## Experiment Design Recommendation

### Experiment Grid
| ID | Backbone | Image Size | Batch Size | Annotations | Splits |
|----|----------|-----------|------------|-------------|--------|
| E1 | efficientnet_b0 | 224 | 32 | No | Both |
| E2 | efficientnet_b0 | 224 | 32 | Yes (ROI dim) | Both |
| E3 | efficientnet_b3 | 300 | 16 | Yes (ROI dim) | Both |
| E4 | resnet50+cbam | 224 | 16 | Yes (ROI dim) | Both |

### Fixed Across All Experiments
- Seed: 42
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 1e-4
- Warmup: 3 epochs
- Scheduler: cosine annealing
- Label smoothing: 0.1
- Early stopping patience: 15
- Dropout: 0.3 (except E1: 0.2 for clean baseline reproduction)

### Evaluation Protocol
For each experiment:
1. Train on stratified split
2. Train on center split
3. Run `scripts/eval.py` for both
4. Run `scripts/gradcam.py` for stratified checkpoint
5. Record: accuracy, macro AUC, per-class sensitivity, Malignant sensitivity, Grad-CAM mean IoU

### Success Criteria
- Primary: Stratified accuracy > 75% (improvement over 67.9%)
- Secondary: Center-holdout accuracy > 55% (improvement over 47.2%)
- Stretch: Malignant sensitivity > 70% (improvement over 60.8%)
- Explainability: Grad-CAM mean IoU > 0.15 (improvement over 0.070)

## Sources

### Primary (HIGH confidence)
- timm 1.0.15 (installed locally) -- verified CBAM support via `create_attn('cbam')`, verified model parameters and default input sizes, verified `block_args=dict(attn_layer='cbam')` for ResNet-50
- Codebase analysis -- verified `BTXRDAnnotatedDataset`, `BTXRDClassifier`, `factory.py`, `train.py`, existing eval pipeline
- Checkpoint inspection -- verified current model is EfficientNet-B0 at 224px (not B3 as config suggests)

### Secondary (MEDIUM confidence)
- [BTXRD Paper (PMC11739492)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11739492/) -- training methodology: YOLOv8s-cls, 300 epochs, 600px, random 80/20 split, "standard hyperparameters"
- [ResNet-50-CBAM on BTXRD (Ulster University 2025)](https://pure.ulster.ac.uk/en/publications/deep-learning-driven-radiographic-classification-of-primary-bone-/) -- 97.41% accuracy on 9-class BTXRD with BM3D + SMOTE, compared against EfficientNet-B3 and YOLOv8s-cls
- [timm documentation (timm.fast.ai)](https://timm.fast.ai/) -- create_model API, model registry
- [Keras EfficientNet docs](https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/) -- B2/B3 parameter counts and image sizes (cross-verified with timm)

### Tertiary (LOW confidence)
- [CNN Attention Guidance for Fracture Classification](https://deepai.org/publication/cnn-attention-guidance-for-improved-orthopedics-radiographic-fracture-classification) -- attention guidance loss concept (5-10% improvement claim from abstract only; full paper behind IEEE paywall)
- [MGANet (Mask Guided Attention)](https://github.com/Markin-Wang/MGANet) -- mask-guided attention for fine-grained classification (architecture reference, not directly verified for our use case)
- Various GitHub issues on EfficientNet memory usage -- suggest B3 at 300px with BS=16 should fit in 8GB VRAM with mixed precision

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified locally, timm CBAM support confirmed
- Architecture patterns: HIGH -- code examples verified with actual model creation and forward passes
- Annotation-guided training: MEDIUM -- `BTXRDAnnotatedDataset` verified in codebase, but optimal dim_factor and attention guidance loss approach not empirically validated
- Experiment design: MEDIUM -- based on standard ML experiment methodology but specific hyperparameters may need tuning
- Pitfalls: HIGH -- based on codebase analysis and verified architecture differences

**Research date:** 2026-02-22
**Valid until:** 2026-03-22 (stable domain; timm and PyTorch unlikely to have breaking changes)
