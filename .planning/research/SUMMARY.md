# Project Research Summary

**Project:** AssureXRay
**Domain:** Medical image classification PoC — 3-class bone tumor radiograph classification (Normal / Benign / Malignant)
**Researched:** 2026-02-19
**Confidence:** HIGH (mature domain; stable, well-documented patterns across all four research areas)

---

## Executive Summary

AssureXRay is a proof-of-concept classifier for the BTXRD dataset: 3,746 radiograph images from 3 centers, labeled Normal (1,879), Benign (1,525), and Malignant (342). The task is well-understood — transfer learning from ImageNet-pretrained EfficientNet-B0 via the `timm` library is the correct approach, albumentations handles the radiograph-specific augmentation pipeline, and `pytorch-grad-cam` provides the explainability layer that medical reviewers require. The entire pipeline should be built as a series of standalone CLI scripts connected by filesystem artifacts, not a monolithic script or an orchestration framework. This is the established pattern for auditable ML PoCs in clinical research settings.

The recommended approach is a 7-phase sequential build: data acquisition and audit, data loading and splitting, model training, evaluation on both split strategies, explainability and inference, documentation, and an optional Streamlit demo. Phases 1 through 4 are strictly sequential — each phase's outputs are the inputs for the next. The critical technical decision at each phase is restraint: weighted CrossEntropyLoss over focal loss (for now), plain PyTorch over Lightning, scikit-learn over torchmetrics, and YAML + argparse over Hydra. The stack is deliberately minimal because a PoC that cannot be reproduced or audited is worthless in a clinical context.

The dominant risk is not technical but methodological: the BTXRD dataset has no patient_id, meaning naive image-level splitting will inflate test metrics by placing different views of the same lesion in both train and test. Every other result depends on this being handled correctly. Secondary risks are evaluating only on accuracy (the model can achieve ~91% accuracy by always predicting "not malignant"), and shortcut learning from center-specific image characteristics (Center 1 is 78% of the dataset and from clinical PACS; Centers 2-3 are web-sourced with potentially different formatting). Both are detectable and preventable if the evaluation framework is designed before training begins.

---

## Key Findings

### Recommended Stack

The stack is narrow and well-justified. `timm` is the backbone source (not `torchvision.models`) because it provides multiple EfficientNet-B0 weight variants and a uniform API. `albumentations` is used for the stochastic training augmentation pipeline because torchvision alone lacks CLAHE (contrast-limited adaptive histogram equalization), which is important for radiograph contrast normalization. `torchvision.transforms.v2` handles the deterministic steps (resize, normalize, tensorize) for both train and inference. `pytorch-grad-cam` handles Grad-CAM without requiring manual gradient hook management. `scikit-learn` and `scipy` handle all offline evaluation and bootstrap confidence intervals.

Plain PyTorch training loops are explicitly preferred over PyTorch Lightning for this PoC. Lightning's callback system adds debugging indirection and complicates Grad-CAM integration (requires unwrapping `LightningModule.model`). For a single-GPU, 3.7K image PoC, it is the wrong abstraction. MONAI is similarly rejected — it is the right tool for 3D volumetric segmentation, not 2D radiograph classification.

**Core technologies:**

| Technology | Version | Purpose | Confidence |
|------------|---------|---------|------------|
| Python | >=3.10, <3.13 (3.12 ideal) | Runtime | HIGH |
| PyTorch | >=2.2, <=2.5 | Deep learning framework | MEDIUM |
| torchvision | >=0.17 (matched to PyTorch) | Transforms, data utilities | MEDIUM |
| timm | >=1.0.0 | EfficientNet-B0 backbone with weights | HIGH |
| albumentations | >=1.4.0 | Medical-relevant training augmentations | HIGH |
| pytorch-grad-cam | >=1.5.0 | Grad-CAM heatmap generation | HIGH |
| scikit-learn | >=1.4 | Offline metrics (AUC, confusion matrix, etc.) | HIGH |
| scipy | >=1.12 | Bootstrap confidence intervals | HIGH |
| pandas | >=2.1 | CSV and metadata handling | HIGH |
| matplotlib + seaborn | >=3.8 / >=0.13 | Plots and visualization | HIGH |
| PyYAML + argparse | >=6.0 / stdlib | Config and CLI | HIGH |
| streamlit | >=1.35 | Optional interactive demo | MEDIUM |

**Verify all versions against PyPI before installing** — research was cut off at May 2025, and PyTorch version pairing (torch + torchvision) is particularly sensitive.

### Expected Features

**Must have (table stakes) — the PoC is not credible without these:**

- Automated BTXRD download and organization script
- Data audit report: class distribution, per-center image statistics, annotation coverage, duplicate detection
- Dataset specification document
- Dual split strategy: image-level stratified (70/15/15) AND center holdout (Center 1 train/val, Centers 2+3 test)
- Proxy patient grouping before any splitting (no patient_id in dataset — must cluster by metadata)
- Transfer learning from ImageNet-pretrained EfficientNet-B0 (via timm)
- Inverse-frequency weighted CrossEntropyLoss for class imbalance
- Standard augmentation pipeline (albumentations CLAHE, ShiftScaleRotate, ElasticTransform + torchvision Resize/Normalize)
- Configurable hyperparameters via YAML config + argparse CLI overrides
- Early stopping with patience (monitor val loss, patience 5-10 epochs)
- Best-checkpoint saving (full dict: weights, optimizer state, config, class names, normalization stats)
- Training loss/metric curves saved to CSV and plotted
- ROC AUC (one-vs-rest, per-class and macro-averaged)
- PR AUC per class (especially critical for Malignant)
- Per-class sensitivity and specificity (the primary clinical metrics)
- Confusion matrix (raw counts + row-normalized)
- Classification report (precision, recall, F1 per class)
- Evaluation on BOTH split strategies with side-by-side comparison
- Operating point analysis (threshold sweep for Malignant class sensitivity/specificity trade-off)
- Grad-CAM heatmaps with qualitative comparison against LabelMe tumor annotations
- Curated TP/FP/FN Grad-CAM examples (systematic, not cherry-picked)
- Single-image inference script (CLI + softmax probabilities output)
- Model card (Mitchell et al. 2019 format)
- PoC report with explicit limitations section
- README with end-to-end setup and run instructions
- requirements.txt with pinned versions

**Should have (high-value differentiators):**

- Bootstrap 95% CIs on AUC and Malignant sensitivity (30 lines of code, transforms credibility)
- Comparison table against paper's YOLOv8s-cls baseline (with caveats about leakage in their split)
- Failure case analysis: what do FP/FN cases have in common?
- Clinical decision framing in report ("at 90% specificity, model achieves X% sensitivity for malignant")
- Streamlit demo app for non-technical stakeholders
- Makefile for one-command pipeline execution

**Defer to post-PoC:**

- Subgroup fairness analysis (requires careful statistics with small subgroups)
- Calibration analysis / reliability diagrams
- Quantitative Grad-CAM IoU against LabelMe annotations (valuable research direction, not needed for feasibility)
- Experiment tracking with W&B or MLflow
- Docker or containerized environment
- 9-subtype classification (statistically meaningless at current Malignant sample sizes)
- Multi-task learning (classification + detection + segmentation)
- DICOM/PACS integration
- Hyperparameter optimization (Optuna, Ray Tune)

### Architecture Approach

AssureXRay is a batch-oriented, script-per-stage ML pipeline. Each pipeline stage is a standalone CLI script that reads from disk and writes to disk. Stages communicate through filesystem conventions (agreed paths in config), not in-memory coupling. This is the correct architecture for a PoC because each stage can be run, debugged, and rerun independently, intermediate artifacts persist between runs, and any intermediate output is inspectable.

The critical structural pattern is the library vs. script distinction: `src/` contains reusable library modules (Dataset class, model definition, metrics, visualization, Grad-CAM logic) that are imported by thin entry-point scripts. Scripts never import other scripts. This separation means the Streamlit demo and the CLI inference script share identical model loading and prediction logic from `src/`, eliminating a major class of preprocessing inconsistency bugs.

**Major components:**

| Component | Responsibility |
|-----------|---------------|
| `scripts/download.py` | Fetch BTXRD from figshare, organize into `data_raw/` |
| `scripts/audit.py` | Profile class distribution, image stats, annotation coverage, per-center quality |
| `scripts/split.py` | Generate both split strategy CSVs (committed to version control for reproducibility) |
| `src/data/dataset.py` | PyTorch Dataset class — maps split CSVs to image tensors via albumentations + torchvision pipeline |
| `src/data/transforms.py` | Single source of truth for all transform pipelines (train/val/test/inference modes) |
| `src/models/classifier.py` | BoneTumorClassifier: timm backbone + dropout + linear head, with `get_gradcam_target_layer()` built in |
| `scripts/train.py` | Plain PyTorch training loop: forward, weighted loss, backward, early stopping, checkpoint saving |
| `scripts/eval.py` | Metrics pipeline: per-class sensitivity/specificity, ROC/PR curves, confusion matrix, both splits |
| `src/evaluation/bootstrap.py` | Bootstrap 95% CIs via scipy |
| `scripts/gradcam.py` | Batch Grad-CAM generation for selected TP/FP/FN examples |
| `scripts/infer.py` | Single-image prediction + Grad-CAM overlay (JSON output + PNG) |
| `app/app.py` | Optional Streamlit demo (wraps infer logic from `src/`) |
| `configs/default.yaml` | All hyperparameters, paths, seeds — read by every script |
| `Makefile` | Convenience targets: `make download`, `make audit`, `make train`, `make eval`, `make all` |

**Key patterns to follow:**

1. Config-driven pipeline: no magic numbers in code; YAML for all hyperparameters and paths
2. Deterministic reproducibility: single `set_seed()` function covering `random`, `numpy`, `torch`, `torch.cuda`, with `cudnn.deterministic=True`
3. Checkpoint-as-contract: checkpoints save the full config, class names, and normalization stats alongside weights
4. Metrics-as-JSON: structured JSON for all evaluation outputs, enabling programmatic comparison across experiments
5. Separation of concerns: model class handles only architecture; loss, optimizer, and training loop live in the training script

### Critical Pitfalls

1. **Same-lesion data leakage across splits (CRITICAL)** — BTXRD has no patient_id. Different views of the same lesion in both train and test inflates metrics by 5-15 percentage points. Prevention: cluster images by metadata (anatomical_site + gender + age + center + diagnosis) to create proxy patient groups, then use GroupKFold-style splitting. Also run evaluation on both the grouped split (honest) and naive image split (for paper comparability). This must be solved in Phase 1-2 before any training.

2. **Misleading metrics from class imbalance (CRITICAL)** — With 9.1% malignant prevalence, a model that always predicts "not malignant" achieves 90.9% accuracy. Primary metric must be per-class sensitivity (recall), especially Malignant sensitivity. PR AUC is more informative than ROC AUC for the Malignant class. Design the evaluation metric suite before training begins.

3. **Center-correlated shortcut learning (CRITICAL)** — Three centers with different imaging equipment, JPEG quality, and web vs. clinical sourcing. The model can detect center identity and exploit it as a proxy for class labels. Prevention: mandatory center holdout evaluation (train Center 1, test Centers 2+3). Grad-CAM audit should check that attention is on anatomical regions, not image borders or artifacts.

4. **Grad-CAM misinterpretation as clinical validation (HIGH)** — Generating only cherry-picked "good" Grad-CAM examples creates false confidence. Grad-CAM shows which regions influenced the prediction, not whether the model understands tumor morphology. Prevention: show FP and FN Grad-CAMs prominently; explicitly disclaim in the model card that Grad-CAM does not constitute clinical validation.

5. **Label noise on the Malignant class (MEDIUM-HIGH)** — Some BTXRD diagnoses are radiologist opinion without pathology confirmation. With only 342 malignant images, even 5-10% label error means 17-34 mislabeled images. If Malignant sensitivity plateaus at 70-75%, this is likely the ceiling imposed by label quality, not a tuning problem. Document this explicitly rather than chasing marginal gains.

6. **Preprocessing inconsistency between train and inference (HIGH)** — Different image loading libraries (PIL vs OpenCV), different interpolation methods, or augmentations accidentally applied during inference will cause the inference script and evaluation to produce different results. Prevention: define a single `get_transforms(mode=...)` function in `src/data/transforms.py` and import it everywhere — never duplicate transform definitions.

---

## Implications for Roadmap

Based on all four research files, the phase structure is well-determined by data dependencies. Phases 1-4 are strictly sequential. Phase 5 depends on Phase 4 output (TP/FP/FN identification). Phase 6 depends on Phases 4 and 5. Phase 7 (demo) depends only on Phase 5.

### Phase 1: Data Foundation

**Rationale:** Nothing else can be built without the data, and the audit may reveal blocking issues that change downstream decisions. The proxy patient grouping strategy must be designed here — it affects everything downstream. This is the phase most likely to surface surprises (watermarks on web-sourced images, unexpected class distributions, annotation gaps).

**Delivers:** Raw data organized in `data_raw/`, `docs/data_audit_report.md`, `docs/dataset_spec.md`, and documented proxy patient grouping strategy.

**Addresses features:** Automated download, data audit report, dataset specification, leakage risk documentation.

**Avoids:** Data leakage via same-lesion multi-angle images (Pitfall 1), center-correlated shortcuts from undetected image quality issues (Pitfall 3), JPEG compression artifact confounders (Pitfall 13).

**Validation gate:** Class distribution matches expected (1879/1525/342). Per-center image quality statistics documented. Proxy patient group sizes are reasonable. No corrupt files.

### Phase 2: Data Loading and Splitting

**Rationale:** Split correctness is the foundation of all evaluation claims. Bugs here (label leakage, wrong stratification, transform inconsistency) invalidate all downstream results. Unit tests are critical at this phase. The transform pipeline must be written once here and imported by all future scripts.

**Delivers:** `data/splits/` CSVs for both split strategies (committed to git), `src/data/dataset.py`, `src/data/transforms.py`, `src/utils/reproducibility.py`, `src/utils/config.py`.

**Addresses features:** Dual split strategy (stratified + center holdout), reproducible seeds, configurable pipeline.

**Avoids:** Transform inconsistency between train and inference (Pitfall 6), wrong set evaluated at eval time (Pitfall 12), seed non-determinism (Pitfall 11).

**Validation gate:** No image ID overlap between splits. Class proportions preserved across stratified splits. Malignant count in each split is non-trivial. Dataset class produces correctly shaped, correctly normalized tensors.

### Phase 3: Model and Training

**Rationale:** Model definition includes `get_gradcam_target_layer()` from the start — retrofitting Grad-CAM hooks after the fact is painful and constitutes the "Grad-CAM as afterthought" anti-pattern. Train on the stratified split first to validate the end-to-end pipeline before committing compute to the center holdout experiment. Use the simplest possible configuration first.

**Delivers:** `src/models/classifier.py`, `scripts/train.py`, `checkpoints/best_stratified.pt`, `checkpoints/best_center.pt`, training logs and loss curves.

**Addresses features:** Transfer learning from EfficientNet-B0 (timm), class imbalance handling (weighted CE), early stopping, checkpoint saving, configurable hyperparameters.

**Avoids:** EfficientNet frozen layer strategy mistakes (Pitfall 7), class weight miscalculation (Pitfall 10), overengineering before a baseline exists (Pitfall 9).

**Validation gate:** Training loss decreases monotonically. Val metric improves then plateaus (early stopping triggers). Checkpoint loads correctly and runs forward pass without error. Malignant class does not collapse (check per-class training accuracy).

### Phase 4: Evaluation

**Rationale:** Evaluation must cover both split strategies. The gap between stratified and center-holdout performance is one of the most informative findings of the entire PoC. Bootstrap CIs are included here because the incremental effort is low (30 lines of scipy) and the credibility gain is high for a clinical audience with N=342 malignant.

**Delivers:** `results/stratified/` and `results/center_holdout/` — metrics.json, confusion_matrix.png, roc_curves.png, pr_curves.png, classification_report.txt. Optionally: bootstrap_ci.json.

**Addresses features:** Full evaluation suite (ROC AUC, PR AUC, per-class sensitivity/specificity, confusion matrix, both splits, operating point analysis, bootstrap CIs).

**Avoids:** Misleading metrics from class imbalance (Pitfall 2), evaluating on wrong split (Pitfall 12), ignoring center-performance gap (Pitfall 3).

**Validation gate:** Metrics are plausible (neither 99% accuracy suggesting leakage nor 33% suggesting random). Center holdout performance is reported. Malignant sensitivity is the headline metric, not accuracy.

### Phase 5: Explainability and Inference

**Rationale:** Grad-CAM example selection depends on TP/FP/FN labels from Phase 4 evaluation — this is why explainability follows evaluation, not precedes it. The inference script wraps the same model loading and prediction logic already proven in eval, so it is mostly integration work. The Streamlit demo, if built, reuses the inference logic.

**Delivers:** `results/gradcam/` (TP/FP/FN overlay PNGs per class, gradcam_summary.md), `scripts/infer.py` (JSON prediction output + overlay image).

**Addresses features:** Grad-CAM heatmaps, systematic TP/FP/FN examples, annotation comparison (qualitative), single-image inference script with confidence scores.

**Avoids:** Grad-CAM misinterpretation (Pitfall 5), cherry-picked examples only.

**Validation gate:** Heatmaps visually highlight anatomical regions for correctly classified tumor images. FP/FN heatmaps suggest plausible failure modes (not background/border attention). Inference script produces identical predictions to eval script on same images.

### Phase 6: Documentation and Reporting

**Rationale:** Documentation is the deliverable. It synthesizes findings from all five previous phases. The model card and PoC report must include an explicit limitations section — this is non-negotiable for a medical AI PoC. Documentation written before results are final is speculation; written after, it is a synthesis.

**Delivers:** `docs/model_card.md`, `docs/poc_report.md`, updated `README.md`.

**Addresses features:** Model card, PoC report with limitations, comparison table against paper's baseline, clinical decision framing, requirements file.

**Avoids:** Overclaiming clinical validity (Pitfall 5), missing limitations section.

**Validation gate:** Model card covers all Mitchell et al. sections. Report explicitly addresses: leakage risk, label noise ceiling, center generalization gap, and the absence of pathology-confirmed labels for all cases. README enables clean-room reproduction.

### Phase 7: Demo (Optional)

**Rationale:** Build only if PoC results are worth demonstrating and time permits. Reuses inference logic from Phase 5. Makes the PoC tangible for non-technical stakeholders (clinicians, management). Must include a "not for clinical use" disclaimer prominently.

**Delivers:** `app/app.py` — Streamlit app with image upload, prediction with confidence bars, Grad-CAM overlay.

**Addresses features:** Streamlit interactive demo.

**Validation gate:** Works on localhost. Disclaimer is visible. Predictions match inference script output.

---

### Phase Ordering Rationale

- **Phases 1-4 are strictly sequential** due to data dependencies: download before audit, audit before split, split before training, training before evaluation.
- **Phases 5 and 6 partially overlap**: Grad-CAM example selection requires eval output (Phase 4), but documentation drafting can begin during Phase 4 while evaluation runs.
- **Phase 7 depends only on Phase 5** (reuses inference logic) and is independent of Phase 6.
- **Proxy patient grouping (Phase 1-2 boundary)** is the single highest-stakes decision: it must be implemented correctly before any training data is touched, because a leaked split cannot be retroactively fixed.
- **Evaluation metric suite (Phase 4) must be designed before training (Phase 3)**: per-class sensitivity as primary metric, PR AUC for Malignant, both split strategies — specifying these before training eliminates the temptation to choose metrics post-hoc based on what looks best.

### Research Flags

Phases with well-documented patterns — standard execution, no additional research needed:

- **Phase 1 (Data Foundation):** Figshare download, pandas data audit, and matplotlib distribution plots are standard. The proxy grouping logic is novel to this dataset but well-specified in PITFALLS.md.
- **Phase 3 (Model and Training):** timm + EfficientNet-B0 + plain PyTorch training loop is documented extensively. Weighted CE formula is spelled out in STACK.md.
- **Phase 4 (Evaluation):** scikit-learn metric APIs are stable. Bootstrap CI via scipy is standard.
- **Phase 6 (Documentation):** Model card format (Mitchell et al.) is fixed. PoC report structure is conventional.

Phases that may need deeper research or validation during implementation:

- **Phase 2 (Data Splitting):** The proxy patient grouping implementation using BTXRD metadata columns needs validation against the actual `dataset.csv` column names and data. Confirm which metadata fields are available and whether they are sufficient for credible proxy groups before writing the split logic.
- **Phase 5 (Grad-CAM):** The correct `target_layer` for timm's EfficientNet-B0 is `model.conv_head` or `model.blocks[-1]` — verify this against the actual timm model structure at the version installed, since layer names can shift across timm releases. Also verify pytorch-grad-cam's API compatibility with the installed timm version.
- **Phase 7 (Demo):** Streamlit version compatibility with the Python and PyTorch environment should be validated — Streamlit sometimes has conflicts with PyTorch's multiprocessing.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | MEDIUM | Core libraries (timm, albumentations, pytorch-grad-cam, scikit-learn) are HIGH confidence. Specific version numbers are MEDIUM — training cutoff May 2025; verify PyTorch/torchvision pairing against current PyPI before installing. |
| Features | HIGH | Medical imaging classification PoC requirements are mature and stable. Table stakes features are cross-validated against SPIRIT-AI, CONSORT-AI, FDA guidance, and MICCAI community standards. Differentiator rankings are MEDIUM (audience-dependent). |
| Architecture | HIGH | Script-per-stage pipeline, config-driven, library/script separation, and checkpoint-as-contract are well-established patterns in academic and industry ML. No novel architecture decisions required. |
| Pitfalls | MEDIUM-HIGH | Data leakage, class imbalance metrics, and shortcut learning pitfalls are HIGH confidence (peer-reviewed literature). BTXRD-specific pitfalls (no patient_id, web-sourced centers) are MEDIUM-HIGH based on project context and paper metadata. EfficientNet fine-tuning strategy is MEDIUM (community best practice, not formally benchmarked for this dataset). |

**Overall confidence:** HIGH for the approach and architecture; MEDIUM for specific version numbers and BTXRD-specific metadata column availability.

### Gaps to Address

- **BTXRD dataset.csv column names:** The proxy patient grouping strategy assumes that `anatomical_site`, `gender`, `age`, `center`, and `diagnosis` columns exist in `dataset.csv`. Verify these column names and data completeness immediately after download (Phase 1). If key columns are missing or sparse, the grouping strategy needs adjustment.
- **PyTorch + torchvision version pairing:** Run `pip index versions torch` before creating the environment. The specific paired versions (e.g., torch 2.4.x + torchvision 0.19.x) must be matched — mismatched versions cause silent import errors or CUDA failures.
- **timm EfficientNet-B0 layer names:** Verify `model.conv_head` or `model.blocks[-1]` is the correct Grad-CAM target for the installed timm version. Layer names have changed across timm major versions.
- **BTXRD GitHub repo status:** Before implementation, manually check `https://github.com/SHUNHANYAO/BTXRD` for any dataset updates, errata, or improved splitting methodology published since the original paper.
- **Web-sourced image quality:** Centers 2 (Radiopaedia) and 3 (MedPix) may have text overlays, watermarks, or annotation marks drawn on the images. The severity of this issue cannot be assessed without downloading and inspecting the data. Factor in a preprocessing decision point during Phase 1 audit.

---

## Sources

### Primary (HIGH confidence)
- Yao et al. "A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors." Scientific Data, 2025. DOI: 10.1038/s41597-024-04311-y — BTXRD dataset, class distributions, baseline YOLOv8s-cls results
- timm (pytorch-image-models) library documentation and GitHub — EfficientNet-B0 backbone, weight variants, `create_model` API
- pytorch-grad-cam (jacobgil) GitHub — Grad-CAM implementation, target layer selection, EfficientNet compatibility
- albumentations official documentation — CLAHE, ElasticTransform, ShiftScaleRotate for medical imaging
- PyTorch official documentation — `torch.amp` (native AMP replacing `torch.cuda.amp` in 2.x)
- torchvision transforms V2 documentation — stable since torchvision 0.17
- Mitchell et al. "Model Cards for Model Reporting." FAT* 2019 — model card format
- Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017 — Grad-CAM method
- SPIRIT-AI and CONSORT-AI guidelines (Lancet Digital Health, 2020) — reporting standards for clinical AI

### Secondary (MEDIUM confidence)
- Roberts et al. "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans." Nature Machine Intelligence, 2021 — data leakage and generalization pitfalls
- Varoquaux, G. and Cheplygina, V. "Machine learning for medical imaging: methodological failures and recommendations." npj Digital Medicine, 2022 — common failure modes
- Geirhos et al. "Shortcut Learning in Deep Neural Networks." Nature Machine Intelligence, 2020 — center-correlated shortcut learning
- DeGrave et al. "AI for radiographic COVID-19 detection selects shortcuts over signal." Nature Machine Intelligence, 2021 — medical imaging shortcut learning
- Rajpurkar et al. "AI in Health and Medicine." Nature Medicine, 2022 — medical AI evaluation standards
- FDA "Artificial Intelligence and Machine Learning in Software as a Medical Device" guidance

### Tertiary (LOW-MEDIUM confidence)
- EfficientNet-B0 phased unfreezing strategy for medical imaging — community best practice, not formally benchmarked for BTXRD
- JPEG compression artifact detection as confounder — documented in forensic imaging; relevance to this specific dataset unverified without data inspection

---
*Research completed: 2026-02-19*
*Ready for roadmap: yes*
