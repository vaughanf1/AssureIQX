# Requirements: AssureXRay

**Defined:** 2026-02-19
**Core Value:** Deliver a reproducible, auditable 3-class bone tumor classification baseline with clear explainability outputs that a clinician can inspect and trust.

## v1 Requirements

Requirements for initial PoC release. Each maps to roadmap phases.

### Data Pipeline

- [ ] **DATA-01**: Download script fetches BTXRD dataset from figshare and organizes into `data_raw/` (images/, annotations/, dataset.csv)
- [ ] **DATA-02**: Data audit report covering class distribution, image dimension histogram, missing values, annotation coverage, duplicate detection, per-center breakdown
- [ ] **DATA-03**: Dataset specification document describing all 37 columns, label derivation logic (malignant=1 → Malignant, benign=1 → Benign, tumor=0 → Normal), and data provenance
- [ ] **DATA-04**: Primary split: image-level stratified train/val/test (70/15/15) with class-label stratification and fixed random seed
- [ ] **DATA-05**: Secondary split: center holdout (Center 1 → train/val, Centers 2+3 → test) with stratified train/val within Center 1
- [ ] **DATA-06**: Split manifests saved as CSV files with image_id, split assignment, and label
- [ ] **DATA-07**: Leakage risk explicitly documented in audit report and PoC report (no patient_id, same-lesion multi-angle images)

### Model Training

- [ ] **TRAIN-01**: PyTorch dataset class loading images + labels from split manifests with configurable transforms
- [ ] **TRAIN-02**: Augmentation pipeline: resize 224x224, ImageNet normalization, CLAHE (albumentations), horizontal flip, small rotation (±15°)
- [ ] **TRAIN-03**: EfficientNet-B0 backbone via timm with pretrained ImageNet weights, 3-class output head
- [ ] **TRAIN-04**: Inverse-frequency weighted cross-entropy loss for class imbalance
- [ ] **TRAIN-05**: Training script with configurable hyperparameters via YAML config (lr, batch_size, epochs, patience, backbone, loss_type)
- [ ] **TRAIN-06**: Early stopping on validation loss with configurable patience
- [ ] **TRAIN-07**: Checkpoint saving: best model (by val_loss) and final model
- [ ] **TRAIN-08**: Training log CSV (epoch, train_loss, val_loss, val_metrics) and loss curve plot

### Evaluation

- [ ] **EVAL-01**: ROC AUC (one-vs-rest per-class + macro) with plotted ROC curves
- [ ] **EVAL-02**: PR AUC (per-class) with plotted precision-recall curves
- [ ] **EVAL-03**: Per-class sensitivity (recall) and specificity from confusion matrix
- [ ] **EVAL-04**: Confusion matrix heatmap (absolute counts + row-normalized)
- [ ] **EVAL-05**: Classification report (precision, recall, F1 per class)
- [ ] **EVAL-06**: Evaluation on both split strategies with comparison table
- [ ] **EVAL-07**: Bootstrap 95% confidence intervals on AUC and per-class sensitivity (1000 iterations)
- [ ] **EVAL-08**: Comparison table against BTXRD paper's YOLOv8s-cls baseline results

### Explainability

- [ ] **EXPL-01**: Grad-CAM heatmap generation using pytorch-grad-cam on EfficientNet-B0 last conv layer
- [ ] **EXPL-02**: Curated heatmap gallery: 3-5 examples each for TP, FP, FN per class, saved as image grid
- [ ] **EXPL-03**: Heatmap overlay comparison against LabelMe tumor annotations (qualitative: does model attend to tumor region?)

### Inference

- [ ] **INFR-01**: Single-image inference script: image path + checkpoint → class prediction + softmax confidences + Grad-CAM overlay
- [ ] **INFR-02**: Batch inference mode for evaluating a directory of images

### Documentation

- [ ] **DOCS-01**: `dataset_spec.md` — column definitions, label schema, data provenance, license (CC BY-NC-ND 4.0)
- [ ] **DOCS-02**: `data_audit_report.md` — auto-generated from audit script with embedded figures
- [ ] **DOCS-03**: `model_card.md` — architecture, training data, performance, limitations, ethical considerations
- [ ] **DOCS-04**: `poc_report.md` — executive summary, methods, results (both splits), clinical relevance, limitations, next steps
- [ ] **DOCS-05**: `README.md` — setup instructions, data download, train/eval/infer commands, project structure
- [ ] **DOCS-06**: `requirements.txt` — pinned Python dependencies

### Infrastructure

- [ ] **INFRA-01**: Project scaffold with standard ML PoC directory structure
- [ ] **INFRA-02**: Makefile with targets: download, audit, split, train, evaluate, gradcam, report, demo, all
- [ ] **INFRA-03**: Reproducible random seeds for all stochastic operations
- [ ] **INFRA-04**: YAML config file for all hyperparameters and paths

### Demo (Optional)

- [ ] **DEMO-01**: Streamlit app: upload image → prediction with confidence bars + Grad-CAM overlay + non-clinical-use disclaimer

## v2 Requirements

Deferred to future work. Tracked but not in current roadmap.

### Extended Classification

- **VEXT-01**: 9-subtype classification (requires more data or advanced techniques for rare subtypes)
- **VEXT-02**: Multi-task learning (classification + detection using annotations)
- **VEXT-03**: Segmentation task using LabelMe masks

### Advanced Evaluation

- **VEXT-04**: Subgroup fairness analysis (performance by age, gender, anatomical site)
- **VEXT-05**: Calibration analysis (reliability diagram + ECE)
- **VEXT-06**: Quantitative Grad-CAM IoU against annotations

### Production Readiness

- **VEXT-07**: REST API for inference
- **VEXT-08**: Docker containerized environment
- **VEXT-09**: Experiment tracking (W&B or MLflow)
- **VEXT-10**: External dataset validation

## Out of Scope

| Feature | Reason |
|---------|--------|
| Hyperparameter optimization / AutoML | PoC goal is feasibility, not squeezing marginal AUC gains. HPO on 3.7K images risks overfitting to val set |
| Ensemble of multiple architectures | Complicates inference, explainability (whose Grad-CAM?), and reproducibility |
| Custom loss functions (triplet, contrastive) | Exotic losses add complexity without proven benefit at PoC scale |
| 9-subtype classification | Some subtypes have <50 samples — results would be statistically meaningless |
| DICOM handling / PACS integration | BTXRD provides JPEGs; DICOM pipeline adds zero benefit for this dataset |
| Test-time augmentation | Complicates inference for marginal gains; note as future improvement |
| Production deployment / serving | This is a PoC, not a service |
| Patient-level grouping heuristics | No patient_id exists; fabricating groupings introduces unreliable assumptions. Acknowledge limitation honestly instead |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Pending |
| INFRA-02 | Phase 1 | Pending |
| INFRA-03 | Phase 1 | Pending |
| INFRA-04 | Phase 1 | Pending |
| DOCS-05 | Phase 1 | Pending |
| DOCS-06 | Phase 1 | Pending |
| DATA-01 | Phase 2 | Pending |
| DATA-02 | Phase 2 | Pending |
| DATA-03 | Phase 2 | Pending |
| DOCS-01 | Phase 2 | Pending |
| DOCS-02 | Phase 2 | Pending |
| DATA-04 | Phase 3 | Pending |
| DATA-05 | Phase 3 | Pending |
| DATA-06 | Phase 3 | Pending |
| DATA-07 | Phase 3 | Pending |
| TRAIN-01 | Phase 3 | Pending |
| TRAIN-02 | Phase 3 | Pending |
| TRAIN-03 | Phase 4 | Pending |
| TRAIN-04 | Phase 4 | Pending |
| TRAIN-05 | Phase 4 | Pending |
| TRAIN-06 | Phase 4 | Pending |
| TRAIN-07 | Phase 4 | Pending |
| TRAIN-08 | Phase 4 | Pending |
| EVAL-01 | Phase 5 | Pending |
| EVAL-02 | Phase 5 | Pending |
| EVAL-03 | Phase 5 | Pending |
| EVAL-04 | Phase 5 | Pending |
| EVAL-05 | Phase 5 | Pending |
| EVAL-06 | Phase 5 | Pending |
| EVAL-07 | Phase 5 | Pending |
| EVAL-08 | Phase 5 | Pending |
| EXPL-01 | Phase 6 | Pending |
| EXPL-02 | Phase 6 | Pending |
| EXPL-03 | Phase 6 | Pending |
| INFR-01 | Phase 6 | Pending |
| INFR-02 | Phase 6 | Pending |
| DOCS-03 | Phase 7 | Pending |
| DOCS-04 | Phase 7 | Pending |
| DEMO-01 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 35 total
- Mapped to phases: 35
- Unmapped: 0 ✓

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-19 after initial definition*
