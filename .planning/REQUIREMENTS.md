# Requirements: AssureXRay

**Defined:** 2026-02-19
**Core Value:** Deliver a reproducible, auditable 3-class bone tumor classification baseline with clear explainability outputs that a clinician can inspect and trust.

## v1 Requirements

Requirements for initial PoC release. Each maps to roadmap phases.

### Data Pipeline

- [x] **DATA-01**: Download script fetches BTXRD dataset from figshare and organizes into `data_raw/` (images/, annotations/, dataset.csv)
- [x] **DATA-02**: Data audit report covering class distribution, image dimension histogram, missing values, annotation coverage, duplicate detection, per-center breakdown
- [x] **DATA-03**: Dataset specification document describing all 37 columns, label derivation logic (malignant=1 -> Malignant, benign=1 -> Benign, tumor=0 -> Normal), and data provenance
- [x] **DATA-04**: Primary split: image-level stratified train/val/test (70/15/15) with class-label stratification and fixed random seed
- [x] **DATA-05**: Secondary split: center holdout (Center 1 -> train/val, Centers 2+3 -> test) with stratified train/val within Center 1
- [x] **DATA-06**: Split manifests saved as CSV files with image_id, split assignment, and label
- [x] **DATA-07**: Leakage risk explicitly documented in audit report and PoC report (no patient_id, same-lesion multi-angle images)

### Model Training

- [x] **TRAIN-01**: PyTorch dataset class loading images + labels from split manifests with configurable transforms
- [x] **TRAIN-02**: Augmentation pipeline: resize 224x224, ImageNet normalization, CLAHE (albumentations), horizontal flip, small rotation (+/-15 degrees)
- [x] **TRAIN-03**: EfficientNet-B0 backbone via timm with pretrained ImageNet weights, 3-class output head
- [x] **TRAIN-04**: Inverse-frequency weighted cross-entropy loss for class imbalance
- [x] **TRAIN-05**: Training script with configurable hyperparameters via YAML config (lr, batch_size, epochs, patience, backbone, loss_type)
- [x] **TRAIN-06**: Early stopping on validation loss with configurable patience
- [x] **TRAIN-07**: Checkpoint saving: best model (by val_loss) and final model
- [x] **TRAIN-08**: Training log CSV (epoch, train_loss, val_loss, val_metrics) and loss curve plot

### Evaluation

- [x] **EVAL-01**: ROC AUC (one-vs-rest per-class + macro) with plotted ROC curves
- [x] **EVAL-02**: PR AUC (per-class) with plotted precision-recall curves
- [x] **EVAL-03**: Per-class sensitivity (recall) and specificity from confusion matrix
- [x] **EVAL-04**: Confusion matrix heatmap (absolute counts + row-normalized)
- [x] **EVAL-05**: Classification report (precision, recall, F1 per class)
- [x] **EVAL-06**: Evaluation on both split strategies with comparison table
- [x] **EVAL-07**: Bootstrap 95% confidence intervals on AUC and per-class sensitivity (1000 iterations)
- [x] **EVAL-08**: Comparison table against BTXRD paper's YOLOv8s-cls baseline results

### Explainability

- [x] **EXPL-01**: Grad-CAM heatmap generation using pytorch-grad-cam on EfficientNet-B0 last conv layer
- [x] **EXPL-02**: Curated heatmap gallery: 3-5 examples each for TP, FP, FN per class, saved as image grid
- [x] **EXPL-03**: Heatmap overlay comparison against LabelMe tumor annotations (qualitative: does model attend to tumor region?)

### Inference

- [x] **INFR-01**: Single-image inference script: image path + checkpoint -> class prediction + softmax confidences + Grad-CAM overlay
- [x] **INFR-02**: Batch inference mode for evaluating a directory of images

### Documentation

- [x] **DOCS-01**: `dataset_spec.md` -- column definitions, label schema, data provenance, license (CC BY-NC-ND 4.0)
- [x] **DOCS-02**: `data_audit_report.md` -- auto-generated from audit script with embedded figures
- [x] **DOCS-03**: `model_card.md` -- architecture, training data, performance, limitations, ethical considerations
- [x] **DOCS-04**: `poc_report.md` -- executive summary, methods, results (both splits), clinical relevance, limitations, next steps
- [ ] **DOCS-05**: `README.md` -- setup instructions, data download, train/eval/infer commands, project structure
- [ ] **DOCS-06**: `requirements.txt` -- pinned Python dependencies

### Infrastructure

- [ ] **INFRA-01**: Project scaffold with standard ML PoC directory structure
- [ ] **INFRA-02**: Makefile with targets: download, audit, split, train, evaluate, gradcam, report, demo, all
- [ ] **INFRA-03**: Reproducible random seeds for all stochastic operations
- [ ] **INFRA-04**: YAML config file for all hyperparameters and paths

### Demo (Optional)

- [ ] **DEMO-01**: Streamlit app: upload image -> prediction with confidence bars + Grad-CAM overlay + non-clinical-use disclaimer

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
| 9-subtype classification | Some subtypes have <50 samples -- results would be statistically meaningless |
| DICOM handling / PACS integration | BTXRD provides JPEGs; DICOM pipeline adds zero benefit for this dataset |
| Test-time augmentation | Complicates inference for marginal gains; note as future improvement |
| Production deployment / serving | This is a PoC, not a service |
| Patient-level grouping heuristics | No patient_id exists; fabricating groupings introduces unreliable assumptions. Acknowledge limitation honestly instead |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| INFRA-01 | Phase 1 | Complete |
| INFRA-02 | Phase 1 | Complete |
| INFRA-03 | Phase 1 | Complete |
| INFRA-04 | Phase 1 | Complete |
| DOCS-05 | Phase 1 | Complete |
| DOCS-06 | Phase 1 | Complete |
| DATA-01 | Phase 2 | Complete |
| DATA-02 | Phase 2 | Complete |
| DATA-03 | Phase 2 | Complete |
| DOCS-01 | Phase 2 | Complete |
| DOCS-02 | Phase 2 | Complete |
| DATA-04 | Phase 3 | Complete |
| DATA-05 | Phase 3 | Complete |
| DATA-06 | Phase 3 | Complete |
| DATA-07 | Phase 3 | Complete |
| TRAIN-01 | Phase 3 | Complete |
| TRAIN-02 | Phase 3 | Complete |
| TRAIN-03 | Phase 4 | Complete |
| TRAIN-04 | Phase 4 | Complete |
| TRAIN-05 | Phase 4 | Complete |
| TRAIN-06 | Phase 4 | Complete |
| TRAIN-07 | Phase 4 | Complete |
| TRAIN-08 | Phase 4 | Complete |
| EVAL-01 | Phase 5 | Complete |
| EVAL-02 | Phase 5 | Complete |
| EVAL-03 | Phase 5 | Complete |
| EVAL-04 | Phase 5 | Complete |
| EVAL-05 | Phase 5 | Complete |
| EVAL-06 | Phase 5 | Complete |
| EVAL-07 | Phase 5 | Complete |
| EVAL-08 | Phase 5 | Complete |
| EXPL-01 | Phase 6 | Complete |
| EXPL-02 | Phase 6 | Complete |
| EXPL-03 | Phase 6 | Complete |
| INFR-01 | Phase 6 | Complete |
| INFR-02 | Phase 6 | Complete |
| DOCS-03 | Phase 7 | Complete |
| DOCS-04 | Phase 7 | Complete |
| DEMO-01 | Phase 8 | Pending |

**Coverage:**
- v1 requirements: 39 total
- Mapped to phases: 39
- Unmapped: 0

---
*Requirements defined: 2026-02-19*
*Last updated: 2026-02-20 after Phase 6 completion*
