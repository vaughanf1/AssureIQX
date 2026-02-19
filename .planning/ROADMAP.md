# Roadmap: AssureXRay

## Overview

AssureXRay delivers a reproducible, auditable 3-class bone tumor classification baseline (Normal / Benign / Malignant) on the BTXRD dataset, with clinician-facing Grad-CAM explainability and a complete evaluation pack. The roadmap follows the natural data dependency chain: scaffold the project, acquire and audit data, prepare splits and loaders, train the model, evaluate comprehensively, generate explainability outputs and inference tooling, synthesize documentation, and optionally build a demo. Phases 1-5 are strictly sequential; Phase 6 depends on Phase 5 outputs; Phase 7 depends on all prior phases; Phase 8 depends on Phase 6.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Scaffold and Infrastructure** - Project skeleton, config, Makefile, reproducibility utilities
- [x] **Phase 2: Data Acquisition and Audit** - Download BTXRD, profile the dataset, produce audit and spec documents
- [ ] **Phase 3: Splits and Dataset Loader** - Dual split strategy, split manifests, PyTorch dataset class, augmentation pipeline
- [ ] **Phase 4: Model Training** - EfficientNet-B0 classifier, weighted loss, training loop with early stopping and checkpoints
- [ ] **Phase 5: Evaluation** - Full metric suite on both splits, bootstrap CIs, comparison against paper baseline
- [ ] **Phase 6: Explainability and Inference** - Grad-CAM heatmaps, curated gallery, annotation comparison, single-image and batch inference
- [ ] **Phase 7: Documentation and Reports** - Model card, PoC report with clinical framing and limitations
- [ ] **Phase 8: Streamlit Demo (Optional)** - Interactive image upload with prediction and Grad-CAM overlay

## Phase Details

### Phase 1: Scaffold and Infrastructure
**Goal**: A developer can clone the repo, install dependencies, and see the complete project structure with all configuration in place -- ready to receive data and code
**Depends on**: Nothing (first phase)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, DOCS-05, DOCS-06
**Success Criteria** (what must be TRUE):
  1. Running `pip install -r requirements.txt` installs all dependencies without errors on Python 3.10-3.12
  2. The project directory structure matches the architecture spec (`src/`, `scripts/`, `configs/`, `data/`, `results/`, `docs/`, `app/`) with `__init__.py` files and placeholder modules
  3. `configs/default.yaml` contains all hyperparameters, paths, and seeds referenced in the architecture (lr, batch_size, epochs, patience, backbone, loss_type, split ratios, random seed)
  4. `make` with no target prints available Makefile targets (download, audit, split, train, evaluate, gradcam, infer, report, demo, all)
  5. `src/utils/reproducibility.py` sets deterministic seeds for random, numpy, torch, and torch.cuda when called
**Plans**: 2 plans

Plans:
- [x] 01-01-PLAN.md -- Project directory structure, .gitignore, requirements.txt, and README.md
- [x] 01-02-PLAN.md -- Configuration system, Makefile, reproducibility utilities, and placeholder scripts

### Phase 2: Data Acquisition and Audit
**Goal**: The raw BTXRD dataset is downloaded, organized, profiled, and documented -- all data quality issues are surfaced before any modeling begins
**Depends on**: Phase 1
**Requirements**: DATA-01, DATA-02, DATA-03, DOCS-01, DOCS-02
**Success Criteria** (what must be TRUE):
  1. Running `make download` (or `python scripts/download.py`) fetches the BTXRD dataset from figshare and organizes files into `data_raw/images/`, `data_raw/annotations/`, and `data_raw/dataset.csv`
  2. Running `make audit` (or `python scripts/audit.py`) generates `docs/data_audit_report.md` with embedded figures showing class distribution (confirming 1,879 / 1,525 / 342), image dimension histogram, per-center breakdown, missing value counts, annotation coverage, and duplicate detection results
  3. `docs/dataset_spec.md` documents all 37 columns of `dataset.csv`, the label derivation logic (malignant=1 then Malignant, benign=1 then Benign, tumor=0 then Normal), data provenance per center, and the CC BY-NC-ND 4.0 license
  4. The audit report explicitly documents the leakage risk from same-lesion multi-angle images (no patient_id) and notes which metadata columns are available for proxy patient grouping
**Plans**: 2 plans

Plans:
- [x] 02-01-PLAN.md -- Download script, config fix, dependency update, and data verification
- [x] 02-02-PLAN.md -- Data audit script with report generation and dataset specification document

### Phase 3: Splits and Dataset Loader
**Goal**: Data is split into reproducible train/val/test sets using two strategies, and a PyTorch dataset class loads images with the correct augmentation pipeline for each mode (train/val/test)
**Depends on**: Phase 2
**Requirements**: DATA-04, DATA-05, DATA-06, DATA-07, TRAIN-01, TRAIN-02
**Success Criteria** (what must be TRUE):
  1. Running `make split` (or `python scripts/split.py`) produces CSV manifests in `data/splits/` for both strategies: `stratified_{train,val,test}.csv` (70/15/15) and `center_{train,val,test}.csv` (Center 1 train/val, Centers 2+3 test), each containing image_id, split, and label columns
  2. No image_id appears in more than one split within the same strategy, class proportions are preserved in stratified splits, and Malignant count in each split is non-trivial (verifiable by inspecting the CSV files)
  3. The PyTorch dataset class loads images from split manifests and returns correctly shaped tensors (batch, 3, 224, 224) with ImageNet normalization applied
  4. Training mode applies the full augmentation pipeline (CLAHE, horizontal flip, rotation +/-15 degrees, resize 224x224, ImageNet normalize) while validation/test modes apply only deterministic transforms (resize, normalize)
  5. The leakage risk from same-lesion multi-angle images is documented in both the split script output and referenced in the audit report, with the proxy grouping strategy (if feasible from available metadata) applied or its absence explicitly justified
**Plans**: TBD

Plans:
- [ ] 03-01: Split strategy implementation and manifest generation
- [ ] 03-02: PyTorch dataset class and augmentation pipeline

### Phase 4: Model Training
**Goal**: A trained EfficientNet-B0 classifier exists for both split strategies, with training logs proving convergence and class-imbalance handling preventing Malignant class collapse
**Depends on**: Phase 3
**Requirements**: TRAIN-03, TRAIN-04, TRAIN-05, TRAIN-06, TRAIN-07, TRAIN-08
**Success Criteria** (what must be TRUE):
  1. Running `make train` (or `python scripts/train.py --config configs/default.yaml`) trains the EfficientNet-B0 model with pretrained ImageNet weights and a 3-class output head, using inverse-frequency weighted cross-entropy loss
  2. Training completes with early stopping triggered by validation loss plateau (configurable patience), and saves both best (by val_loss) and final checkpoints to `checkpoints/`
  3. Checkpoints contain model weights, optimizer state, full config, class names, and normalization stats -- loading a checkpoint and running a forward pass produces valid 3-class softmax outputs
  4. Training logs (`results/{split_name}/training_log.csv`) record epoch, train_loss, val_loss, and val_metrics, and loss curve plots are saved as PNG files showing convergence
  5. Per-class training accuracy confirms the Malignant class is not collapsed (Malignant recall > 0 throughout training)
**Plans**: TBD

Plans:
- [ ] 04-01: Model architecture and loss function
- [ ] 04-02: Training loop, early stopping, and checkpoint management
- [ ] 04-03: Train on both split strategies and verify convergence

### Phase 5: Evaluation
**Goal**: Both trained models are comprehensively evaluated with clinically relevant metrics, producing a side-by-side comparison that reveals the generalization gap between stratified and center-holdout performance
**Depends on**: Phase 4
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08
**Success Criteria** (what must be TRUE):
  1. Running `make evaluate` (or `python scripts/eval.py`) produces per-split result directories (`results/stratified/`, `results/center_holdout/`) each containing: ROC curves (one-vs-rest per class + macro AUC), PR curves (per class), confusion matrix heatmaps (absolute + row-normalized), and a classification report (precision, recall, F1 per class)
  2. Per-class sensitivity (recall) and specificity are computed and reported, with Malignant sensitivity as the headline metric rather than overall accuracy
  3. A comparison table shows side-by-side metrics for both split strategies, making the center-holdout generalization gap explicit
  4. Bootstrap 95% confidence intervals (1000 iterations) are computed for AUC and per-class sensitivity, saved as JSON
  5. A comparison against the BTXRD paper's YOLOv8s-cls baseline is included, with caveats about differences in split methodology
**Plans**: TBD

Plans:
- [ ] 05-01: Core metrics and visualization pipeline
- [ ] 05-02: Dual-split comparison, bootstrap CIs, and baseline comparison

### Phase 6: Explainability and Inference
**Goal**: Clinicians can inspect Grad-CAM heatmaps showing where the model attends for each prediction, and a CLI tool produces predictions with confidence scores and visual overlays for any input image
**Depends on**: Phase 5
**Requirements**: EXPL-01, EXPL-02, EXPL-03, INFR-01, INFR-02
**Success Criteria** (what must be TRUE):
  1. Running `make gradcam` (or `python scripts/gradcam.py`) generates Grad-CAM heatmap overlays for systematically selected examples: 3-5 each of TP, FP, FN per class, saved as image grids in `results/gradcam/`
  2. Heatmaps for correctly classified tumor images visually highlight anatomical regions (not image borders or artifacts), and FP/FN heatmaps suggest interpretable failure modes
  3. A qualitative comparison of Grad-CAM attention regions against LabelMe tumor annotations is produced and documented (does the model attend to the annotated tumor region?)
  4. Running `python scripts/infer.py --image path/to/image.jpg --checkpoint path/to/best.pt` outputs class prediction, softmax confidence scores, and a Grad-CAM overlay PNG
  5. Batch inference mode (`python scripts/infer.py --input-dir path/to/images/`) processes a directory of images and outputs predictions for each
**Plans**: TBD

Plans:
- [ ] 06-01: Grad-CAM generation and curated gallery
- [ ] 06-02: Single-image and batch inference scripts

### Phase 7: Documentation and Reports
**Goal**: The PoC is fully documented with a model card, a comprehensive PoC report, and a README that enables clean-room reproduction -- documentation is the deliverable
**Depends on**: Phase 6
**Requirements**: DOCS-03, DOCS-04
**Success Criteria** (what must be TRUE):
  1. `docs/model_card.md` follows the Mitchell et al. (2019) format and documents: model architecture, training data description, intended use, performance metrics (both splits), limitations, and ethical considerations including the non-clinical-use disclaimer
  2. `docs/poc_report.md` contains: executive summary, methods (data, splits, model, training), results (both splits with comparison table), Grad-CAM findings, explicit limitations section (leakage risk, label noise ceiling, center generalization gap, absence of pathology-confirmed labels), clinical relevance framing, and recommended next steps
  3. The PoC report includes a clinical decision framing statement (e.g., "at X% specificity, the model achieves Y% sensitivity for malignant tumors")
  4. `README.md` is updated with complete setup instructions, data download commands, train/eval/infer CLI commands, and project structure -- a new developer can reproduce results from scratch
**Plans**: TBD

Plans:
- [ ] 07-01: Model card and PoC report
- [ ] 07-02: README finalization and end-to-end reproduction verification

### Phase 8: Streamlit Demo (Optional)
**Goal**: A non-technical user can upload a radiograph image and see a prediction with confidence bars and Grad-CAM overlay in a browser, with a clear disclaimer that this is not for clinical use
**Depends on**: Phase 6
**Requirements**: DEMO-01
**Success Criteria** (what must be TRUE):
  1. Running `make demo` (or `streamlit run app/app.py`) launches a local web app that allows image upload
  2. Uploaded images produce a class prediction with confidence bar chart and Grad-CAM heatmap overlay, matching the output of the CLI inference script
  3. A prominent "NOT FOR CLINICAL USE -- Research Prototype Only" disclaimer is visible on every page of the app
**Plans**: TBD

Plans:
- [ ] 08-01: Streamlit demo application

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Scaffold and Infrastructure | 2/2 | Complete | 2026-02-19 |
| 2. Data Acquisition and Audit | 2/2 | Complete | 2026-02-19 |
| 3. Splits and Dataset Loader | 0/2 | Not started | - |
| 4. Model Training | 0/3 | Not started | - |
| 5. Evaluation | 0/2 | Not started | - |
| 6. Explainability and Inference | 0/2 | Not started | - |
| 7. Documentation and Reports | 0/2 | Not started | - |
| 8. Streamlit Demo (Optional) | 0/1 | Not started | - |
