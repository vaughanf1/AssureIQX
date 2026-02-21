# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Deliver a reproducible, auditable 3-class bone tumor classification baseline with clear explainability outputs that a clinician can inspect and trust.
**Current focus:** Phase 7 - Documentation and Reports (Complete)

## Current Position

Phase: 7 of 8 (Documentation and Reports) -- Complete
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-02-21 -- Completed 07-02-PLAN.md

Progress: [███████████████░] 15/16 (94%)

## Performance Metrics

**Velocity:**
- Total plans completed: 15
- Average duration: 5.3 min
- Total execution time: 83 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 Scaffold | 2/2 | 7 min | 3.5 min |
| 02 Data Acquisition | 2/2 | 18 min | 9 min |
| 03 Splits & Loader | 2/2 | 17 min | 8.5 min |
| 04 Model Training | 3/3 | 5 min | 1.7 min |
| 05 Evaluation | 2/2 | 24 min | 12 min |
| 06 Explainability | 2/2 | 5 min | 2.5 min |
| 07 Documentation | 2/2 | 7 min | 3.5 min |

**Recent Trend:**
- Last 5 plans: 05-02 (14 min), 06-01 (2 min), 06-02 (3 min), 07-01 (5 min), 07-02 (2 min)
- Trend: Documentation plans efficient due to well-structured data sources from prior phases

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: 8-phase sequential pipeline following data dependency chain
- [Roadmap]: Dual split strategy (stratified + center holdout) implemented in Phase 3
- [Roadmap]: Evaluation metric suite designed before training (Phase 5 criteria defined upfront)
- [01-01]: Placeholder modules are docstring-only stubs (no imports or code)
- [01-01]: torch==2.6.0 / torchvision==0.21.0 matched pair pinned
- [01-02]: yaml.safe_load only (never yaml.load) for config security
- [01-02]: CLI overrides via dot-notation with yaml.safe_load type coercion
- [01-02]: torch.use_deterministic_algorithms(True, warn_only=True) for reproducibility
- [01-02]: Standardized script template: shebang, docstring, PROJECT_ROOT, argparse, config+seed, NotImplementedError
- [02-01]: Dataset ships as xlsx, converted to CSV during extraction (openpyxl added)
- [02-01]: Column names use spaces (hip bone, simple bone cyst) and hyphens (ankle-joint), no underscores
- [02-01]: Image filenames use .jpeg extension (IMG000001.jpeg), not .jpg
- [02-01]: Nested Annotations/Annotations/ is a ZIP artifact, removed during extraction
- [02-02]: No missing values in dataset (all 37 columns fully populated)
- [02-02]: 21 exact duplicate image pairs found via phash -- inform Phase 3 splitting
- [02-02]: wrist-joint column is empty (all zeros) -- present in schema but no data
- [02-02]: Mixed image_id extensions: 3,719 .jpeg + 27 .jpg
- [02-02]: Proxy grouping: 295 groups (center+age+gender), 941 groups (+site) -- insufficient for patient-level grouping
- [03-01]: Representative-based splitting: first duplicate group member participates in split, all members assigned to same partition
- [03-01]: Center-holdout val_ratio=15% applied to Center 1 only (439 val images from 2938 Center 1 total)
- [03-01]: phash hash_size=8 distance=0 for duplicate detection (21 pairs found, matches Phase 2 audit)
- [03-01]: Split CSV format: image_id,split,label (3 columns, no index)
- [03-02]: CLAHE p=1.0 (always applied) for radiograph contrast enhancement in training
- [03-02]: get_test_transforms aliases get_val_transforms (identical deterministic pipeline)
- [03-02]: Image.open().convert("RGB") handles both grayscale and color inputs uniformly
- [03-02]: Dataset validates labels at init time (fail-fast on unknown labels)
- [04-01]: gradcam_target_layer returns model.bn2 (BatchNormAct2d 1280) -- verified for Phase 6
- [04-01]: save_checkpoint includes class_weights as list for checkpoint self-documentation
- [04-01]: compute_class_weights returns CPU tensor -- caller responsible for .to(device)
- [04-01]: Full fine-tuning (no frozen layers) as recommended for small dataset transfer learning
- [04-02]: Weighted CrossEntropyLoss for training only; unweighted for val/early stopping -- prevents class weight from distorting stopping signal
- [04-02]: Split-prefixed checkpoint names (best_stratified.pt, best_center.pt) -- avoids overwriting when running both splits
- [04-02]: Results directory naming: stratified/ and center_holdout/ -- matches Phase 3 convention
- [05-01]: Softmax applied exactly once in run_inference; model outputs raw logits
- [05-01]: One-vs-Rest strategy for ROC and PR curves via label_binarize
- [05-01]: Malignant sensitivity logged as headline metric over overall accuracy
- [05-02]: Percentile method for bootstrap CIs with class-presence guard (critical for 51 Malignant samples)
- [05-02]: 7 caveats for BTXRD baseline comparison (mAP vs AUC, random split, different architecture/size)
- [05-02]: Generalization gap as center_holdout minus stratified (negative = worse)
- [06-01]: Target predicted class for Grad-CAM (shows model's actual decision rationale)
- [06-01]: IoU threshold 0.5 for Grad-CAM binarization (standard in literature)
- [06-01]: Stratified checkpoint for gallery (better performance = more interpretable heatmaps)
- [06-01]: Mean IoU 0.070 between Grad-CAM and annotations -- model attention weakly aligned with tumor regions
- [06-02]: Grad-CAM targets predicted class at inference (ground truth unavailable)
- [06-02]: Class names loaded from checkpoint metadata for portability
- [06-02]: PIL.Image.fromarray for overlay saving (avoids BGR/RGB cv2 confusion)
- [07-01]: Mitchell et al. 2019 format for model card (9 sections plus references)
- [07-01]: Clinical decision framing as sensitivity at specificity
- [07-01]: 9 documented limitations covering data, model, and evaluation concerns
- [07-02]: 10-section README structure for progressive disclosure
- [07-02]: Report target verifies docs instead of echoing manual message
- [07-02]: All target uses train-all for both splits

### Pending Todos

None.

### Blockers/Concerns

- [03-01]: Center 3 has only 27 Normal images in center-holdout test set -- documented in model card and PoC report
- [03-01]: Same-lesion multi-angle image leakage risk documented in both deliverables as limitation #1
- [05-01]: Center-holdout generalization gap documented with full comparison table in both deliverables
- [05-02]: Malignant sensitivity CI width (27 pp) documented as limitation #6 in both deliverables
- [06-01]: Low Grad-CAM IoU (mean 0.070) documented as limitation #7 in both deliverables

## Session Continuity

Last session: 2026-02-21T02:24Z
Stopped at: Completed 07-02-PLAN.md
Resume file: None
