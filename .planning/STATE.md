# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Deliver a reproducible, auditable 3-class bone tumor classification baseline with clear explainability outputs that a clinician can inspect and trust.
**Current focus:** Phase 5 - Evaluation (COMPLETE)

## Current Position

Phase: 5 of 8 (Evaluation) -- Complete
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-02-20 -- Completed 05-02-PLAN.md

Progress: [███████████░░░░░] 11/16 (68%)

## Performance Metrics

**Velocity:**
- Total plans completed: 11
- Average duration: 5.9 min
- Total execution time: 71 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 Scaffold | 2/2 | 7 min | 3.5 min |
| 02 Data Acquisition | 2/2 | 18 min | 9 min |
| 03 Splits & Loader | 2/2 | 17 min | 8.5 min |
| 04 Model Training | 3/3 | 5 min | 1.7 min |
| 05 Evaluation | 2/2 | 24 min | 12 min |

**Recent Trend:**
- Last 5 plans: 04-01 (2 min), 04-02 (3 min), 04-03 (n/a), 05-01 (10 min), 05-02 (14 min)
- Trend: Evaluation plans longer due to model inference on 1372 test images + 2000 bootstrap iterations

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

### Pending Todos

None.

### Blockers/Concerns

- [Research]: timm EfficientNet-B0 Grad-CAM target layer name must be verified against installed version in Phase 6
- [03-01]: Center 3 has only 27 Normal images in center-holdout test set -- model evaluation should note this imbalance
- [03-01]: Same-lesion multi-angle image leakage risk documented but NOT mitigated (no patient_id available)
- [05-01]: Center-holdout generalization gap is substantial (macro AUC 0.627 vs 0.846 stratified) -- expected but important for Phase 7 reporting
- [05-02]: Malignant sensitivity CI width is 27 percentage points (0.474-0.743) due to only 51 test samples -- Phase 7 must flag this as a key limitation

## Session Continuity

Last session: 2026-02-20T16:22Z
Stopped at: Completed 05-02-PLAN.md (Phase 5 complete)
Resume file: None
