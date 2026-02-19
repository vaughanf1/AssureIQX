# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-02-19)

**Core value:** Deliver a reproducible, auditable 3-class bone tumor classification baseline with clear explainability outputs that a clinician can inspect and trust.
**Current focus:** Phase 4 - Model and Training

## Current Position

Phase: 3 of 8 (Splits and Dataset Loader) -- COMPLETE
Plan: 2 of 2 in current phase
Status: Phase complete
Last activity: 2026-02-19 -- Completed 03-02-PLAN.md

Progress: [██████░░░░░░░░░░] 6/16 (38%)

## Performance Metrics

**Velocity:**
- Total plans completed: 6
- Average duration: 7 min
- Total execution time: 42 min

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01 Scaffold | 2/2 | 7 min | 3.5 min |
| 02 Data Acquisition | 2/2 | 18 min | 9 min |
| 03 Splits & Loader | 2/2 | 17 min | 8.5 min |

**Recent Trend:**
- Last 5 plans: 02-01 (11 min), 02-02 (7 min), 03-01 (15 min), 03-02 (2 min)
- Trend: 03-02 fast -- straightforward implementation with no blockers

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

### Pending Todos

None.

### Blockers/Concerns

- [Research]: timm EfficientNet-B0 Grad-CAM target layer name must be verified against installed version in Phase 6
- [03-01]: Center 3 has only 27 Normal images in center-holdout test set -- model evaluation should note this imbalance
- [03-01]: Same-lesion multi-angle image leakage risk documented but NOT mitigated (no patient_id available)

## Session Continuity

Last session: 2026-02-20T17:31Z
Stopped at: Completed 03-02-PLAN.md (Phase 3 complete)
Resume file: None
