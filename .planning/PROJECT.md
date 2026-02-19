# AssureXRay

## What This Is

A lean, reproducible proof-of-concept for 3-class classification of primary bone tumors from radiographs (Normal / Benign / Malignant). Built on the BTXRD dataset with clinician-friendly Grad-CAM explainability and a complete evaluation + reporting pack. Intended as a classification baseline to validate feasibility before investing in more complex architectures.

## Core Value

Deliver a reproducible, auditable classification baseline with clear explainability outputs that a clinician can inspect and trust — not a black box.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Download script fetches BTXRD dataset from figshare and organizes into `data_raw/`
- [ ] Data audit report covering class distribution, image dimensions, annotation coverage, missing values, and quality flags
- [ ] Dataset spec document describing columns, label schema, and data provenance
- [ ] Dual split strategy: primary (image-level stratified train/val/test 70/15/15) + secondary (center holdout: Center 1 train/val, Centers 2+3 test)
- [ ] Leakage warning in reports acknowledging same-lesion multi-angle images without patient_id
- [ ] PyTorch dataset loader with augmentation pipeline (resize, normalize, horizontal flip, rotation)
- [ ] 3-class classifier using pretrained EfficientNet-B0 backbone with class-imbalance handling (weighted loss or focal loss)
- [ ] Training script with configurable hyperparameters, early stopping, and checkpoint saving
- [ ] Evaluation script producing: ROC AUC (one-vs-rest), PR AUC, per-class sensitivity/specificity, confusion matrix, optional bootstrap CIs
- [ ] Evaluation on both split strategies with comparison in reports
- [ ] Grad-CAM heatmap generation for selected examples: TP, TN, FP, FN per class
- [ ] Inference script: single image in → prediction + Grad-CAM overlay out
- [ ] Model card documenting architecture, training details, performance, and limitations
- [ ] PoC report summarizing findings, clinical relevance, and next steps
- [ ] Optional Streamlit demo: upload image → prediction + heatmap display

### Out of Scope

- Multi-subtype classification (9 subtypes) — insufficient samples per subtype for PoC, defer to v2
- Segmentation or detection tasks — annotations exist but classification baseline comes first
- External dataset validation — only BTXRD for this PoC
- Hyperparameter optimization / AutoML — manual tuning sufficient for baseline
- Production deployment or API — this is a PoC, not a service
- Mobile or edge inference — desktop/server only

## Context

**Dataset:** BTXRD (Bone Tumor X-ray Radiograph Dataset)
- Paper: Yao et al., "A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors" (Scientific Data, 2025)
- DOI: https://doi.org/10.1038/s41597-024-04311-y
- Data: https://doi.org/10.6084/m9.figshare.27865398
- Code: https://github.com/SHUNHANYAO/BTXRD

**Dataset composition:**
- 3,746 JPEG radiographs: 1,879 normal, 1,525 benign, 342 malignant
- 3 source centers: Center 1 (Chinese hospitals, 78%), Center 2 (Radiopaedia), Center 3 (MedPix)
- Clinical metadata: age, gender, anatomical site (15 locations), shooting angle (frontal/lateral/oblique), region (upper limb/lower limb/pelvis)
- 9 tumor subtypes: osteochondroma (754), osteosarcoma (297), multiple osteochondromas (263), simple bone cyst (206), giant cell tumor, osteofibroma, synovial osteochondroma, other benign, other malignant
- Annotations: LabelMe JSON format (bounding boxes + segmentation masks for tumor images only)

**Labels file:** `dataset.csv` with 37 columns
- Key columns: `image_id`, `center`, `age`, `gender`, `tumor` (0/1), `benign` (0/1), `malignant` (0/1)
- Class derivation: `malignant=1` → Malignant, `benign=1` → Benign, `tumor=0` → Normal

**Known issues:**
- No patient_id — same lesion may have multiple angle images (frontal/lateral/oblique), creating leakage risk
- Class imbalance: Malignant (342) is ~5.5x smaller than Normal (1,879)
- Some images diagnosed empirically by radiologists (no pathology confirmation)
- Paper's baseline used YOLOv8s-cls with 80/20 random split — no patient-level grouping

**Prior work baseline (from paper):**
- YOLOv8s-cls accuracy: Normal 91.3%, Benign 88.1%, Malignant 73.4% (validation set)

## Constraints

- **Tech stack**: PyTorch + torchvision pretrained models (EfficientNet-B0 primary, ResNet50 fallback)
- **Reproducibility**: All scripts must be runnable from CLI with fixed seeds and config files
- **Scope**: Minimal PoC — no overengineering, no premature abstraction
- **License**: BTXRD dataset is CC BY-NC-ND 4.0 — non-commercial use only
- **Compute**: Must train on a single GPU (consumer-grade, ~8GB VRAM)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| 3-class framing (Normal/Benign/Malignant) | Most clinically useful — benign vs malignant distinction drives treatment decisions | — Pending |
| EfficientNet-B0 as primary backbone | Good accuracy-to-compute ratio, well-suited for medical imaging PoC | — Pending |
| Dual split strategy | Image-level stratified gives standard results; center holdout tests generalization | — Pending |
| Weighted loss for imbalance | Simpler than focal loss, sufficient for baseline; can upgrade if needed | — Pending |
| LabelMe annotations for Grad-CAM validation | Can compare attention regions to expert-annotated tumor regions | — Pending |

---
*Last updated: 2026-02-19 after initialization*
