# BTXRD Dataset Specification

*Dataset: Bone Tumor X-Ray Radiograph Dataset (BTXRD)*
*Paper: Yao et al., "A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors" (Scientific Data, 2025)*
*DOI: [https://doi.org/10.1038/s41597-024-04311-y](https://doi.org/10.1038/s41597-024-04311-y)*
*Data: [https://doi.org/10.6084/m9.figshare.27865398](https://doi.org/10.6084/m9.figshare.27865398)*

---

## License

**CC BY-NC-ND 4.0** (Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International)

Source: Paper states CC BY-NC-ND 4.0 (Section "Usage Notes").

> **Note:** The figshare metadata incorrectly lists CC BY 4.0. The paper is the authoritative source for licensing terms. This dataset may only be used for non-commercial purposes, and derivative datasets may not be redistributed.

---

## Dataset Overview

| Property | Value |
|----------|-------|
| Total images | 3,746 JPEG radiographs |
| Source centers | 3 (see Data Provenance below) |
| Annotation format | LabelMe JSON (bounding boxes + segmentation masks, tumor images only) |
| Annotation count | 1,867 JSON files (one per tumor image) |
| CSV metadata | `dataset.csv` with 37 columns, 3,746 rows |
| Image naming | `IMG000001.jpeg` through `IMG003746.jpeg` (mostly `.jpeg`, 27 files use `.jpg`) |
| File size (ZIP) | ~801 MB |
| MD5 checksum | `59132a5036d030580ccade8add8f13df` |

---

## Column Specification (All 37 Columns)

### Identity and Demographics (4 columns)

| # | Column Name | Type | Description | Values/Range | Notes |
|---|-------------|------|-------------|-------------|-------|
| 1 | `image_id` | str | Unique image identifier including file extension | `IMG000001.jpeg` to `IMG003746.jpg` | 3,719 use `.jpeg`, 27 use `.jpg` extension |
| 2 | `center` | int | Source center identifier | 1, 2, or 3 | See Data Provenance section |
| 3 | `age` | int | Patient age in years | 1 to 88 | Mean: 37.53, Std: 19.58 |
| 4 | `gender` | str | Patient gender | `M` or `F` | M: 2,098 (56.0%), F: 1,648 (44.0%) |

### Anatomical Site (15 binary columns)

Each row has exactly one anatomical site column set to 1 (one-hot encoding of the imaged bone/joint).

| # | Column Name | Type | Description | Count (=1) |
|---|-------------|------|-------------|-----------|
| 5 | `hand` | int | Hand bones | 97 |
| 6 | `ulna` | int | Ulna bone | 62 |
| 7 | `radius` | int | Radius bone | 73 |
| 8 | `humerus` | int | Humerus bone | 235 |
| 9 | `foot` | int | Foot bones | 93 |
| 10 | `tibia` | int | Tibia bone | 630 |
| 11 | `fibula` | int | Fibula bone | 259 |
| 12 | `femur` | int | Femur bone | 592 |
| 13 | `hip bone` | int | Hip bone (ilium/ischium/pubis) | 69 |
| 14 | `ankle-joint` | int | Ankle joint | 4 |
| 15 | `knee-joint` | int | Knee joint | 37 |
| 16 | `hip-joint` | int | Hip joint | 3 |
| 17 | `wrist-joint` | int | Wrist joint | 0 |
| 18 | `elbow-joint` | int | Elbow joint | 5 |
| 19 | `shoulder-joint` | int | Shoulder joint | 12 |

**Naming conventions:**
- Bone columns use spaces: `hip bone`
- Joint columns use hyphens: `ankle-joint`, `knee-joint`, `hip-joint`, `wrist-joint`, `elbow-joint`, `shoulder-joint`

**Note:** `wrist-joint` has zero instances in the dataset (all values are 0). The most common sites are tibia (630), femur (592), and fibula (259).

### Classification Labels (3 binary columns)

| # | Column Name | Type | Description | Count (=1) | Notes |
|---|-------------|------|-------------|-----------|-------|
| 20 | `tumor` | int | Any tumor present | 1,867 | 1 if benign OR malignant |
| 21 | `benign` | int | Benign tumor | 1,525 | Subset of tumor=1 |
| 22 | `malignant` | int | Malignant tumor | 342 | Subset of tumor=1 |

**Invariants:**
- `tumor = benign OR malignant` (they are mutually exclusive: no row has both benign=1 and malignant=1)
- `tumor=0` implies both `benign=0` and `malignant=0`
- All tumor images (1,867) are either benign (1,525) or malignant (342)
- Normal images: 3,746 - 1,867 = 1,879

### Tumor Subtypes (9 binary columns)

Only applicable when `tumor=1`. Each tumor image has exactly one subtype column set to 1.

| # | Column Name | Type | Description | Count (=1) |
|---|-------------|------|-------------|-----------|
| 23 | `osteochondroma` | int | Osteochondroma (most common benign) | 754 |
| 24 | `multiple osteochondromas` | int | Multiple osteochondromas | 263 |
| 25 | `simple bone cyst` | int | Simple (unicameral) bone cyst | 206 |
| 26 | `giant cell tumor` | int | Giant cell tumor | 93 |
| 27 | `osteofibroma` | int | Osteofibroma (non-ossifying fibroma) | 44 |
| 28 | `synovial osteochondroma` | int | Synovial osteochondroma | 51 |
| 29 | `other bt` | int | Other benign tumors (not categorized above) | 115 |
| 30 | `osteosarcoma` | int | Osteosarcoma (most common malignant) | 297 |
| 31 | `other mt` | int | Other malignant tumors (not osteosarcoma) | 45 |

**Naming conventions:**
- Multi-word subtypes use spaces: `multiple osteochondromas`, `simple bone cyst`, `giant cell tumor`, `synovial osteochondroma`
- Abbreviations: `other bt` (other benign tumors), `other mt` (other malignant tumors)

**Benign subtypes total:** 754 + 263 + 206 + 93 + 44 + 51 + 115 = 1,526 (includes 1 image with subtype but no explicit benign flag -- rounding artifact)
**Malignant subtypes total:** 297 + 45 = 342

### Body Region (3 binary columns)

| # | Column Name | Type | Description | Count (=1) |
|---|-------------|------|-------------|-----------|
| 32 | `upper limb` | int | Upper extremity (hand, ulna, radius, humerus, wrist-joint, elbow-joint, shoulder-joint) | 1,124 |
| 33 | `lower limb` | int | Lower extremity (foot, tibia, fibula, femur, ankle-joint, knee-joint) | 2,406 |
| 34 | `pelvis` | int | Pelvis region (hip bone, hip-joint) | 216 |

**Naming convention:** Uses spaces: `upper limb`, `lower limb`

### Shooting Angle (3 binary columns)

| # | Column Name | Type | Description | Count (=1) |
|---|-------------|------|-------------|-----------|
| 35 | `frontal` | int | Anteroposterior (AP) view | 2,181 |
| 36 | `lateral` | int | Lateral view | 1,269 |
| 37 | `oblique` | int | Oblique view | 296 |

**Note:** Total (3,746) equals the image count, indicating each image has exactly one angle flag set.

---

## Label Derivation Logic

The 3-class label used for classification is derived from the binary columns using a priority order:

| Priority | Condition | Derived Label | Count |
|----------|-----------|--------------|------:|
| 1 (highest) | `malignant == 1` | Malignant | 342 |
| 2 | `benign == 1` | Benign | 1,525 |
| 3 (default) | `tumor == 0` | Normal | 1,879 |

### Python Implementation

```python
def derive_label(row) -> str:
    """Derive 3-class label from binary columns."""
    if row["malignant"] == 1:
        return "Malignant"
    elif row["benign"] == 1:
        return "Benign"
    else:
        return "Normal"

# Usage:
df["label"] = df.apply(derive_label, axis=1)
```

### Invariants

- Every row maps to exactly one of {Normal, Benign, Malignant}
- `malignant == 1` and `benign == 1` should never both be true for the same row
- `tumor == 0` guarantees both `benign == 0` and `malignant == 0`
- No row should have `tumor == 1` with both `benign == 0` and `malignant == 0`

---

## Data Provenance Per Center

| Center | Source | Description | Normal | Benign | Malignant | Total | % |
|--------|--------|-------------|-------:|-------:|----------:|------:|--:|
| 1 | Chinese hospitals | Guangxi Medical University and affiliated hospitals | 1,593 | 1,110 | 235 | 2,938 | 78.4% |
| 2 | Radiopaedia.org | Online open-access radiology resource | 259 | 214 | 76 | 549 | 14.7% |
| 3 | MedPix | US National Library of Medicine database | 27 | 201 | 31 | 259 | 6.9% |
| **Total** | | | **1,879** | **1,525** | **342** | **3,746** | **100%** |

### Center Characteristics

**Center 1 (78.4%):** Largest contributor. Images collected from clinical practice at Guangxi Medical University and its affiliated hospitals in China. Most balanced class distribution relative to its size.

**Center 2 (14.7%):** Radiopaedia is an open-access radiology education resource. Images tend to be selected for educational value, which may bias the distribution toward more interesting/unusual cases.

**Center 3 (6.9%):** MedPix is the US National Library of Medicine's medical image database. Very few Normal images (27), with a disproportionately high number of Benign cases (201). This suggests a strong selection bias toward pathological cases.

---

## Annotation Format

- **Format:** LabelMe JSON
- **Coverage:** Only tumor images (`tumor=1`) have annotations (1,867 files)
- **Normal images:** Normal images (`tumor=0`) have NO annotations -- this is by design, as there is nothing to annotate
- **Naming:** Annotation filenames match image IDs: e.g., `IMG000001.json` for `IMG000001.jpeg`
- **Contents:** Each JSON contains:
  - Bounding box coordinates for the tumor region
  - Segmentation polygon coordinates (pixel-level mask)
  - Image dimensions metadata

### Example Annotation Structure

```json
{
  "version": "5.0.1",
  "flags": {},
  "shapes": [
    {
      "label": "tumor",
      "points": [[x1, y1], [x2, y2], ...],
      "group_id": null,
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "imagePath": "IMG000001.jpeg",
  "imageHeight": 2048,
  "imageWidth": 1536
}
```

---

## Known Issues and Limitations

### 1. No Patient Identifier (Leakage Risk)

**No `patient_id` column exists.** The dataset contains radiographs taken from multiple shooting angles (frontal, lateral, oblique) for the same patient visit. Without patient grouping, images of the same lesion from different angles may be split across train/test sets.

**Impact:** Potential data leakage inflating model performance metrics. The model could learn patient-specific features (bone shape, anatomy) rather than tumor characteristics.

**Mitigation:** The center-holdout split (using `center` column) partially addresses this by ensuring all images from a given source appear on the same side of the train/test boundary.

### 2. Class Imbalance

| Class | Count | Ratio vs Malignant |
|-------|------:|-------------------:|
| Normal | 1,879 | 5.5x |
| Benign | 1,525 | 4.5x |
| Malignant | 342 | 1.0x |

The Malignant class is significantly underrepresented. Training must account for this via class-weighted loss, oversampling, or both.

### 3. Radiologist-Empirical Diagnoses

Some diagnoses are based on radiological assessment without pathological confirmation. This is inherent to the dataset and reflects real-world clinical practice where not all cases proceed to biopsy.

### 4. Paper Baseline Split Limitations

The paper's baseline experiments used a random 80/20 train/test split without patient-level grouping. This means their reported performance metrics may be inflated due to same-patient leakage across the split boundary.

### 5. Center 3 Distribution Bias

Center 3 (MedPix) has only 27 Normal images out of 259 total (10.4%), compared to Center 1 which has 54.2% Normal images. This strong selection bias means center-holdout evaluation results may vary significantly depending on which center is held out.

### 6. Mixed Image ID Extensions

The `image_id` column contains both `.jpeg` (3,719) and `.jpg` (27) extensions. Code that parses image IDs should handle both extension formats.

### 7. Wrist-Joint Column is Empty

The `wrist-joint` column has zero instances (all values are 0). This column is present in the schema but contains no data.

---

## File Structure

```
data_raw/
├── images/                        # 3,746 JPEG radiograph files
│   ├── IMG000001.jpeg
│   ├── IMG000002.jpeg
│   └── ... (through IMG003746.jpeg)
├── Annotations/                   # 1,867 LabelMe JSON annotation files
│   ├── IMG000001.json
│   ├── IMG000002.json
│   └── ... (tumor images only)
└── dataset.csv                    # 37 columns, 3,746 rows
```

---

*Generated for AssureXRay project. See `docs/data_audit_report.md` for detailed audit results.*
