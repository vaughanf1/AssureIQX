# AssureXRay -- Bone Tumor Classification Model Card

**Date:** 2026-02-21
**Version:** v1.0-PoC
**Format:** Mitchell et al. (2019) "Model Cards for Model Reporting"

---

> **NOT FOR CLINICAL USE** -- This model is a research proof-of-concept. It has not been validated for diagnostic, screening, or treatment-planning purposes. Do not use this model for clinical decision-making.

---

## 1. Model Details

| Field | Value |
|-------|-------|
| Organization | Research proof-of-concept (AssureXRay) |
| Architecture | EfficientNet-B0 (timm 1.0.15) |
| Parameters | 4,011,391 |
| Fine-tuning | Full fine-tuning from ImageNet pretrained weights |
| Input | 224 x 224 RGB radiograph images |
| Output | 3-class softmax (Normal, Benign, Malignant) |
| Optimizer | Adam (lr=0.001, weight_decay=0.0001) |
| Scheduler | Cosine annealing |
| Batch size | 32 |
| Early stopping | Patience 7 (validation loss) |
| Loss | Weighted cross-entropy (inverse-frequency class weights) |
| Framework | PyTorch 2.6.0 / torchvision 0.21.0 |
| Seed | 42 (deterministic mode enabled) |

The model uses a standard EfficientNet-B0 backbone with the original classifier head replaced by a 3-class linear layer with 0.2 dropout. All layers are fine-tuned (no frozen layers). Class weights are computed as the inverse frequency of each class in the training set to address the 5.5x imbalance between Normal and Malignant classes.

## 2. Intended Use

**Primary intended use:** Research proof-of-concept for bone tumor radiograph classification feasibility. This model demonstrates whether transfer learning with EfficientNet-B0 can distinguish Normal, Benign, and Malignant bone lesions on the BTXRD dataset.

**Primary intended users:** Researchers and machine learning engineers evaluating bone tumor classification approaches and exploring the BTXRD dataset.

**Out-of-scope uses:**
- Clinical diagnosis or screening
- Treatment planning or clinical decision support
- Deployment in any healthcare setting
- Use as a standalone diagnostic tool
- Commercial applications (dataset license: CC BY-NC-ND 4.0)

### **NOT FOR CLINICAL USE**

This model has not undergone clinical validation, regulatory review, or prospective testing. It must not be used to inform any medical decisions. The performance metrics reported here reflect retrospective evaluation on a single dataset and do not establish clinical utility.

## 3. Factors

### Relevant Factors

- **Imaging center (source bias):** The training data is heavily skewed toward Center 1 (78% of all images, from Chinese hospitals). Centers 2 (Radiopaedia) and 3 (MedPix) contribute 15% and 7% respectively. This source imbalance is the dominant factor affecting generalization.
- **Anatomical site:** The dataset contains 14 anatomical sites (e.g., femur, tibia, humerus). Tumor appearance varies substantially by anatomical location.
- **Tumor subtype:** 9 tumor subtypes are present. The model classifies at the 3-class level (Normal/Benign/Malignant), not at the subtype level. Some subtypes (e.g., osteochondroma, n=754) are far better represented than others (e.g., other malignant, small counts).

### Demographic Factors

- **Age and gender:** Available in the dataset metadata but not analyzed for subgroup fairness in this proof-of-concept. Future work should evaluate per-demographic performance.

### Environmental Factors

- **Radiograph quality:** Varies by center and imaging equipment. Center 1 images come from clinical hospital settings; Centers 2 and 3 images come from online teaching repositories (Radiopaedia, MedPix) which may have different quality characteristics.
- **Image acquisition:** Mixed shooting angles (frontal, lateral, oblique). No standardization was applied beyond resizing and CLAHE contrast enhancement.

## 4. Metrics

### Primary Metrics

| Metric | Rationale |
|--------|-----------|
| **Macro AUC** | Threshold-independent measure of discrimination ability across all 3 classes. Averages one-vs-rest AUC equally, preventing the majority class from dominating. |
| **Malignant Sensitivity** | The headline clinical metric. In a screening context, missing a malignant tumor (false negative) is more harmful than a false alarm. Sensitivity measures the rate of correctly identifying malignant cases. |

### Supporting Metrics

- **Per-class AUC:** One-vs-rest AUC for Normal, Benign, and Malignant individually
- **Per-class sensitivity (recall):** Proportion of true positives correctly identified per class
- **Per-class specificity:** Proportion of true negatives correctly identified per class
- **Per-class precision:** Positive predictive value per class
- **Per-class F1-score:** Harmonic mean of precision and recall per class
- **Overall accuracy:** Proportion of all correct predictions

### Uncertainty Quantification

Bootstrap 95% confidence intervals computed with 1,000 iterations using the percentile method. CIs are reported for macro AUC and per-class sensitivity. The bootstrap resamples the test set with replacement, recomputing metrics each iteration.

## 5. Training Data

| Field | Value |
|-------|-------|
| Dataset | BTXRD (Bone Tumor X-ray Radiograph Dataset) |
| Total images | 3,746 radiographs |
| Source centers | 3 (Center 1: Chinese hospitals, Center 2: Radiopaedia, Center 3: MedPix) |
| License | CC BY-NC-ND 4.0 |

**Class distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 1,879 | 50.2% |
| Benign | 1,525 | 40.7% |
| Malignant | 342 | 9.1% |

**Center distribution:**

| Center | Images | Percentage |
|--------|--------|------------|
| Center 1 (Chinese hospitals) | ~2,938 | 78% |
| Center 2 (Radiopaedia) | ~561 | 15% |
| Center 3 (MedPix) | ~247 | 7% |

**Citation:** Yao et al., "A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors," Scientific Data, 2025. DOI: 10.1038/s41597-024-04311-y

**Preprocessing:** CLAHE contrast enhancement (applied to all images), ImageNet normalization, resize to 224x224. Training augmentation includes horizontal flip and rotation (+/-15 degrees).

## 6. Evaluation Data

Two evaluation strategies were employed to assess both standard and generalization performance:

### Stratified Split

- **Strategy:** Image-level stratified random split (70% train / 15% val / 15% test)
- **Test set size:** 564 images
- **Test class distribution:** 282 Normal, 231 Benign, 51 Malignant
- **Purpose:** Standard evaluation with class proportions preserved

### Center-Holdout Split

- **Strategy:** Center 1 for training/validation, Centers 2+3 for testing
- **Test set size:** 808 images
- **Test class distribution:** 286 Normal, 415 Benign, 107 Malignant
- **Purpose:** Tests generalization to unseen imaging sources -- a proxy for external validation

**Note:** Neither split strategy achieves patient-level separation due to the absence of patient identifiers in the dataset. Same-lesion multi-angle images may appear in both training and test sets.

## 7. Quantitative Analyses

### Stratified Split Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Macro AUC | 0.846 | 0.814--0.873 |
| Accuracy | 67.9% | -- |
| Malignant Sensitivity | 60.8% | 47.4%--74.3% |
| Malignant Specificity | 95.7% | -- |
| Malignant Precision | 58.5% | -- |
| Malignant F1 | 0.596 | -- |
| Benign Sensitivity | 78.4% | 72.8%--83.6% |
| Benign Specificity | 62.5% | -- |
| Benign Precision | 59.2% | -- |
| Normal Sensitivity | 60.6% | 54.6%--66.3% |
| Normal Specificity | 87.9% | -- |
| Normal Precision | 83.4% | -- |

**Per-class AUC:**

| Class | AUC |
|-------|-----|
| Normal | 0.843 |
| Benign | 0.788 |
| Malignant | 0.906 |

### Center-Holdout Split Performance

| Metric | Value | 95% CI |
|--------|-------|--------|
| Macro AUC | 0.627 | 0.594--0.658 |
| Accuracy | 47.2% | -- |
| Malignant Sensitivity | 36.4% | 27.0%--44.8% |
| Malignant Specificity | 79.9% | -- |
| Malignant Precision | 21.7% | -- |
| Malignant F1 | 0.272 | -- |
| Benign Sensitivity | 64.3% | 59.9%--68.8% |
| Benign Specificity | 41.2% | -- |
| Benign Precision | 53.6% | -- |
| Normal Sensitivity | 26.2% | 20.9%--32.0% |
| Normal Specificity | 89.5% | -- |
| Normal Precision | 57.7% | -- |

**Per-class AUC:**

| Class | AUC |
|-------|-----|
| Normal | 0.653 |
| Benign | 0.574 |
| Malignant | 0.653 |

### Generalization Gap (Center-Holdout minus Stratified)

| Metric | Stratified | Center-Holdout | Gap |
|--------|-----------|----------------|-----|
| Macro AUC | 0.846 | 0.627 | -0.219 |
| Malignant Sensitivity | 60.8% | 36.4% | -24.3 pp |
| Accuracy | 67.9% | 47.2% | -20.8 pp |

The substantial generalization gap (-0.219 AUC) demonstrates that the model has learned center-specific features from Center 1 that do not transfer well to Centers 2 and 3.

## 8. Caveats and Recommendations

### Limitations

**1. Same-lesion multi-angle image leakage risk.** The dataset lacks patient identifiers. The same lesion may be imaged from multiple angles (frontal, lateral, oblique), and these images could appear in both training and test sets under the stratified split. This inflates stratified performance estimates. Duplicate detection found 21 exact image pairs, but near-duplicates from different angles are not detectable.

**2. Label noise ceiling.** Labels are based on radiologist diagnosis, not pathology confirmation. Radiographic diagnosis of bone tumors has inherent uncertainty, particularly for distinguishing benign from malignant lesions. This sets an upper bound on achievable classification accuracy.

**3. Substantial center generalization gap.** Macro AUC drops from 0.846 (stratified) to 0.627 (center-holdout), a gap of -0.219. Malignant sensitivity drops from 60.8% to 36.4%. The model has overfit to Center 1 imaging characteristics.

**4. Absence of pathology-confirmed labels.** Ground truth is radiologist-assigned labels, not histopathological diagnosis. An unknown fraction of labels may be incorrect, particularly for ambiguous cases.

**5. Small Malignant test set.** Only 51 Malignant images in the stratified test set and 107 in the center-holdout test set. This produces wide confidence intervals and limits statistical power for the most clinically important class.

**6. Wide confidence intervals.** Malignant sensitivity 95% CI spans 27 percentage points (47.4%--74.3%) on the stratified split, reflecting the small sample size. The true sensitivity could plausibly be anywhere in this range.

**7. Low Grad-CAM IoU with expert annotations.** Mean IoU between Grad-CAM attention maps and expert tumor annotations is 0.070 (on a 0--1 scale). Only 1 of 5 evaluated images showed IoU above 0.1. The model may be using contextual cues (bone shape, surrounding tissue) rather than focusing on the tumor itself.

**8. Single architecture evaluated.** Only EfficientNet-B0 was trained. No architecture comparison, ensemble methods, or hyperparameter optimization was performed. Better architectures or configurations likely exist.

**9. Center 3 Normal class sparsity.** Center 3 contributes only 27 Normal images to the center-holdout test set. Evaluation of Normal class performance on Center 3 data is unreliable.

### Recommendations

- **Do not use for clinical decision-making.** This model is a research baseline only.
- **Patient-level splits required.** Future work must obtain patient identifiers to create proper train/test separation and eliminate leakage risk.
- **Pathology-confirmed labels needed.** Clinical-grade evaluation requires histopathologically verified ground truth.
- **Multi-center external validation required.** Performance on data from hospitals and imaging equipment not represented in BTXRD must be established before any clinical consideration.
- **Architecture and hyperparameter exploration.** Compare against ResNet, DenseNet, Vision Transformer, and conduct systematic hyperparameter optimization.
- **Larger Malignant sample.** Additional malignant cases would narrow confidence intervals and improve evaluation reliability.

## 9. Ethical Considerations

### Non-Clinical Use Disclaimer

**NOT FOR CLINICAL USE.** This model must not be used for diagnosis, screening, treatment planning, or any clinical decision-making. It is a research proof-of-concept only.

### Dataset Bias

The training data is predominantly from Chinese hospitals (Center 1, 78%). Performance on radiographs from other populations, imaging equipment, or clinical settings is unknown and likely degraded, as evidenced by the center-holdout results.

### Potential for Misuse

If deployed as a clinical tool without proper validation, this model could:
- Miss malignant tumors (36.4% sensitivity on out-of-distribution data means ~64% of malignant cases would be missed)
- Generate false confidence in incorrect predictions
- Disproportionately fail for patient populations not represented in the training data

### License Restrictions

The BTXRD dataset is licensed under CC BY-NC-ND 4.0, which prohibits commercial use and derivative works. Any model trained on this data inherits these restrictions.

---

**References:**

1. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220--228.
2. Yao, S., et al. (2025). A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors. *Scientific Data*. DOI: 10.1038/s41597-024-04311-y
3. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML 2019*.
4. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV 2017*.
