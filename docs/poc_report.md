# AssureXRay: Proof-of-Concept Report

## 3-Class Bone Tumor Classification from Radiographs

**Date:** 2026-02-21
**Version:** v1.0
**Authors:** AssureXRay Research Team

---

> **NOT FOR CLINICAL USE** -- This is a research proof-of-concept. The model has not been validated for diagnostic purposes. Do not use this system for clinical decision-making, screening, or treatment planning.

---

## 1. Executive Summary

This report presents a proof-of-concept (PoC) for automated 3-class classification of primary bone tumors from plain radiographs. Using the publicly available BTXRD dataset (3,746 images from 3 centers), we trained an EfficientNet-B0 classifier to distinguish Normal, Benign, and Malignant bone lesions. The model was evaluated under two complementary strategies: a standard stratified split and a center-holdout split that tests generalization to unseen imaging sources.

On the stratified test set (564 images), the model achieves a macro AUC of 0.846 (95% CI: 0.814--0.873) and overall accuracy of 67.9%. On the center-holdout test set (808 images from Centers 2 and 3), macro AUC drops to 0.627 (95% CI: 0.594--0.658) with accuracy of 47.2%, revealing a substantial generalization gap of -0.219 AUC.

At the default operating point, the model achieves 60.8% sensitivity (95% CI: 47.4%--74.3%) for malignant tumors at 95.7% specificity on the stratified test set. On the center-holdout test set, malignant sensitivity drops to 36.4% (95% CI: 27.0%--44.8%) at 79.9% specificity.

These results demonstrate feasibility for within-distribution classification but reveal that cross-center generalization requires substantial further work. Grad-CAM explainability analysis shows low overlap (mean IoU 0.070) between model attention and expert tumor annotations, suggesting the model relies on contextual cues rather than focal tumor features. We document 9 key limitations and recommend patient-level splits with pathology-confirmed labels as prerequisites for any future clinical consideration.

## 2. Disclaimer

**NOT FOR CLINICAL USE**

This proof-of-concept is a research exercise to evaluate the feasibility of automated bone tumor classification from radiographs. The model described in this report:

- Has not been validated in a prospective clinical setting
- Has not undergone regulatory review (FDA, CE marking, or equivalent)
- Produces unreliable predictions on data from imaging centers not represented in training
- Must not be used to inform any clinical decisions regarding patient diagnosis or treatment
- Is restricted to non-commercial use under the CC BY-NC-ND 4.0 dataset license

## 3. Methods

### 3.1 Dataset

The Bone Tumor X-ray Radiograph Dataset (BTXRD) is a publicly available collection of 3,746 plain radiograph images annotated for bone tumor classification, localization, and segmentation (Yao et al., Scientific Data, 2025).

**Dataset composition:**

| Characteristic | Value |
|---------------|-------|
| Total images | 3,746 |
| Image format | JPEG (224x224 after resize) |
| Source centers | 3 |
| Anatomical sites | 14 |
| Tumor subtypes | 9 |
| License | CC BY-NC-ND 4.0 |

**Class distribution:**

| Class | Count | Percentage |
|-------|-------|------------|
| Normal | 1,879 | 50.2% |
| Benign | 1,525 | 40.7% |
| Malignant | 342 | 9.1% |

**Center distribution:**

| Center | Source | Images | Percentage |
|--------|--------|--------|------------|
| Center 1 | Chinese hospitals | ~2,938 | 78% |
| Center 2 | Radiopaedia | ~561 | 15% |
| Center 3 | MedPix | ~247 | 7% |

The Malignant class is 5.5x underrepresented relative to Normal. Center 1 dominates the dataset, creating a risk of learning center-specific imaging characteristics rather than generalizable tumor features.

**Citation:** Yao, S., et al. (2025). A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors. *Scientific Data*. DOI: 10.1038/s41597-024-04311-y

### 3.2 Data Processing

**Preprocessing pipeline (applied to all images):**
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Applied with probability 1.0 to enhance radiograph contrast, improving visibility of bone structures and potential lesions.
2. **Resize:** All images resized to 224x224 pixels.
3. **ImageNet normalization:** Mean = [0.485, 0.456, 0.406], Std = [0.229, 0.224, 0.225].
4. **RGB conversion:** All images loaded as RGB via `Image.open().convert("RGB")` to handle both grayscale and color inputs uniformly.

**Training augmentation (applied only during training):**
- Horizontal flip (p=0.5)
- Random rotation (+/-15 degrees)

**Validation/test transforms:** Deterministic pipeline only (CLAHE, resize, normalize). No augmentation applied during evaluation.

### 3.3 Split Strategy

Two complementary split strategies were employed:

**Strategy 1: Stratified Split (70/15/15)**
- Image-level stratified random split preserving class proportions
- Train: 2,624 images | Validation: 558 images | Test: 564 images
- Purpose: Standard evaluation baseline with representative class distribution in each partition
- Limitation: Does not test cross-center generalization; potential same-lesion leakage across splits

**Strategy 2: Center-Holdout Split**
- Train/Validation: Center 1 only (~2,938 images; 85% train, 15% validation)
- Test: Centers 2 + 3 (808 images)
- Purpose: Proxy for external validation -- tests whether features learned from Center 1 transfer to unseen imaging sources
- Limitation: Training on a single center limits learned feature diversity

**Duplicate handling:** Perceptual hashing (phash, hash_size=8, distance=0) detected 21 exact duplicate image pairs. All members of a duplicate group are assigned to the same split partition to prevent train-test leakage from identical images.

**Known limitation:** The dataset lacks patient identifiers. Same-lesion multi-angle images (frontal, lateral, oblique views of the same lesion) cannot be grouped, creating a leakage risk for the stratified split.

### 3.4 Model Architecture

| Parameter | Value |
|-----------|-------|
| Backbone | EfficientNet-B0 (timm 1.0.15) |
| Total parameters | 4,011,391 |
| Pretrained weights | ImageNet |
| Fine-tuning | Full (all layers trainable) |
| Classifier head | Linear(1280, 3) with 0.2 dropout |
| Output | 3-class softmax (Normal, Benign, Malignant) |

Full fine-tuning was chosen over freezing the backbone because the BTXRD dataset (3,746 images) is small enough that the entire network benefits from adaptation to the radiograph domain, while being large enough to avoid severe overfitting with early stopping.

### 3.5 Training

| Hyperparameter | Value |
|---------------|-------|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Scheduler | Cosine annealing |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping patience | 7 (validation loss) |
| Loss function | Weighted cross-entropy |
| Class weights | Inverse-frequency (auto-computed) |
| Random seed | 42 |
| Deterministic mode | Enabled (torch.use_deterministic_algorithms) |

**Class imbalance handling:** Inverse-frequency class weights are applied to the cross-entropy loss during training, giving the Malignant class approximately 5.5x higher weight than Normal. Validation loss is computed without class weights to provide an unbiased stopping signal.

**Training was conducted separately for each split strategy**, producing two checkpoints: `best_stratified.pt` and `best_center.pt`.

### 3.6 Evaluation

**Metrics computed:**
- **Macro AUC** (one-vs-rest, averaged across classes)
- **Per-class AUC** (one-vs-rest)
- **Per-class average precision** (area under PR curve)
- **Per-class sensitivity** (recall = TP / (TP + FN))
- **Per-class specificity** (TN / (TN + FP))
- **Per-class precision** (TP / (TP + FP))
- **Per-class F1-score** (harmonic mean of precision and recall)
- **Overall accuracy**

**Bootstrap confidence intervals:** 95% CIs computed with 1,000 bootstrap iterations using the percentile method. Each iteration resamples the test set with replacement and recomputes all metrics. A class-presence guard ensures that bootstrap samples containing fewer than 2 members of any class are discarded.

## 4. Results

### 4.1 Stratified Split

The stratified split evaluates within-distribution performance on 564 test images (282 Normal, 231 Benign, 51 Malignant).

**Summary metrics:**

| Metric | Value | 95% CI |
|--------|-------|--------|
| Macro AUC | 0.846 | 0.814--0.873 |
| Accuracy | 67.9% | -- |

**Per-class performance:**

| Class | AUC | Sensitivity | 95% CI | Specificity | Precision | F1 | Support |
|-------|-----|-------------|--------|-------------|-----------|-----|---------|
| Normal | 0.843 | 60.6% | 54.6%--66.3% | 87.9% | 83.4% | 0.702 | 282 |
| Benign | 0.788 | 78.4% | 72.8%--83.6% | 62.5% | 59.2% | 0.674 | 231 |
| Malignant | 0.906 | 60.8% | 47.4%--74.3% | 95.7% | 58.5% | 0.596 | 51 |

**Per-class average precision:**

| Class | Average Precision |
|-------|-------------------|
| Normal | 0.837 |
| Benign | 0.707 |
| Malignant | 0.714 |

The model shows strong discrimination for the Malignant class (AUC 0.906) despite moderate sensitivity (60.8%). High Malignant specificity (95.7%) means few false alarms, but the 60.8% sensitivity means approximately 2 in 5 malignant cases are missed. The Benign class has the highest sensitivity (78.4%) but lowest specificity (62.5%), indicating the model tends to over-predict Benign.

**Figures:**
- ROC curves: `../results/stratified/roc_curves.png`
- PR curves: `../results/stratified/pr_curves.png`
- Confusion matrix: `../results/stratified/confusion_matrix.png`
- Normalized confusion matrix: `../results/stratified/confusion_matrix_normalized.png`
- Loss curve: `../results/stratified/loss_curve.png`

### 4.2 Center-Holdout Split

The center-holdout split evaluates generalization to unseen imaging sources on 808 test images (286 Normal, 415 Benign, 107 Malignant) from Centers 2 and 3.

**Summary metrics:**

| Metric | Value | 95% CI |
|--------|-------|--------|
| Macro AUC | 0.627 | 0.594--0.658 |
| Accuracy | 47.2% | -- |

**Per-class performance:**

| Class | AUC | Sensitivity | 95% CI | Specificity | Precision | F1 | Support |
|-------|-----|-------------|--------|-------------|-----------|-----|---------|
| Normal | 0.653 | 26.2% | 20.9%--32.0% | 89.5% | 57.7% | 0.361 | 286 |
| Benign | 0.574 | 64.3% | 59.9%--68.8% | 41.2% | 53.6% | 0.585 | 415 |
| Malignant | 0.653 | 36.4% | 27.0%--44.8% | 79.9% | 21.7% | 0.272 | 107 |

**Per-class average precision:**

| Class | Average Precision |
|-------|-------------------|
| Normal | 0.534 |
| Benign | 0.585 |
| Malignant | 0.255 |

Performance degrades substantially across all metrics. Normal sensitivity collapses to 26.2%, and Malignant sensitivity drops to 36.4%. The model fails to generalize Center 1 features to the imaging characteristics of Radiopaedia and MedPix sources.

**Figures:**
- ROC curves: `../results/center_holdout/roc_curves.png`
- PR curves: `../results/center_holdout/pr_curves.png`
- Confusion matrix: `../results/center_holdout/confusion_matrix.png`
- Normalized confusion matrix: `../results/center_holdout/confusion_matrix_normalized.png`
- Loss curve: `../results/center_holdout/loss_curve.png`

### 4.3 Comparison Table

Side-by-side comparison of the two split strategies:

| Metric | Stratified | Center-Holdout | Gap |
|--------|-----------|----------------|-----|
| **Macro AUC** | 0.846 (0.814--0.873) | 0.627 (0.594--0.658) | **-0.219** |
| **Accuracy** | 67.9% | 47.2% | **-20.8 pp** |
| **Malignant Sensitivity** | 60.8% (47.4%--74.3%) | 36.4% (27.0%--44.8%) | **-24.3 pp** |
| Malignant Specificity | 95.7% | 79.9% | -15.8 pp |
| Benign Sensitivity | 78.4% (72.8%--83.6%) | 64.3% (59.9%--68.8%) | -14.0 pp |
| Normal Sensitivity | 60.6% (54.6%--66.3%) | 26.2% (20.9%--32.0%) | -34.5 pp |
| Malignant AUC | 0.906 | 0.653 | -0.253 |
| Normal AUC | 0.843 | 0.653 | -0.190 |
| Benign AUC | 0.788 | 0.574 | -0.214 |

The generalization gap is consistent across all metrics and classes. The largest absolute drop is in Normal sensitivity (-34.5 pp), likely because Normal images from Centers 2 and 3 have different radiographic appearances than Center 1 normal images. The clinically critical Malignant sensitivity gap (-24.3 pp) means the model would miss nearly two-thirds of malignant cases on out-of-distribution data.

### 4.4 BTXRD Baseline Comparison

The original BTXRD paper (Yao et al., 2025) reported classification results using YOLOv8s-cls:

| Metric | BTXRD Paper (YOLOv8s-cls) | AssureXRay Stratified | AssureXRay Center-Holdout |
|--------|---------------------------|----------------------|--------------------------|
| Normal Precision | 91.3% | 83.4% | 57.7% |
| Benign Precision | 88.1% | 59.2% | 53.6% |
| Malignant Precision | 73.4% | 58.5% | 21.7% |
| Normal Recall | 89.8% | 60.6% | 26.2% |
| Benign Recall | 87.5% | 78.4% | 64.3% |
| Malignant Recall | 83.9% | 60.8% | 36.4% |

**Important caveats for this comparison** (all 7 documented):

1. **Random 80/20 split without patient-level grouping (potential data leakage).** The BTXRD paper used a simple random 80/20 split without grouping by patient, lesion, or center. Same-lesion multi-angle images may appear in both training and validation sets, inflating reported metrics.

2. **Validation set used for reporting (no separate held-out test set).** The paper reports performance on the validation partition, not a held-out test set. This may overestimate generalization performance.

3. **Different image size (600px vs 224px).** The BTXRD paper used 600x600 pixel images, while AssureXRay uses 224x224. Higher resolution may preserve fine-grained details important for classification.

4. **Different architecture (YOLOv8s-cls vs EfficientNet-B0).** The models have different capacities, inductive biases, and training dynamics. YOLOv8s-cls is a classification variant of an object detection architecture.

5. **Different training duration (300 epochs vs early stopping at ~5 epochs).** The BTXRD paper trained for 300 epochs; AssureXRay used early stopping which typically triggered within 5-10 epochs. Longer training may have allowed the BTXRD model to learn more complex features.

6. **Paper reports mAP@0.5 (detection metric) which is not directly comparable to AUC.** The BTXRD paper uses mean average precision at IoU 0.5 as its primary metric, which is a detection/localization metric, not directly comparable to classification AUC.

7. **No macro AUC or specificity reported in paper -- only precision/recall per class.** The BTXRD paper does not report AUC, specificity, or confidence intervals, limiting the metrics available for direct comparison.

Given these methodological differences, the BTXRD paper's higher numbers likely reflect a combination of data leakage (no patient-level split), higher resolution input, longer training, and a different evaluation protocol. A fair comparison would require identical splits, preprocessing, and evaluation metrics.

## 5. Explainability (Grad-CAM)

### 5.1 Method

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) was applied to visualize the spatial regions of each radiograph that most influenced the model's prediction. Grad-CAM was configured as follows:

| Parameter | Value |
|-----------|-------|
| Target layer | `model.bn2` (BatchNormAct2d, 1280 channels) |
| Target class | Predicted class (not ground truth) |
| Binarization threshold | 0.5 |
| Gallery examples | 3 per class (TP, FP, FN categories) |

Targeting the predicted class (rather than ground truth) shows the model's actual decision rationale -- what it was "looking at" when it made its prediction, whether correct or incorrect.

### 5.2 Qualitative Findings

Grad-CAM galleries were generated for all three classes across TP (true positive), FP (false positive), and FN (false negative) categories:

- **Correctly classified cases (TP):** Heatmaps generally highlight bone regions, but attention is often diffuse across the entire bone structure rather than concentrated on the tumor or lesion boundary.
- **False positives (FP):** Attention patterns are similar to TPs, suggesting the model uses the same features for both correct and incorrect predictions -- it lacks discriminative features to distinguish true from false positives.
- **False negatives (FN):** Heatmaps for missed malignant tumors often show attention outside the tumor region, indicating the model failed to detect relevant features in these cases.

**Gallery figures:**
- `../results/gradcam/gallery_Normal.png`
- `../results/gradcam/gallery_Benign.png`
- `../results/gradcam/gallery_Malignant.png`

### 5.3 Annotation Comparison

Expert tumor annotations (LabelMe format bounding boxes) were compared with Grad-CAM attention maps using Intersection over Union (IoU):

| Image | Class | Confidence | IoU |
|-------|-------|------------|-----|
| IMG000997.jpeg | Benign | 99.99% | 0.015 |
| IMG001295.jpeg | Benign | 99.99% | 0.003 |
| IMG001840.jpeg | Benign | 99.99% | 0.093 |
| IMG001796.jpeg | Benign | 99.99% | 0.237 |
| IMG000902.jpeg | Benign | 99.97% | 0.000 |

**Summary statistics:**
- Mean IoU: 0.070
- Images with IoU > 0.1: 1 out of 5 (20%)

**Interpretation:** The extremely low IoU (0.070 on a 0--1 scale) indicates that the model's attention regions have minimal overlap with expert-annotated tumor boundaries. Even for highly confident correct predictions (>99.9%), the model is not primarily attending to the tumor itself. This suggests the model may be relying on:

- **Contextual cues:** Bone shape, surrounding tissue patterns, or overall radiograph appearance
- **Center-specific artifacts:** Image quality characteristics, borders, or acquisition patterns specific to training centers
- **Global features:** Diffuse patterns across the entire image rather than focal tumor features

This finding is consistent with the poor center-holdout performance: if the model relies on center-specific contextual cues rather than tumor morphology, it will fail when those cues change.

**Annotation comparison figure:** `../results/gradcam/annotation_comparison.png`

## 6. Limitations

This proof-of-concept has 9 documented limitations that must be considered when interpreting results:

### 6.1 Same-Lesion Multi-Angle Image Leakage Risk

The BTXRD dataset lacks patient identifiers. Multiple images of the same lesion (frontal, lateral, and oblique views) may exist and could be split across training and test sets in the stratified strategy. Perceptual hashing detected 21 exact duplicate pairs, which were grouped to the same partition. However, near-duplicate multi-angle images of the same lesion are not detectable by hashing and represent an unmitigated leakage risk. This means stratified performance metrics (macro AUC 0.846) may be inflated by the model recognizing the same lesion from a different angle.

### 6.2 Label Noise Ceiling

Ground truth labels are based on radiologist interpretation of plain radiographs, not pathology confirmation. Radiographic diagnosis of bone tumors has inherent uncertainty -- some benign and malignant lesions appear similar on plain films. This creates a ceiling on achievable classification accuracy that cannot be exceeded without better ground truth. The true error rate of the labels is unknown.

### 6.3 Substantial Center Generalization Gap

The model's macro AUC drops from 0.846 (stratified) to 0.627 (center-holdout), a gap of -0.219. Malignant sensitivity drops from 60.8% to 36.4% (-24.3 pp). Accuracy drops from 67.9% to 47.2% (-20.8 pp). This demonstrates that the model has learned Center 1-specific imaging characteristics rather than generalizable tumor features. On out-of-distribution data, the model performs barely above chance for some classes.

### 6.4 Absence of Pathology-Confirmed Labels

None of the BTXRD labels are confirmed by histopathological examination. In clinical practice, bone tumor diagnosis requires biopsy and pathological analysis. Radiologist diagnosis alone, while often accurate, introduces label noise particularly for ambiguous cases. A clinical-grade evaluation would require pathology-confirmed ground truth.

### 6.5 Small Malignant Test Set

The Malignant class -- the most clinically important -- has only 51 samples in the stratified test set and 107 in the center-holdout test set. This small sample size limits statistical power for the class where performance matters most. Effect sizes that might be clinically significant could go undetected.

### 6.6 Wide Confidence Intervals

Due to the small Malignant test sample, the 95% confidence interval for Malignant sensitivity spans 27 percentage points on the stratified split (47.4%--74.3%). The true sensitivity could plausibly be anywhere in this range. On the center-holdout split, the CI spans 18 percentage points (27.0%--44.8%). These wide intervals mean point estimates should not be interpreted as precise measurements.

### 6.7 Low Grad-CAM IoU with Expert Annotations

Mean IoU between Grad-CAM attention maps and expert tumor annotations is 0.070. Only 1 of 5 evaluated images shows focal attention overlapping the annotated tumor region (IoU > 0.1). This indicates the model may not be attending to clinically relevant features. Even when predictions are correct, the model's reasoning pathway may not align with clinical reasoning, limiting interpretability and trust.

### 6.8 Single Architecture Evaluated

Only EfficientNet-B0 was trained and evaluated. No architecture comparison (ResNet, DenseNet, Vision Transformer), ensemble methods, or systematic hyperparameter optimization was performed. It is likely that different architectures or training configurations could achieve better performance. The results represent a single baseline, not an optimized solution.

### 6.9 Center 3 Normal Class Sparsity

Center 3 (MedPix) contributes only 27 Normal images to the center-holdout test set. This means Normal class performance metrics on Center 3 data are based on an extremely small sample and should be interpreted with caution. Per-center subgroup analysis was not performed due to this sparsity.

## 7. Clinical Relevance

### 7.1 Feasibility Demonstration

This proof-of-concept demonstrates that transfer learning with a standard convolutional neural network (EfficientNet-B0) can achieve moderate discrimination between Normal, Benign, and Malignant bone lesions on within-distribution radiograph data (macro AUC 0.846). This establishes a baseline for future bone tumor classification research on the BTXRD dataset.

### 7.2 Clinical Decision Framing

At the default operating point, the model achieves 60.8% sensitivity (95% CI: 47.4%--74.3%) for malignant tumors at 95.7% specificity on the stratified test set. On the center-holdout test set, malignant sensitivity drops to 36.4% (95% CI: 27.0%--44.8%) at 79.9% specificity.

**Interpretation for clinical context:**
- On within-distribution data: approximately 2 in 5 malignant tumors would be missed, but very few benign cases would be incorrectly flagged as malignant.
- On out-of-distribution data: approximately 2 in 3 malignant tumors would be missed, making the model clinically unacceptable.
- The 95% CI for stratified Malignant sensitivity (47.4%--74.3%) means the true miss rate could range from roughly 1 in 4 to 1 in 2 -- unacceptable variability for a screening tool.

### 7.3 Why This Model Must Not Be Used Clinically

1. **Insufficient sensitivity:** Missing 36--64% of malignant tumors is unacceptable for any clinical application.
2. **Poor generalization:** Performance degrades severely on out-of-distribution data, and real-world deployment would encounter diverse imaging equipment, protocols, and patient populations.
3. **Unexplainable attention:** Grad-CAM analysis shows the model is not reliably attending to tumor regions, meaning correct predictions may be coincidental rather than based on clinically meaningful features.
4. **Unverified labels:** No pathology confirmation means the model was trained on potentially noisy labels.
5. **No prospective validation:** All results are retrospective on a single dataset.

### 7.4 Requirements for Clinical Consideration

Before this approach could be considered for any clinical application, the following would be required:

- Patient-level data splits eliminating all forms of leakage
- Pathology-confirmed ground truth labels
- Multi-center prospective validation on diverse patient populations
- Substantially higher sensitivity (>90%) for malignant tumors
- Regulatory review and approval (FDA, CE marking)
- Integration into a clinical workflow as a decision-support tool (not standalone)
- Fairness evaluation across demographic subgroups
- Rigorous failure mode analysis with clinical oversight

## 8. Recommended Next Steps

1. **Obtain patient identifiers** for the BTXRD dataset (or a similar dataset) and implement true patient-level train/test splitting to eliminate same-lesion leakage risk and establish unbiased performance estimates.

2. **Acquire pathology-confirmed labels** for at least the test set. Histopathological confirmation would establish a reliable ground truth and enable meaningful evaluation of classification accuracy.

3. **Evaluate additional architectures** including ResNet-50, DenseNet-121, and Vision Transformers. Conduct systematic hyperparameter optimization (learning rate, augmentation strategy, image resolution) and evaluate ensemble approaches.

4. **Implement domain adaptation techniques** to address the center generalization gap. Options include domain-adversarial training, style transfer augmentation, or mixed-center training strategies.

5. **Expand the Malignant class** by collecting additional malignant tumor images from diverse sources. A larger Malignant sample would narrow confidence intervals and enable more reliable performance estimation.

6. **Perform subgroup analysis** by age, gender, anatomical site, and tumor subtype to identify populations where the model performs differently and potential fairness concerns.

7. **Investigate attention alignment** by exploring attention supervision methods (e.g., training with annotation-guided attention loss) to encourage the model to attend to clinically relevant regions.

## 9. References

1. Yao, S., et al. (2025). A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors. *Scientific Data*. DOI: 10.1038/s41597-024-04311-y. PMID: PMC11739492.

2. Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model Cards for Model Reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT*)*, 220--228.

3. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618--626.

4. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *Proceedings of the International Conference on Machine Learning (ICML)*, 6105--6114.

5. Wightman, R. (2019). PyTorch Image Models (timm). GitHub. https://github.com/huggingface/pytorch-image-models

## 10. Appendix

### A. Full Per-Class Metrics -- Stratified Split

| Class | Precision | Recall | F1-Score | Specificity | AUC | Avg Precision | Support |
|-------|-----------|--------|----------|-------------|-----|---------------|---------|
| Normal | 83.4% | 60.6% | 0.702 | 87.9% | 0.843 | 0.837 | 282 |
| Benign | 59.2% | 78.4% | 0.674 | 62.5% | 0.788 | 0.707 | 231 |
| Malignant | 58.5% | 60.8% | 0.596 | 95.7% | 0.906 | 0.714 | 51 |
| **Macro avg** | 67.0% | 66.6% | 0.658 | -- | **0.846** | -- | 564 |
| **Weighted avg** | 71.2% | 67.9% | 0.681 | -- | -- | -- | 564 |

### B. Full Per-Class Metrics -- Center-Holdout Split

| Class | Precision | Recall | F1-Score | Specificity | AUC | Avg Precision | Support |
|-------|-----------|--------|----------|-------------|-----|---------------|---------|
| Normal | 57.7% | 26.2% | 0.361 | 89.5% | 0.653 | 0.534 | 286 |
| Benign | 53.6% | 64.3% | 0.585 | 41.2% | 0.574 | 0.585 | 415 |
| Malignant | 21.7% | 36.4% | 0.272 | 79.9% | 0.653 | 0.255 | 107 |
| **Macro avg** | 44.3% | 42.3% | 0.406 | -- | **0.627** | -- | 808 |
| **Weighted avg** | 50.8% | 47.2% | 0.464 | -- | -- | -- | 808 |

### C. Bootstrap Confidence Intervals

**Stratified Split (1,000 iterations, percentile method):**

| Metric | Point Estimate | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
|--------|---------------|----------------|-------------- |-------------- |
| Macro AUC | 0.846 | 0.846 | 0.814 | 0.873 |
| Normal Sensitivity | 60.6% | 60.6% | 54.6% | 66.3% |
| Benign Sensitivity | 78.4% | 78.5% | 72.8% | 83.6% |
| Malignant Sensitivity | 60.8% | 60.8% | 47.4% | 74.3% |

**Center-Holdout Split (1,000 iterations, percentile method):**

| Metric | Point Estimate | Bootstrap Mean | 95% CI Lower | 95% CI Upper |
|--------|---------------|----------------|-------------- |-------------- |
| Macro AUC | 0.627 | 0.627 | 0.594 | 0.658 |
| Normal Sensitivity | 26.2% | 26.3% | 20.9% | 32.0% |
| Benign Sensitivity | 64.3% | 64.3% | 59.9% | 68.8% |
| Malignant Sensitivity | 36.4% | 36.4% | 27.0% | 44.8% |

### D. Figure References

| Figure | Path |
|--------|------|
| Stratified ROC curves | `../results/stratified/roc_curves.png` |
| Stratified PR curves | `../results/stratified/pr_curves.png` |
| Stratified confusion matrix | `../results/stratified/confusion_matrix.png` |
| Stratified normalized confusion matrix | `../results/stratified/confusion_matrix_normalized.png` |
| Stratified loss curve | `../results/stratified/loss_curve.png` |
| Center-holdout ROC curves | `../results/center_holdout/roc_curves.png` |
| Center-holdout PR curves | `../results/center_holdout/pr_curves.png` |
| Center-holdout confusion matrix | `../results/center_holdout/confusion_matrix.png` |
| Center-holdout normalized confusion matrix | `../results/center_holdout/confusion_matrix_normalized.png` |
| Center-holdout loss curve | `../results/center_holdout/loss_curve.png` |
| Grad-CAM gallery (Normal) | `../results/gradcam/gallery_Normal.png` |
| Grad-CAM gallery (Benign) | `../results/gradcam/gallery_Benign.png` |
| Grad-CAM gallery (Malignant) | `../results/gradcam/gallery_Malignant.png` |
| Grad-CAM annotation comparison | `../results/gradcam/annotation_comparison.png` |
