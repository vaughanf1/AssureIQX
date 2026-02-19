# Domain Pitfalls: Bone Tumor Radiograph Classification PoC

**Domain:** Medical image classification -- 3-class bone tumor classifier (Normal/Benign/Malignant)
**Dataset:** BTXRD (3,746 images, 3 centers, heavy class imbalance)
**Researched:** 2026-02-19
**Overall Confidence:** MEDIUM-HIGH (well-documented domain; specific BTXRD details from project context and paper metadata)

> **Note on sources:** WebSearch and WebFetch were unavailable during this research session. Findings are drawn from training knowledge of established peer-reviewed literature in medical imaging ML (Roberts et al. 2021 Nature Machine Intelligence; Oakden-Rayner et al. on hidden stratification; Varoquaux & Cheplygina 2022; Winkler et al. on shortcut learning in medical imaging). These pitfalls are well-established consensus in the field, not speculative. Confidence levels are noted per pitfall.

---

## Critical Pitfalls

Mistakes that would invalidate results, require a complete restart, or produce a clinically dangerous model.

---

### Pitfall 1: Data Leakage via Same-Lesion Multi-Angle Images

**Confidence:** HIGH
**What goes wrong:** The BTXRD dataset has no `patient_id` column. The same lesion is photographed at multiple angles (frontal, lateral, oblique). A random image-level split will place different views of the same lesion into both training and test sets. The model memorizes lesion-specific texture/shape patterns and reports inflated test metrics that do not reflect real-world generalization.

**Why it happens:** Standard `train_test_split` with stratification only balances class labels across splits -- it has no concept of grouped observations. The BTXRD paper's own baseline used an 80/20 random split with no patient-level grouping, so this mistake is baked into the published baseline itself.

**Consequences:**
- Test accuracy is overstated (potentially by 5-15 percentage points)
- Model appears to generalize but has actually memorized individual lesions
- Any downstream clinical claims are invalid
- Reviewers or collaborators who notice this will dismiss all results

**Prevention:**
1. **Proxy patient grouping:** Cluster images by metadata overlap (same `anatomical_site` + same `gender` + same `age` + same `center` + same `diagnosis`) to create proxy patient groups. Images sharing all these attributes are likely from the same patient and must stay in the same split.
2. **Use `GroupKFold` or manual grouped splitting** with proxy patient groups as the group key.
3. **Validate the grouping:** After splitting, audit whether any images in the test set have near-identical metadata profiles to training images.
4. **Report both:** Run evaluation on both the naive image-level split (for comparability with the paper's baseline) and the grouped split (for honest generalization). Report the difference as a leakage estimate.

**Detection (warning signs):**
- Test accuracy significantly exceeds the paper's baseline (91.3%/88.1%/73.4%) with a simpler model
- Very high accuracy with minimal training
- Grad-CAM highlights background/edge artifacts rather than tumor regions
- Large gap between grouped-split and image-split metrics

**Phase:** Must be addressed in data preparation (Phase 1-2). Cannot be retroactively fixed -- a leaked split invalidates all downstream experiments.

---

### Pitfall 2: Misleading Metrics from Class Imbalance

**Confidence:** HIGH
**What goes wrong:** With 1,879 normal / 1,525 benign / 342 malignant, a model that simply predicts "Normal" for everything achieves ~50% accuracy. A model that predicts "not Malignant" for everything achieves ~91% accuracy. Reporting overall accuracy or even macro-averaged metrics without per-class breakdown hides catastrophic failure on the clinically most important class (Malignant).

**Why it happens:** Teams default to accuracy as the primary metric because it is the simplest to compute and report. With 9.1% malignant prevalence, the base rate is already 90.9% for "not malignant." Weighted loss during training partially helps, but if evaluation metrics are not chosen carefully, the team declares success while the model fails exactly where it matters most.

**Consequences:**
- Model appears to work but misses most malignant cases (low sensitivity for Malignant class)
- The most clinically dangerous failure mode (false negative for malignancy) goes undetected
- Stakeholders make incorrect feasibility conclusions

**Prevention:**
1. **Primary metric: Per-class sensitivity (recall)**, especially Malignant sensitivity. A bone tumor classifier that misses malignancies is worse than useless.
2. **Report the full suite:** Per-class sensitivity, specificity, PPV, NPV. One-vs-rest ROC AUC and PR AUC per class. Confusion matrix.
3. **PR AUC over ROC AUC for Malignant class.** ROC AUC is optimistic under class imbalance because the True Negative rate dominates. PR AUC for the Malignant class gives a more honest view.
4. **Set a clinically meaningful threshold for Malignant sensitivity** before training -- e.g., "We require >= 80% sensitivity for Malignant at a specificity >= 70%." Evaluate against this, not just aggregate numbers.
5. **Report the "trivial baseline":** What do you get from a majority-class predictor and from a stratified random predictor? If your model barely beats these, the result is not meaningful.

**Detection (warning signs):**
- Team reports overall accuracy without per-class breakdown
- Malignant class F1 or sensitivity is below 60%
- ROC AUC looks good (>0.90) but PR AUC for Malignant is poor (<0.50)
- No confusion matrix in reports

**Phase:** Evaluation framework must be designed before training (Phase 2-3). Metrics should be specified in the experiment design, not chosen after seeing results.

---

### Pitfall 3: Center-Correlated Shortcuts (Spurious Correlations)

**Confidence:** HIGH
**What goes wrong:** The three source centers (Center 1: Chinese hospitals 78%; Center 2: Radiopaedia; Center 3: MedPix) have different imaging equipment, protocols, image resolutions, JPEG compression levels, and annotation borders. The class distribution also differs by center. The model learns to classify based on center-specific artifacts (scanner noise patterns, image borders, text overlays, aspect ratios) rather than tumor morphology.

**Why it happens:** Deep CNNs are universal function approximators -- they will learn whatever signal most easily reduces training loss. If center identity correlates with class labels (e.g., Center 2 has proportionally more benign cases), the model can achieve good in-distribution accuracy by detecting which center the image came from rather than looking at the tumor. This is the "shortcut learning" phenomenon documented extensively by Geirhos et al. (2020) and DeGrave et al. (2021) in medical imaging.

**Consequences:**
- Model fails catastrophically on data from a new center (or even a new scanner at an existing center)
- Grad-CAM reveals the model is looking at image borders, text overlays, or background noise
- Results are not clinically valid -- the model is a center detector, not a tumor detector

**Prevention:**
1. **Center holdout evaluation is mandatory** (already planned as secondary split). Train on Center 1 only, test on Centers 2+3. This directly measures generalization beyond center-specific artifacts.
2. **Image preprocessing to reduce center signals:**
   - Crop or mask any text overlays, borders, or annotations visible in raw images
   - Standardize image dimensions and aspect ratios
   - Apply consistent windowing/normalization
   - Consider whether CLAHE (contrast-limited adaptive histogram equalization) helps reduce scanner-specific contrast differences
3. **Grad-CAM audit:** After training, generate Grad-CAM overlays for correctly classified images from each center. If attention is on borders/background rather than anatomical regions, the model has learned shortcuts.
4. **Compare Grad-CAM with LabelMe annotations:** The dataset includes expert tumor region annotations. Quantify overlap between Grad-CAM activation and annotated tumor regions (e.g., intersection-over-union). Low overlap = model is not looking at the tumor.
5. **Data augmentation to break shortcuts:** Color jitter, random cropping, and random erasing can disrupt center-specific low-level patterns.

**Detection (warning signs):**
- Center holdout performance is dramatically worse than image-level split performance (>15% gap)
- Model accuracy correlates with center identity
- Grad-CAM highlights edges/corners/text rather than anatomy
- A simple linear probe on image statistics (mean pixel value, aspect ratio) can predict center with high accuracy

**Phase:** Must be addressed in data preparation (preprocessing) and validated in evaluation. The center holdout split is the primary detection mechanism.

---

### Pitfall 4: Label Noise Amplified on Rare Classes

**Confidence:** MEDIUM-HIGH
**What goes wrong:** The project context notes that "some diagnoses are based on radiologist opinion without pathology confirmation." In radiology, inter-observer agreement for bone tumor characterization on plain radiographs is moderate at best (kappa ~0.4-0.7 depending on the study and lesion type). With only 342 malignant samples, even a 5-10% label error rate means 17-34 mislabeled malignant images. For a class this small, this noise is devastating -- it is proportionally much more harmful than the same error rate on the 1,879 normal images.

**Why it happens:** Radiograph-only diagnosis of bone tumors is genuinely difficult. Many benign lesions have aggressive radiographic features (e.g., giant cell tumor), and some early malignancies look indolent. Without histopathological confirmation, labels reflect radiologist judgment, which has inherent uncertainty. Different centers may also apply different diagnostic criteria.

**Consequences:**
- The model's ceiling is bounded by label quality, not model capacity
- Malignant class performance hits a hard wall because the model is being trained on some incorrect labels
- Benign/Malignant confusion is partially real clinical ambiguity being treated as model failure
- Bootstrap confidence intervals on Malignant metrics will be wide, reflecting this noise

**Prevention:**
1. **Acknowledge the ceiling explicitly.** The model card and PoC report must state that label quality, not model architecture, is likely the primary performance bottleneck for the Malignant class.
2. **Do NOT chase marginal Malignant accuracy gains through hyperparameter tuning.** If Malignant sensitivity plateaus at 70-75%, that may reflect label noise, not a tuning problem.
3. **Analyze error patterns:** For misclassified Malignant cases, check whether they share characteristics (specific subtypes, specific centers, borderline age groups). Systematic errors suggest label issues; random errors suggest model limitations.
4. **Consider label smoothing** during training (e.g., soft labels 0.9/0.05/0.05 instead of hard 1/0/0) to make the model robust to label noise.
5. **Report confusion between Benign and Malignant separately** from confusion with Normal. Benign-Malignant confusion is clinically different from Normal-Malignant confusion.

**Detection (warning signs):**
- Malignant sensitivity plateaus well below benign/normal despite tuning
- High Benign-Malignant confusion that does not improve
- Misclassified Malignant cases cluster in specific subtypes (e.g., "other malignant" which is a catch-all)
- Inter-center disagreement on the same apparent lesion type

**Phase:** Must be understood during data audit (Phase 1) and acknowledged in evaluation/reporting (Phase 4-5). No code fix -- this is a data quality limitation to document.

---

### Pitfall 5: Grad-CAM Misinterpretation as Clinical Validation

**Confidence:** HIGH
**What goes wrong:** Teams generate Grad-CAM heatmaps showing that "the model looks at the right region" and conclude the model has learned clinically meaningful features. Grad-CAM is a post-hoc attribution method with significant limitations: it shows which regions influenced the final prediction, not whether the model understands tumor morphology. A model can look at the right region for the wrong reasons (e.g., detecting bone destruction artifacts rather than tumor cellularity patterns visible on radiographs).

**Why it happens:** There is strong pressure to demonstrate "explainability" in medical AI. Grad-CAM is the most accessible method. The temptation is to cherry-pick examples where Grad-CAM aligns with the tumor and present these as evidence of clinical validity.

**Consequences:**
- False confidence in model reliability
- Overclaiming in the PoC report
- Stakeholders believe the model "understands" tumors when it may not
- Missed opportunity to discover what the model actually learned

**Prevention:**
1. **Quantitative Grad-CAM evaluation, not cherry-picked examples:**
   - Compute IoU (intersection-over-union) between Grad-CAM activation (thresholded) and LabelMe tumor annotations across ALL correctly classified tumor images
   - Report the distribution of IoU scores, not just hand-picked good examples
   - Separately report IoU for TP, FP cases
2. **Show failure cases prominently.** Include Grad-CAM overlays for: (a) correct predictions with poor localization, (b) incorrect predictions with good localization, (c) all false negatives for Malignant class.
3. **Do not claim clinical validity from Grad-CAM alone.** The model card should state: "Grad-CAM highlights indicate regions of network attention, not clinical reasoning. Overlap with annotated tumor regions suggests (but does not prove) that the model uses relevant image features."
4. **Compare Grad-CAM for Normal images.** What does the model attend to when there is no tumor? If it attends to random bone regions, that is expected. If it attends to specific artifacts, that reveals shortcuts.
5. **Use multiple attribution methods** if time permits (Grad-CAM, Grad-CAM++, ScoreCAM) and check if they agree. Disagreement between methods reduces confidence in any single attribution.

**Detection (warning signs):**
- Report only shows 4-6 "best" Grad-CAM examples
- No quantitative overlap metrics with annotations
- Grad-CAM for Normal images shows suspiciously specific attention patterns
- All examples shown are from the same center

**Phase:** Must be designed into the evaluation framework (Phase 3-4). Quantitative Grad-CAM analysis requires having the LabelMe annotations parsed and aligned, which should happen in data preparation.

---

## Moderate Pitfalls

Mistakes that cause delays, inflated expectations, or technical debt.

---

### Pitfall 6: Preprocessing Inconsistencies Between Training and Inference

**Confidence:** HIGH
**What goes wrong:** Training pipeline applies a specific sequence of transforms (resize, normalization with ImageNet mean/std, augmentations) but the inference script or Streamlit demo applies a slightly different sequence. Common mismatches: different interpolation methods for resize, different normalization constants, augmentations accidentally left on during inference, or different image loading libraries (PIL vs OpenCV) that handle color channels differently.

**Prevention:**
1. **Single source of truth for transforms:** Define a `get_transforms(mode="train"|"val"|"test"|"inference")` function in one module. All scripts import from this function. Never duplicate transform definitions.
2. **Explicit interpolation specification:** Always specify `interpolation=InterpolationMode.BILINEAR` (or whichever method is chosen). Default varies between libraries and versions.
3. **PIL vs OpenCV consistency:** Pick one image loading library and use it everywhere. PIL loads as RGB, OpenCV loads as BGR. Mixing them silently corrupts the color channel order and breaks ImageNet-pretrained normalization.
4. **Test the inference pipeline:** Write a test that runs one training image through both the training pipeline (eval mode, no augmentation) and the inference pipeline and asserts the outputs are identical.
5. **Log normalization parameters in the model checkpoint/config.** Do not hardcode ImageNet mean/std in multiple files.

**Detection (warning signs):**
- Inference results are dramatically different from test set evaluation
- Streamlit demo predictions disagree with evaluation script predictions on the same images
- Model performs well on test set but poorly on "new" images that are actually from the same distribution

**Phase:** Architecture design (Phase 2) -- define the transform pipeline once. Validate in integration testing (Phase 4).

---

### Pitfall 7: EfficientNet-B0 Frozen Layer Strategy Mistakes

**Confidence:** MEDIUM
**What goes wrong:** Teams either freeze too many layers (underfitting -- the frozen ImageNet features do not transfer well to grayscale-like radiographs) or freeze too few layers (overfitting -- with only 3,746 images, fine-tuning a full EfficientNet-B0 from the earliest layers overfits quickly). Medical radiographs are visually very different from ImageNet; the lower convolutional layers (edge/texture detectors) transfer reasonably, but mid-level and higher-level features need adaptation.

**Prevention:**
1. **Phased unfreezing:** Start with only the classifier head trainable (2-5 epochs), then unfreeze the last 2-3 blocks of EfficientNet-B0 at a lower learning rate (10x reduction). This is well-established for medical imaging transfer learning.
2. **Monitor overfitting explicitly:** Track train vs. validation loss at every epoch. If training loss drops while validation loss increases after unfreezing, you unfroze too much too soon.
3. **Use aggressive regularization when fine-tuning:** Dropout (0.3-0.5 on the classifier head), weight decay (1e-4 to 1e-3), and augmentation.
4. **Grayscale handling:** Radiographs are grayscale but EfficientNet-B0 expects 3-channel input. Either replicate the single channel to 3 channels, or convert to pseudo-RGB. Do NOT train on true 1-channel input unless you modify the first conv layer, which breaks pretrained weight loading.

**Detection (warning signs):**
- Validation loss starts increasing within 5 epochs of unfreezing
- Large gap between training accuracy (~98%) and validation accuracy (~80%)
- Performance does not improve beyond the frozen-backbone baseline

**Phase:** Model training configuration (Phase 3).

---

### Pitfall 8: Ignoring Image Quality Heterogeneity

**Confidence:** MEDIUM-HIGH
**What goes wrong:** The three centers have different imaging equipment, protocols, and likely different image quality characteristics (resolution, noise levels, contrast, compression artifacts). Web-sourced images from Radiopaedia/MedPix (Centers 2 and 3) may have watermarks, annotations drawn on the image, overlaid text, cropped regions, or different aspect ratios than clinical PACS images from Center 1. Training on a mix without handling these differences causes the model to learn quality/source artifacts as predictive features.

**Prevention:**
1. **Data audit with per-center statistics:** Compute and report per-center image dimensions, aspect ratios, mean/std pixel values, file sizes, and JPEG quality estimates. Large differences confirm heterogeneity.
2. **Visual inspection of a random sample from each center** (10-20 images) looking specifically for: text overlays, annotation marks, borders, watermarks, cropping patterns.
3. **Preprocessing to normalize quality:**
   - Resize all images to a consistent resolution (224x224 for EfficientNet-B0 is standard, but consider 384x384 if detail matters)
   - Apply consistent contrast normalization
   - Consider ROI cropping using the LabelMe annotations for tumor images (though Normal images have no annotations)
4. **Document quality issues per center** in the data audit report. This informs interpretation of center holdout results.

**Detection (warning signs):**
- Per-center image statistics differ substantially
- Visual inspection reveals watermarks/annotations on web-sourced images
- Center holdout test performance is dramatically lower than primary split performance, even after accounting for distribution shift

**Phase:** Data audit and preprocessing (Phase 1-2).

---

### Pitfall 9: Overengineering the PoC

**Confidence:** HIGH
**What goes wrong:** The team spends weeks implementing focal loss with tunable gamma, multiple backbone architectures, ensemble methods, mixup/cutmix augmentation, test-time augmentation, self-supervised pretraining, and complex learning rate schedules -- before having a single working baseline. The PoC never ships because there is always "one more thing to try."

**Prevention:**
1. **Get a working end-to-end pipeline with the simplest possible approach first:**
   - EfficientNet-B0, frozen backbone, weighted cross-entropy, basic augmentation, 30 epochs
   - This should be runnable in a single afternoon
2. **Only add complexity when the simple version reveals a specific problem:**
   - Malignant recall is terrible? Try focal loss.
   - Overfitting? Add augmentation or freeze more layers.
   - Don't add solutions to problems you haven't observed yet.
3. **Time-box the project.** A PoC has a deadline. Define "done" as "we have a trained model, evaluation on both splits, Grad-CAM outputs, and a model card." Everything else is v2.
4. **Track experiments in a simple table** (CSV or markdown), not a complex MLflow setup. Experiment tracking infra is not the deliverable.

**Detection (warning signs):**
- More than 3 days spent before first training run completes
- Team is debating architecture choices without baseline results
- Requirements creep: adding segmentation, detection, or subtype classification

**Phase:** All phases -- this is a discipline issue, not a technical one. The roadmap should enforce "baseline first, then iterate."

---

### Pitfall 10: Incorrect Handling of Class Weights

**Confidence:** MEDIUM-HIGH
**What goes wrong:** Class weights are computed incorrectly, applied to the wrong loss function, or interact poorly with the learning rate. Common mistakes:
- Using `1/class_count` directly (too aggressive for extreme imbalance -- the Malignant weight would be ~5.5x Normal, which can destabilize training)
- Applying class weights to both the loss AND using oversampling (double correction)
- Using `WeightedRandomSampler` with replacement but forgetting that this means the model sees some Malignant images 5+ times per epoch (risking memorization of those specific images)

**Prevention:**
1. **Start with inverse-frequency weights scaled by total samples:** `w_c = N_total / (N_classes * N_c)`. This gives Normal=0.66, Benign=0.82, Malignant=3.65. These are moderate and stable.
2. **Do not combine class-weighted loss with oversampling** unless you deliberately compensate. Pick one strategy.
3. **If using `WeightedRandomSampler`:** Set the effective epoch length so that the minority class is seen ~2-3x, not 5-6x. Monitor for memorization by checking if Malignant training accuracy hits 100% while validation stays low.
4. **Focal loss as a principled alternative:** Focal loss (gamma=2.0, alpha from class frequencies) naturally downweights easy examples and upweights hard ones, partly addressing imbalance without explicit sampling manipulation. But start with weighted CE and only switch if it underperforms.
5. **Validate the weight calculation.** Print the class weights to the training log. It takes 2 minutes and catches bugs.

**Detection (warning signs):**
- Training loss oscillates wildly or diverges
- Malignant training accuracy hits 100% but Malignant validation accuracy is < 50%
- All predictions shift to Malignant (overcorrected weights)
- Loss values are suspiciously different from expected cross-entropy range

**Phase:** Model training (Phase 3).

---

## Minor Pitfalls

Mistakes that cause annoyance, confuse reporting, or waste time, but are recoverable.

---

### Pitfall 11: Seed Non-Determinism Across Runs

**Confidence:** MEDIUM
**What goes wrong:** The team sets `random.seed(42)` and `torch.manual_seed(42)` but gets different results across runs because CUDA operations, DataLoader workers, and data augmentation all have separate sources of randomness. The PoC report quotes specific numbers that cannot be reproduced.

**Prevention:**
1. Set all seeds in one place: `random.seed()`, `np.random.seed()`, `torch.manual_seed()`, `torch.cuda.manual_seed_all()`
2. Set `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` (trades some speed for reproducibility)
3. Set `num_workers=0` in DataLoader for reproducibility-critical runs (multi-worker data loading introduces non-determinism). Use `num_workers>0` for faster training but acknowledge the non-determinism.
4. **Use `generator` argument in PyTorch `DataLoader` and `WeightedRandomSampler`** with a fixed seed.
5. Accept that perfect determinism on GPU is not always possible and report results as mean +/- std over 3-5 runs with different seeds.

**Phase:** Training infrastructure (Phase 2-3).

---

### Pitfall 12: Forgetting to Evaluate on the Correct Image Set

**Confidence:** MEDIUM
**What goes wrong:** The evaluation script accidentally evaluates on the validation set instead of the test set, or uses the wrong split file, or includes images from the training set. This is embarrassingly common and hard to catch without explicit checks.

**Prevention:**
1. **Split files saved as CSVs with image IDs**, not directory structures. Verify by asserting no overlap between split files.
2. **Evaluation script takes split file path as an explicit argument**, never a hardcoded path.
3. **Log the number of images evaluated** and compare against expected test set size.
4. **Assert disjointness** at the start of evaluation: `assert len(set(train_ids) & set(test_ids)) == 0`

**Phase:** Evaluation framework (Phase 3-4).

---

### Pitfall 13: JPEG Compression Artifacts as Confounders

**Confidence:** LOW-MEDIUM
**What goes wrong:** JPEG compression creates artifacts at block boundaries (8x8 pixel blocks). Different centers may have saved images at different JPEG quality levels. Models can detect JPEG quality level from these artifacts, and if quality correlates with center (which correlates with class distribution), this becomes another shortcut feature.

**Prevention:**
1. **Check JPEG quality levels per center** during data audit (can be estimated from file size relative to image dimensions).
2. **Consider re-saving all images at a consistent JPEG quality** (e.g., quality=95) during preprocessing to normalize compression artifacts. This loses information but removes the confounder.
3. **Add JPEG compression as a training augmentation** (torchvision `RandomJPEGQuality` or albumentations `ImageCompression`) to make the model robust to compression variation.

**Phase:** Data audit (Phase 1) for detection; preprocessing (Phase 2) for mitigation.

---

### Pitfall 14: Not Saving Enough Checkpoints and Experiment Metadata

**Confidence:** MEDIUM
**What goes wrong:** Only the "best" checkpoint is saved. Later, the team wants to analyze learning dynamics, try different classification thresholds, or generate Grad-CAM for a different epoch -- and can't. Or: the team can't remember which hyperparameters produced which results.

**Prevention:**
1. **Save checkpoints at every N epochs** (e.g., every 5 epochs) plus the best by validation metric.
2. **Each checkpoint file includes:** model state_dict, optimizer state_dict, epoch number, validation metrics, and the full config dict.
3. **Experiment log table** (even just a markdown table in the repo): run ID, date, config hash, key hyperparameters, primary metrics.
4. **Save the split files alongside results** so any evaluation can be reproduced against exactly the same test set.

**Phase:** Training infrastructure (Phase 2-3).

---

## Phase-Specific Warnings

| Phase Topic | Likely Pitfall | Severity | Mitigation |
|---|---|---|---|
| Data download & audit | Images from web sources (Centers 2/3) have overlays/watermarks | Moderate | Visual inspection + per-center quality stats |
| Data splitting | Same-lesion images leak across splits | **Critical** | Proxy patient grouping before any splitting |
| Data splitting | Class stratification fails for Malignant with small N | Moderate | Use stratified grouped splitting; verify Malignant count in each fold |
| Preprocessing | Grayscale-to-RGB conversion done differently in train vs. inference | Moderate | Single transform definition function |
| Preprocessing | Resize interpolation inconsistency | Minor | Explicit interpolation parameter everywhere |
| Training | Class weight miscalculation or double-correction | Moderate | Print weights, pick ONE imbalance strategy |
| Training | Overfitting due to too-aggressive fine-tuning | Moderate | Phased unfreezing, monitor train-vs-val gap |
| Evaluation | Reporting accuracy instead of per-class sensitivity | **Critical** | Design metric suite before training |
| Evaluation | Evaluating on validation instead of test set | Moderate | Split file assertions, logged image counts |
| Grad-CAM | Cherry-picking good examples, no quantitative analysis | Moderate | IoU with LabelMe annotations across all images |
| Grad-CAM | Attribution to non-anatomical regions (shortcuts) | **Critical** | Center holdout Grad-CAM audit |
| Reporting | Overclaiming clinical validity from a PoC | Moderate | Explicit limitations section in model card |
| Reporting | Not comparing to trivial baselines | Minor | Always report majority-class and random baselines |

## BTXRD-Specific Warnings

These pitfalls are unique to this dataset and would not appear in generic medical imaging guidance.

| Issue | Details | Impact | Action |
|---|---|---|---|
| No patient_id | Cannot do true patient-level splitting | **Critical** | Build proxy groups from metadata; document limitation |
| Center 1 dominance (78%) | Model primarily learns Center 1 characteristics | High | Center holdout split exposes this; balance center representation in training if possible |
| Web-sourced images | Centers 2 and 3 are from Radiopaedia/MedPix, not clinical PACS | High | May have annotations, watermarks, different formatting; inspect and clean |
| Malignant subtypes are heterogeneous | 297 osteosarcoma + 45 "other malignant" -- very different radiographic appearances | Medium | Monitor per-subtype performance within Malignant class if subtype labels are available |
| "Other benign"/"other malignant" categories | Catch-all categories with heterogeneous appearance -- harder to learn | Medium | These samples may hurt more than help; consider whether to include or exclude them in sensitivity analysis |
| LabelMe annotations only for tumor images | Normal images have no region annotations, so Grad-CAM validation only possible for Benign/Malignant | Low | Document that Grad-CAM on Normal images cannot be validated against ground truth regions |
| Paper's baseline (YOLOv8s-cls) used random split | Published numbers have potential leakage -- cannot be used as a fair comparison target | Medium | Report comparison with caveats; your grouped-split numbers may look "worse" but are more honest |
| CC BY-NC-ND 4.0 license | Cannot modify or use commercially; cannot redistribute derivative datasets | Low | Ensure preprocessing does not create a "derivative dataset" for distribution; keep processed data local |

## Priority Summary

If the team addresses only five things from this document, it should be these:

1. **Proxy patient grouping before splitting** (Pitfall 1) -- without this, all metrics are meaningless
2. **Per-class sensitivity as primary metric, not accuracy** (Pitfall 2) -- without this, you cannot assess clinical viability
3. **Center holdout evaluation** (Pitfall 3) -- without this, you cannot claim generalization
4. **Quantitative Grad-CAM analysis with LabelMe annotations** (Pitfall 5) -- without this, explainability claims are unsubstantiated
5. **Single transform pipeline shared across train/eval/inference** (Pitfall 6) -- without this, evaluation and demo results will disagree

## Sources and Confidence Notes

| Finding | Primary Source | Confidence |
|---|---|---|
| Data leakage via patient grouping | Established medical imaging ML literature (Roberts et al. 2021; Varoquaux & Cheplygina 2022) + BTXRD project context confirming no patient_id | HIGH |
| Class imbalance metric pitfalls | Standard ML methodology (Saito & Rehmsmeier 2015 on PR curves; He & Garcia 2009) | HIGH |
| Center-correlated shortcuts | Geirhos et al. 2020 "Shortcut Learning"; DeGrave et al. 2021 (COVID-19 CXR shortcuts); Zech et al. 2018 (hospital label detection) | HIGH |
| Label noise on rare classes | Established pattern in medical imaging; specific BTXRD label quality from project context | MEDIUM-HIGH |
| Grad-CAM limitations | Adebayo et al. 2018 "Sanity Checks for Saliency Maps"; Arun et al. 2021 on attribution in medical imaging | HIGH |
| EfficientNet transfer to radiographs | Broadly established; specific freeze/unfreeze strategy is community best practice | MEDIUM |
| JPEG compression artifacts | Documented in forensic imaging literature; relevance to medical imaging is emerging | LOW-MEDIUM |
| BTXRD-specific issues | Derived directly from project context and paper metadata | MEDIUM-HIGH |

> **Gaps:** WebSearch and WebFetch were unavailable. The following could not be verified against the latest sources:
> - Whether the BTXRD paper's GitHub repo has been updated with improved splitting methodology since publication
> - Whether new papers have been published using BTXRD with better baselines
> - Current best practices for EfficientNet-B0 fine-tuning in PyTorch (specific API changes)
> - Any dataset errata or known labeling corrections published by the BTXRD authors
>
> **Recommendation:** Before starting implementation, manually check the BTXRD GitHub repo (https://github.com/SHUNHANYAO/BTXRD) for updates, and search for recent papers citing the BTXRD dataset.
