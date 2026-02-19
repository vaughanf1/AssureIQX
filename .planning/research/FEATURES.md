# Feature Landscape

**Domain:** Medical image classification PoC -- 3-class bone tumor radiograph classification (Normal/Benign/Malignant)
**Researched:** 2026-02-19
**Overall confidence:** HIGH (medical imaging classification is a mature, well-documented field with stable conventions established through MICCAI, RSNA, SPIRIT-AI/CONSORT-AI guidelines, and FDA pre-submission guidance)

---

## Table Stakes

Features users expect. Missing = PoC is incomplete, not publishable, not credible to clinical or ML reviewers.

### Data Pipeline

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Automated dataset download and organization | Reproducibility requires anyone to recreate from scratch | Low | Script to fetch from figshare, extract, organize into `data_raw/` |
| Data audit report | Cannot trust a model without understanding the data it trained on; standard in clinical ML | Medium | Class distribution, image dimensions histogram, missing values, annotation coverage, duplicate detection, per-center breakdown |
| Dataset specification document | Reviewers and collaborators need to understand what columns mean, label derivation logic, provenance | Low | Columns, label schema, data source description, license note |
| Dual split strategy with stratification | Standard practice to avoid biased evaluation; center holdout specifically tests generalization -- critical for multi-center medical data | Medium | Primary: image-level stratified 70/15/15. Secondary: center holdout (Center 1 train/val, Centers 2+3 test). Both must use stratification on class labels |
| Leakage risk documentation | BTXRD has no patient_id -- same lesion may appear in multiple views. Not disclosing this is a scientific integrity issue | Low | Explicit warning in reports. Note: cannot fix without patient IDs, but must acknowledge |
| Reproducible random seeds | Any ML PoC that cannot be reproduced is not credible | Low | Fixed seeds for all random operations (split, augmentation, weight init, dataloader shuffle) |

### Model Training

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Transfer learning from ImageNet-pretrained backbone | Training from scratch on ~3.7K images would be negligent; transfer learning is the universal standard for small medical imaging datasets | Low | EfficientNet-B0 primary. Freeze-then-unfreeze or full fine-tune are both acceptable |
| Class imbalance handling | Malignant (342) vs Normal (1,879) is 5.5:1 ratio. Ignoring this guarantees poor malignant sensitivity -- the most clinically important class | Low | Inverse-frequency weighted cross-entropy loss. Simple, proven, sufficient for baseline |
| Standard augmentation pipeline | Medical image augmentation is expected to prevent overfitting on small datasets | Low | Resize to 224x224, normalize to ImageNet stats, horizontal flip, small rotation (10-15 deg). Nothing exotic needed |
| Configurable hyperparameters via config file or CLI args | Reproducibility and experimentation require externalized configuration, not hardcoded values | Low | Learning rate, batch size, epochs, patience, backbone, loss type at minimum |
| Early stopping with patience | Without early stopping, model either overfits or requires manual monitoring | Low | Monitor validation loss, patience of 5-10 epochs |
| Checkpoint saving (best model) | Must save best model for evaluation and inference; losing trained weights defeats the purpose | Low | Save best val_loss model and optionally last model |
| Training log / loss curves | Reviewers need to see convergence behavior; also needed for debugging | Low | CSV log of train/val loss per epoch, plus a plot (matplotlib) |

### Evaluation

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| ROC AUC (one-vs-rest, macro, per-class) | The standard discriminative metric for medical classifiers. Expected by every clinical ML reviewer | Medium | Per-class OvR ROC curves plotted, macro-averaged AUC reported. Use sklearn |
| PR AUC (per-class) | Especially important for the imbalanced malignant class where ROC can be misleadingly optimistic | Medium | Per-class precision-recall curves plotted, average precision reported |
| Per-class sensitivity (recall) and specificity | Clinicians think in sensitivity/specificity, not accuracy. Malignant sensitivity is THE critical number | Low | From confusion matrix. Must report per-class, not just macro |
| Confusion matrix (absolute counts and normalized) | The single most intuitive evaluation artifact for any classifier | Low | Heatmap plot. Both raw counts and row-normalized (recall-normalized) versions |
| Classification report (precision, recall, F1 per class) | Standard sklearn output. Omitting it looks lazy | Low | Text table + saved to file |
| Evaluation on BOTH split strategies | The dual split exists to answer different questions. Evaluating on only one wastes the secondary split | Low | Same metrics computed for both. Comparison table in report |
| Operating point analysis (threshold selection) | Default 0.5 threshold is almost never optimal for imbalanced medical data. Must at least show what happens at different thresholds | Medium | Show sensitivity/specificity trade-off at various thresholds for the malignant class specifically |
| Statistical summary with confidence context | Point estimates without any uncertainty context are insufficient for medical claims | Medium | At minimum, report dataset sizes so readers can gauge power. Bootstrap 95% CIs on AUC and sensitivity are ideal but can be "optional" -- see Differentiators |

### Explainability

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Grad-CAM heatmap generation | The minimum viable explainability for a medical imaging classifier. Reviewers and clinicians expect to see WHERE the model is looking | Medium | Generate for last conv layer of backbone. Overlay on original image with alpha blending |
| Curated example heatmaps (TP, FP, FN per class) | Cherry-picked examples are dishonest; systematic examples (TP/FP/FN) show real behavior | Medium | Select 3-5 examples per category (TP Normal, TP Benign, TP Malignant, FP each, FN each). Save as grid or individual images |
| Heatmap comparison against tumor annotations | BTXRD provides LabelMe bounding boxes/masks. Not comparing Grad-CAM attention to expert annotations is a missed opportunity that reviewers WILL ask about | Medium | Overlay Grad-CAM on annotated region. Qualitative assessment: "does the model attend to the tumor?" IoU or region overlap is a differentiator (see below) |

### Inference

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Single-image inference script | Must be able to run the model on a new image from CLI without loading the full training pipeline | Low | `python inference.py --image path/to/img.jpg --checkpoint path/to/model.pt` outputs class prediction, confidence scores, and Grad-CAM overlay |
| Prediction output with confidence scores | Outputting just the class label without probabilities is insufficient for medical context | Low | Softmax probabilities for all 3 classes |

### Documentation and Reporting

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| Model card | Industry standard for responsible ML. Documents what the model is, what it was trained on, how it performs, and its limitations. Mitchell et al. (2019) format is expected | Medium | Architecture, training data, intended use, performance metrics, ethical considerations, limitations, caveats |
| PoC report | The deliverable that communicates findings to stakeholders. Without it, the PoC is just code with no conclusions | Medium | Executive summary, methods, results, clinical relevance discussion, limitations, next steps |
| Limitations section (explicit, honest) | Medical AI that does not explicitly state its limitations is irresponsible. Reviewers check for this first | Low | No patient_id (leakage risk), small malignant class, single dataset, no external validation, no pathology ground truth for some cases, CC BY-NC-ND license |
| Requirements file / environment specification | Cannot reproduce without knowing exact dependencies | Low | `requirements.txt` or `pyproject.toml` with pinned versions |
| README with setup and run instructions | Standard for any code deliverable | Low | Clone, install, download data, train, evaluate, inference steps |

---

## Differentiators

Features that set this PoC apart from a basic homework exercise. Not expected, but valued. These strengthen the PoC's credibility and demonstrate thoroughness.

### Evaluation Enhancements

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Bootstrap confidence intervals (95% CIs on AUC, sensitivity, specificity) | Transforms point estimates into statistically responsible claims. Especially valuable with N=342 malignant | Medium | 1000-2000 bootstrap iterations. Report as "AUC: 0.85 [0.79-0.90]" |
| Calibration analysis (reliability diagram + ECE) | Shows whether predicted probabilities are trustworthy -- critical for clinical decision support | Medium | Reliability diagram, Expected Calibration Error (ECE). If poorly calibrated, note that temperature scaling could help |
| Subgroup fairness analysis (performance by age group, gender, anatomical site) | Demonstrates awareness of bias -- increasingly expected in medical AI | Medium | Break down sensitivity/specificity by demographic and anatomical subgroups. Flag any large disparities |
| Cross-center performance comparison | Beyond just center holdout, showing per-center performance highlights domain shift | Low | Table: metrics for Center 1 vs Center 2 vs Center 3. If Center 2/3 performance drops significantly, this is a finding worth reporting |
| Cohen's kappa or inter-rater-equivalent metric | Contextualizes model performance against known human variability | Low | Report alongside other metrics for additional context |

### Explainability Enhancements

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Quantitative Grad-CAM vs annotation overlap (IoU / pointing game) | Goes beyond qualitative "looks right" to measurable alignment between model attention and expert-labeled tumor regions | Medium | Compute IoU between thresholded Grad-CAM and LabelMe annotation mask. Report mean IoU per class. This is a strong differentiator because most PoCs only show cherry-picked heatmaps |
| Failure case analysis | Systematic examination of misclassifications: what do FPs and FNs have in common? (e.g., small tumors, certain anatomy, certain center) | Medium | Cluster or tabulate misclassifications by metadata features. "Malignant FNs are disproportionately from lower limb oblique views" is a valuable finding |
| Multiple explainability methods (Grad-CAM + Grad-CAM++) | Shows robustness of attention patterns across methods | Low | Grad-CAM++ is a small addition on top of Grad-CAM. Side-by-side comparison adds credibility |

### Clinical Framing

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Clinical decision context in report | Framing results in terms of clinical workflow: "At the high-sensitivity operating point, the model catches 95% of malignant cases at the cost of 30% false positive rate" | Low | No code, just thoughtful writing in the PoC report. Highly valued by clinical reviewers |
| Comparison table against paper's baseline | BTXRD paper reports YOLOv8s-cls results. Comparing against their numbers shows where this approach stands | Low | Table comparing this PoC vs paper's results on equivalent metrics |
| Sensitivity at fixed specificity (and vice versa) | Clinical practice often has a fixed acceptable false positive rate. Reporting "sensitivity at 90% specificity" is more actionable than AUC alone | Low | Interpolate from ROC curve. Report for malignant class specifically |

### Interactive Demo

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Streamlit demo app | Makes the PoC tangible for non-technical stakeholders. Upload image, see prediction + heatmap | Medium | Single page: file upload, prediction display with confidence bars, Grad-CAM overlay. Must include disclaimer about non-clinical use |

### Reproducibility Extras

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| Makefile or task runner | One-command pipeline: `make all` runs download, audit, split, train, evaluate, report | Low | Ties all scripts together. Impressive for reproducibility |
| Experiment tracking (W&B or MLflow, lightweight) | Shows ML engineering maturity. Logs hyperparams, metrics, artifacts | Medium | Only if it does not add heavy dependencies. W&B has a free tier. Could also use simple CSV logging as a lighter alternative |
| Docker or conda environment file | Guarantees environment reproducibility across machines | Low | `environment.yml` or `Dockerfile`. Not a container orchestration system, just a single reproducible env |

---

## Anti-Features

Features to explicitly NOT build. Common mistakes in medical image classification PoCs that waste time, add complexity, or create false confidence.

| Anti-Feature | Why Avoid | What to Do Instead |
|--------------|-----------|-------------------|
| **Hyperparameter optimization / AutoML (Optuna, Ray Tune)** | PoC goal is feasibility, not squeezing out 0.5% AUC. HPO on 3.7K images risks overfitting to the validation set. Wastes compute time and adds complexity with marginal benefit | Manual tuning of learning rate and batch size is sufficient. Document what was tried |
| **Ensemble of multiple architectures** | Ensembles complicate inference, explainability (whose Grad-CAM?), and reproducibility. They obscure whether the base model works | Use one backbone (EfficientNet-B0). If results are poor, try ResNet50 as a second experiment, but do not ensemble |
| **Custom loss functions (triplet loss, contrastive learning, etc.)** | Exotic losses add complexity without proven benefit at PoC scale. Weighted CE is well-understood and sufficient | Weighted cross-entropy. If needed, focal loss as one step up. Nothing more exotic |
| **Multi-task learning (classification + detection + segmentation)** | BTXRD supports it, but a PoC should prove one thing well. Multi-task multiplies evaluation complexity and debugging surface | Classification only. Note in "next steps" that annotations exist for future detection/segmentation work |
| **9-subtype classification** | Only 342 malignant images split across subtypes yields some classes with <50 samples. Results will be statistically meaningless and misleadingly poor | 3-class (Normal/Benign/Malignant) only. Mention subtype classification as future work with larger data |
| **Attention mechanisms / custom architecture modifications** | Adds complexity to the backbone with marginal benefit at PoC scale. Pretrained ImageNet features with fine-tuning are the proven approach | Use the pretrained backbone as-is. Fine-tune fully or with freeze schedule. No architectural surgery |
| **Test-time augmentation (TTA)** | Complicates inference pipeline for marginal accuracy gains. Not appropriate for a baseline PoC | Report single-pass inference results. Mention TTA as a potential improvement in next steps |
| **Production API / REST endpoint / containerized serving** | This is a PoC, not a deployment. Building serving infrastructure is premature optimization | Inference script that runs from CLI. Streamlit demo (optional) is the furthest to go |
| **DICOM handling / PACS integration** | BTXRD provides JPEGs. Adding DICOM pipeline adds complexity with zero benefit for this dataset | Work with JPEGs as provided. Note DICOM support as a production requirement in next steps |
| **Federated learning or privacy-preserving training** | Completely out of scope for a single-dataset PoC | Note as a consideration for multi-institution production work |
| **Complex preprocessing (CLAHE, bone segmentation, windowing)** | Standard ImageNet preprocessing (resize, normalize) is sufficient for transfer learning from pretrained models. Domain-specific preprocessing adds variables without proven benefit at baseline stage | Resize to 224x224, ImageNet normalization, standard augmentations only. If results are poor, preprocessing experiments can be a documented next step |
| **Web scraping or external data collection** | License, ethics, and scope issues. BTXRD is sufficient and well-characterized for a PoC | Use BTXRD only. Mention external validation as future work |
| **Interpretability methods beyond Grad-CAM family** | SHAP on images is computationally expensive and often uninformative. LIME is noisy. Grad-CAM/Grad-CAM++ is the standard and sufficient | Grad-CAM is the primary method. Grad-CAM++ as a lightweight addition if time permits. Nothing more |
| **Patient-level grouping (without patient IDs)** | The dataset lacks patient IDs. Attempting to infer groupings (e.g., by visual similarity or metadata heuristics) introduces unreliable assumptions. Better to acknowledge the limitation honestly | Document the leakage risk clearly. Do not fabricate patient groupings |
| **Overdesigned configuration systems (Hydra, complex YAML hierarchies)** | A PoC with 10-15 hyperparameters does not need a configuration framework. argparse or a simple YAML/JSON config is sufficient | Single config file (YAML or JSON) loaded at script start. CLI overrides via argparse for key parameters |
| **Extensive data cleaning / outlier removal** | For a PoC, the dataset should be used as-is to establish baseline. Aggressive cleaning without clinical rationale risks cherry-picking | Audit data quality, flag issues in the audit report, but train on the full provided dataset. Document any excluded images with reasons |

---

## Feature Dependencies

```
Download Script
    |
    v
Data Audit Report  +  Dataset Spec Document
    |
    v
Dual Split Strategy (requires audit to inform stratification)
    |
    v
PyTorch Dataset Loader + Augmentation Pipeline
    |
    v
Training Script (requires loader, config, loss function)
    |                               |
    v                               v
Checkpoint (best model)        Training Log / Loss Curves
    |
    +---------------------------+---------------------------+
    |                           |                           |
    v                           v                           v
Evaluation Script          Grad-CAM Generation         Inference Script
    |                           |
    v                           v
Metrics + Plots           Heatmap Gallery
    |                      (requires annotations
    |                       for comparison)
    v
Bootstrap CIs (optional, extends evaluation)
    |
    +-------------------+
    |                   |
    v                   v
Model Card          PoC Report
                        |
                        v
                  Streamlit Demo (optional, requires inference + Grad-CAM)
```

**Critical path:** Download -> Audit -> Split -> Loader -> Train -> Evaluate -> Report

**Parallel after training:**
- Evaluation and Grad-CAM generation can run in parallel (both need only the saved checkpoint)
- Model card can be drafted in parallel with evaluation (filled in with final metrics later)
- Inference script can be built in parallel with evaluation

**Streamlit demo depends on:** Inference script + Grad-CAM generation both complete

---

## MVP Recommendation

For a credible, publishable PoC, prioritize in this order:

### Must Complete (Table Stakes)

1. **Data pipeline** (download, audit, spec, splits) -- Foundation everything else depends on
2. **Training pipeline** (loader, augmentation, weighted loss, early stopping, checkpoints) -- Core model
3. **Full evaluation suite** (ROC AUC, PR AUC, sensitivity/specificity, confusion matrix, both splits) -- Proves the model works
4. **Grad-CAM with annotation comparison** (qualitative) -- Proves the model looks at the right things
5. **Inference script** -- Makes the model usable
6. **Model card + PoC report + limitations** -- Makes the work communicable and responsible
7. **README + requirements.txt** -- Makes the work reproducible

### High-Value Differentiators to Include If Time Permits

1. **Bootstrap CIs** (Low incremental effort, HIGH credibility boost) -- 30 lines of code, transforms statistical credibility
2. **Comparison against paper's baseline** (Low effort) -- Already have numbers from the paper
3. **Failure case analysis** (Medium effort, very insightful) -- Often reveals the most interesting findings
4. **Clinical decision framing in report** (Low effort, high impact) -- Writing, not code
5. **Streamlit demo** (Medium effort) -- Makes PoC tangible for non-technical audience

### Defer to Post-PoC

- Subgroup fairness analysis: Valuable but requires careful statistical handling with small subgroups
- Calibration analysis: Important for production, not critical for baseline feasibility
- Experiment tracking (W&B/MLflow): Nice to have, not essential for a single-model PoC
- Docker environment: Only needed if sharing across different machines
- Quantitative Grad-CAM IoU: Interesting research direction but not needed for feasibility assessment

---

## Confidence Notes

| Area | Confidence | Rationale |
|------|------------|-----------|
| Table stakes features | HIGH | Medical imaging classification PoC requirements are well-established in literature (SPIRIT-AI, CONSORT-AI, FDA guidance, MICCAI community standards). These conventions have been stable for 3+ years |
| Evaluation metrics | HIGH | ROC AUC, PR AUC, sensitivity/specificity, confusion matrix are universally expected. This is not controversial |
| Explainability requirements | HIGH | Grad-CAM is the de facto standard for CNN explainability in medical imaging. Established by Selvaraju et al. (2017), widely adopted |
| Anti-features | HIGH | Based on extensive pattern of medical imaging PoCs that fail by overengineering. Common failure modes are well-documented in ML retrospectives |
| Differentiator value | MEDIUM | Relative value of differentiators depends on audience (clinical reviewers vs ML reviewers vs stakeholders). Rankings reflect general consensus but specific priorities may vary |
| Complexity estimates | MEDIUM | "Low/Medium/High" ratings assume familiarity with PyTorch and standard ML tooling. May vary based on developer experience |

---

## Sources

- Mitchell, M., et al. "Model Cards for Model Reporting." FAT* 2019 -- Established model card format
- Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks." ICCV 2017 -- Grad-CAM method
- SPIRIT-AI and CONSORT-AI guidelines (Lancet Digital Health, 2020) -- Reporting standards for clinical AI
- FDA "Artificial Intelligence and Machine Learning in Software as a Medical Device" guidance -- Regulatory expectations
- Yao et al. "A Radiograph Dataset for the Classification, Localization, and Segmentation of Primary Bone Tumors." Scientific Data, 2025 -- BTXRD dataset paper and baseline results
- Rajpurkar, P., et al. "AI in Health and Medicine." Nature Medicine, 2022 -- Review of medical AI evaluation standards
- Varoquaux, G. and Cheplygina, V. "Machine learning for medical imaging: methodological failures and recommendations." npj Digital Medicine, 2022 -- Common pitfalls in medical imaging ML

**Note:** WebSearch and WebFetch were unavailable during this research session. All findings are based on domain expertise from training data (through May 2025). The conventions for medical image classification PoCs are mature and stable; LOW risk of material changes since training cutoff. Feature recommendations are cross-validated against the specific BTXRD dataset paper's context and the project's stated requirements.
