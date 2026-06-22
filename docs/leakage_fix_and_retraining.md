# Data Leakage detection, pipeline fix and full retraining

## Key observation (critical issue identified)

- **CONFIRMED: training/evaluation data leakage in grading pipeline**
  All 27 XAI evaluation images (from `IDRiD_055`to`IDRiD_081`) were found inside `grading/images/train/` and their labels existed in `grading/labels/train.csv`

- These images were unintentionally included in training for both:
  - EfficientNet-B4
  - ResNet-50

- **Root cause:**
  The previous dataset split logic only considered segmentation split boundaries (to avoid leakage within segmentation tasks). However, the grading dataset is an independent subset of IDRiD, and overlap between grading-train and segmentation-test (55–81) was not accounted for

- This resulted in unintended exposure of evaluation images during training, invalidating previous Week 1/Week 2 performance estimates

---

## Decision: full retraining

We evaluated three options:

### Option 1: report leakage as limitation
we rejected this because it compromises the core requirement of the project:
- CFS (Counterfactual/Feature-based explanation system) requires faithful model generalization
- Leakage violates interpretability validity

### Option 2: switch to grading test split
we rejected because:
- no corresponding lesion masks exist for required XAI evaluation
- incompatible with segmentation-based fidelity metrics (IoU/Dice)

### Option 3: retrain models with corrected dataset (final decision)
- removed all leaked IDs (`IDRiD_055`to`IDRiD_081`) from `grading/labels/train.csv`
- rebuilt `train_df` ensuring strict exclusion of XAI evaluation set
- verified zero overlap before training

---

## Corrected pipeline

- Clean dataset constructed with explicit exclusion list:
  - `EXCLUDED_IDS = IDRiD_055–IDRiD_081`
- Verified:
  - 27/27 leaked samples detected
  - removed successfully before training
  - final train set reduced accordingly

- Models retrained from scratch:
  - EfficientNet-B4
  - ResNet-50

- Training reset ensured:
  - no checkpoint reuse from leaked models
  - fresh optimizer states
  - deterministic seed fixed

---

## Canonical inference pipeline (post-retrain standardization)

To eliminate inconsistencies across team runs:

### Standardization fixes applied:
1. **Unified image ID format**
   - Standardized to `IDRiD_055` (3-digit format for grading consistency)
   - resolved mismatch across:
     - segmentation filenames
     - JSON test list
     - grading labels

2. **Single source of truth for predictions**
   - Introduced canonical `predictions.csv`
   - Generated only via one inference notebook
   - No duplicate regeneration by individual contributors

3. **Deterministic inference enforced**
   - `model.eval()` explicitly used
   - fixed seed for reproducibility
   - single controlled execution pipeline

---

## Next steps

1. Complete retraining of EfficientNet-B4 and ResNet-50 on cleaned dataset
2. Validate performance metrics (Kappa, F1) post-retrain
3. Run canonical inference pipeline to generate single `predictions.csv`
4. Re-run all XAI modules (GradCAM, LIME, SHAP) on corrected models
5. Finalize evaluation scripts:
   - `fidelity_scoring.py`
   - `eval/stratify.py`
6. Proceed with CFS integration using leakage-free model outputs