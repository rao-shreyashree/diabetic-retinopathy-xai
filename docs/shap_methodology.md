# SHAP Methodology —  Generation

## Method choice: GradientSHAP (not KernelSHAP)

KernelSHAP was considered but rejected. It approximates Shapley values via
many perturbed forward passes per image and is not designed for high-resolution
CNN inputs — this would have been too slow and unstable for 27 test images
across 2 models. GradientSHAP instead uses gradients (similar to GradCAM) combined
with noise and interpolation from a baseline distribution, giving Shapley-style
attribution that is far faster and more stable on deep networks. Implemented via
`captum.attr.GradientShap`.

## Background distribution: Grade-0 (no DR) training images, fixed sample

A black/zero baseline was rejected as physiologically meaningless for a fundus
photograph — black pixels in a fundus image simply represent the area outside
the circular retinal field of view, not "absence of disease." Instead, the
background distribution is a **fixed random sample of Grade-0 (no DR) images
from the training set** (seed=42). This means SHAP attribution answers the
question: *"what does this pixel contribute relative to healthy retinal tissue?"*
— a clinically interpretable framing that ties directly into the thesis's
lesion-localization goal.

The background is sampled **once** (fixed, not resampled per image) and reused
identically across all 27 test images and both models, for consistency and
stability — standard practice for GradientSHAP.

## Hyperparameter tuning: n_samples and background size

Initial defaults (`n_samples=50`, `background=20`) caused kernel crashes due to
memory exhaustion on the development machine (8GB RAM, CPU-only, no GPU).

Diagnostic testing was performed to find a stable, sustainable configuration:

| n_samples | background size | time per image | outcome |
|-----------|-----------------|-----------------|---------|
| 50        | 20              | —               | kernel crash (memory exhaustion) |
| 3         | 3               | ~51s            | stable |
| 5         | 3               | ~166s           | stable but disproportionately slower |

A separate diagnostic isolated the cause: a single model forward+backward pass
takes only ~4 seconds. The gap between this and the observed 51-166s runtimes
confirmed the bottleneck was in captum's internal batch expansion / memory
management during sampling, not the model's own compute cost — a memory
pressure ("thrashing") signature rather than a linear compute scaling issue.

An attempt to mitigate this by running GradientSHAP at a downsampled resolution
(256x256 internally, upsampled back to 512x512 for the model) was tested and
**rejected** — it added interpolation overhead without addressing the actual
bottleneck (captum's internals, not input resolution) and made runtime worse,
not better.

**Final committed configuration: `n_samples=3`, `background=3`.** This was the
largest configuration empirically verified as stable on the available hardware.
The full run (27 images x 2 models = 54 total explanations) completed in
**11.2 minutes with zero failures**, which exceeded expectations given the
slower per-image diagnostic timings — likely due to reduced background system
load and one-time framework warmup costs not present in the smaller diagnostic
tests.

## Preprocessing

Matches `week1_training.ipynb`'s `val_transform` exactly:
`Resize((512, 512))` → `ToTensor()` → `Normalize(ImageNet mean/std)`. No
augmentation at inference time. Architecture loading uses `timm` (not
torchvision) — confirmed necessary by inspecting checkpoint state_dict key
names (`conv_stem`, `blocks.N.M`, `se.conv_reduce/expand`), which match timm's
EfficientNet/ResNet naming convention, not torchvision's.

## Output contract compliance

- Heatmaps saved as `.npy`, shape `(H, W)` matching the **original** fundus
  image resolution (e.g. 2848x4288), not the model's 512x512 input size

- Min-max normalized to [0, 1] per image

- Absolute value taken before normalization, since SHAP attributions can be
  negative (pixel pushes away from predicted class) — for saliency/fidelity
  comparison against binary lesion masks, magnitude of influence matters,
  not sign

- Naming: `{model}_shap_{image_id}.npy`, saved to `results/heatmaps/{model}/shap/`

- Explains the model's **predicted** grade, not ground truth, per contract

- Predictions logged to `results/heatmaps/predictions.csv`
  (image_id, model, predicted_grade, true_grade, confidence)

- Zero failures across all 54 runs — no skipped images, no NaN entries

## Visual sanity check results

Spot-checked across multiple cases (highest confidence, lowest confidence,
and a misclassification):

- **High-confidence correct "healthy" prediction** (IDRiD_80, resnet50,
  conf=1.00): heatmap nearly blank, correctly finding little to attribute
  when the model is maximally certain nothing is abnormal.

- **Low-confidence correct "severe DR" prediction** (IDRiD_61, efficientnetb4,
  conf=0.35): heatmap shows multiple distinct hot clusters scattered across
  the retina, spatially consistent with visible lesion-like spots in the
  original image.

- **Misclassification** (IDRiD_56, efficientnetb4, pred=1 true=0, conf=0.76):
  heatmap concentrates near the optic disc/macula region, suggesting the
  model may be reacting to normal anatomical structures and mistaking them
  for pathology — a plausible explanation for the misclassification.

Across all spot-checks, attribution consistently lands on retinal structures
(never on black background/corners, never uniform noise), and the spatial
pattern varies sensibly with prediction confidence and correctness.

## Known limitations / caveats for the writeup

- `n_samples=3` is low for GradientSHAP by typical standards (literature often
  uses 50+); this is a deliberate, documented hardware-constrained tradeoff,
  not an oversight. Should be flagged in the paper's limitations section.

- Background sample size (3 images) is small; a larger, more diverse Grade-0
  sample would likely produce smoother, more stable attributions if compute
  resources allow in future work.

- Results have not yet been quantitatively validated against ground-truth
  lesion masks (IoU/Dice) — that comparison is Person 3's fidelity scoring
  step, pending.