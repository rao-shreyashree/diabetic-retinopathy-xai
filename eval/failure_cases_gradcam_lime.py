"""
eval/failure_cases_gradcam_lime.py

Reads fidelity_scores_{model}_{method}.csv for gradcam and lime,
finds lowest-IoU images (failure cases) per lesion type per grade,
cross-checks against method_rank_correlation.csv,
saves overlay PNGs to results/heatmaps/{model}/{method}/failure_cases/

Usage:
    python eval/failure_cases_gradcam_lime.py

Writes:
    results/scores/summary/failure_cases_{model}_{method}.csv  (already exists — enriches)
    results/heatmaps/{model}/{method}/failure_cases/{image_id}_{lesion}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = '/content/drive/MyDrive/Projects/diabetic-retinopathy-xai'
FIDELITY_DIR = f'{PROJECT_ROOT}/results/scores/fidelity'
SUMMARY_DIR  = f'{PROJECT_ROOT}/results/scores/summary'
HEATMAP_DIR  = f'{PROJECT_ROOT}/results/heatmaps'
IMG_DIR      = '/content/drive/MyDrive/IDRiD/grading/images/test'

MODELS  = ['efficientnetb4', 'resnet50']
METHODS = ['gradcam', 'lime']

GRADE_LABELS = {0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR',
                3: 'Severe NPDR', 4: 'PDR'}

# ── Helper: load image by IDRiD_55 style ID ──────────────────────────────────
def load_image(image_id, img_dir):
    num   = int(image_id.replace('IDRiD_', ''))
    fname = f'IDRiD_{num:03d}.jpg'
    return Image.open(os.path.join(img_dir, fname)).convert('RGB')

def overlay_heatmap(pil_img, heatmap, alpha=0.5):
    """Blends heatmap (H,W float32 [0,1]) onto PIL image."""
    hm_resized = np.array(
        Image.fromarray(heatmap).resize(
            (pil_img.size[0], pil_img.size[1]), Image.BILINEAR
        )
    )
    colored = (cm.jet(hm_resized)[:, :, :3] * 255).astype(np.uint8)
    orig    = np.array(pil_img)
    blended = (alpha * orig + (1 - alpha) * colored).astype(np.uint8)
    return Image.fromarray(blended)

# ── Load rank correlation for cross-check ────────────────────────────────────
rank_corr = {}
for model in MODELS:
    rc_path = os.path.join(SUMMARY_DIR, f'method_rank_correlation_{model}.csv')
    if os.path.exists(rc_path):
        rank_corr[model] = pd.read_csv(rc_path)
        print(f'Loaded rank correlation for {model}: {rank_corr[model].shape}')
    else:
        print(f'No rank correlation found for {model} — skipping cross-check')

# ── Main: find failure cases per model/method ─────────────────────────────────
N_FAILURE_CASES = 5   # top-N worst IoU cases per lesion type

for model in MODELS:
    for method in METHODS:
        print(f'\n{"="*50}')
        print(f'{model} / {method}')
        print(f'{"="*50}')

        fid_path = os.path.join(FIDELITY_DIR,
                                f'fidelity_scores_{model}_{method}.csv')
        if not os.path.exists(fid_path):
            print(f'  MISSING fidelity CSV — skipping')
            continue

        df = pd.read_csv(fid_path)
        print(f'  Loaded fidelity scores: {df.shape}')
        print(f'  Columns: {df.columns.tolist()}')
        print(f'  IoU range: [{df["iou"].min():.4f}, {df["iou"].max():.4f}]')

        # ── Find failure cases: lowest IoU per lesion type ────────────────────
        failure_rows = []
        for lesion in df['lesion_type'].unique():
            lesion_df = df[df['lesion_type'] == lesion].copy()
            worst     = lesion_df.nsmallest(N_FAILURE_CASES, 'iou')
            failure_rows.append(worst)

        failure_df = pd.concat(failure_rows, ignore_index=True)
        failure_df = failure_df.sort_values(['lesion_type', 'iou'])

        # Cross-check: flag cases where this method ranks worst in correlation
        if model in rank_corr:
            rc = rank_corr[model]
            if 'image_id' in rc.columns:
                worst_rank_ids = set(rc.nsmallest(10, rc.columns[-1])['image_id'].values
                                     if len(rc.columns) > 1 else [])
                failure_df['low_rank_correlation'] = failure_df['image_id'].isin(worst_rank_ids)
            else:
                failure_df['low_rank_correlation'] = False
        else:
            failure_df['low_rank_correlation'] = False

        # Save enriched failure cases CSV
        out_csv = os.path.join(SUMMARY_DIR, f'failure_cases_{model}_{method}.csv')
        failure_df.to_csv(out_csv, index=False)
        print(f'  Saved failure cases CSV: {out_csv}')
        print(f'  Total failure cases: {len(failure_df)}')

        # ── Generate overlay PNGs ─────────────────────────────────────────────
        overlay_dir = os.path.join(HEATMAP_DIR, model, method, 'failure_cases')
        os.makedirs(overlay_dir, exist_ok=True)

        generated = 0
        for _, row in failure_df.iterrows():
            image_id  = row['image_id']
            lesion    = row['lesion_type']
            iou_val   = row['iou']
            grade     = int(row['predicted_grade']) if 'predicted_grade' in row else -1

            # Load heatmap
            hm_path = os.path.join(HEATMAP_DIR, model, method,
                                   f'{model}_{method}_{image_id}.npy')
            if not os.path.exists(hm_path):
                print(f'    Heatmap missing for {image_id} — skipping overlay')
                continue

            try:
                heatmap = np.load(hm_path)
                pil_img = load_image(image_id, IMG_DIR)
                overlay = overlay_heatmap(pil_img, heatmap)

                # Save with grade + lesion + IoU in filename
                grade_str = GRADE_LABELS.get(grade, f'grade{grade}').replace(' ', '_')
                fname     = f'{image_id}_{lesion}_grade{grade}_{grade_str}_iou{iou_val:.3f}.png'
                overlay.save(os.path.join(overlay_dir, fname))
                generated += 1

            except Exception as e:
                print(f'    Failed overlay for {image_id}: {e}')

        print(f'  Generated {generated} overlay PNGs → {overlay_dir}')

        # ── Print summary table ───────────────────────────────────────────────
        print(f'\n  Failure cases by lesion type:')
        summary = failure_df.groupby('lesion_type')['iou'].agg(['mean', 'min', 'count'])
        print(summary.to_string())

        if 'predicted_grade' in failure_df.columns:
            print(f'\n  Failure cases by grade:')
            grade_sum = failure_df.groupby('predicted_grade')['iou'].agg(['mean', 'count'])
            grade_sum.index = [f"Grade {g} ({GRADE_LABELS.get(g,'?')})"
                               for g in grade_sum.index]
            print(grade_sum.to_string())

print('\n\nDone. All failure cases saved.')
