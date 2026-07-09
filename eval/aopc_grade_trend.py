"""
eval/aopc_grade_trend.py

Reads existing aopc_{model}_{method}.csv files and produces:
1. Grade-wise AOPC trend CSV per model/method combo
2. Combined plot showing deletion + insertion AUC across DR grades

Usage:
    python eval/aopc_grade_trend.py

Writes to:
    results/scores/summary/grade_trend_{model}_{method}.csv  (already exists — overwrites)
    results/scores/summary/aopc_grade_trend_plot.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = '/content/drive/MyDrive/Projects/diabetic-retinopathy-xai'
AOPC_DIR     = f'{PROJECT_ROOT}/results/scores/aopc'
SUMMARY_DIR  = f'{PROJECT_ROOT}/results/scores/summary'
os.makedirs(SUMMARY_DIR, exist_ok=True)

MODELS  = ['efficientnetb4', 'resnet50']
METHODS = ['gradcam', 'lime']

GRADE_LABELS = {
    0: 'No DR',
    1: 'Mild NPDR',
    2: 'Moderate NPDR',
    3: 'Severe NPDR',
    4: 'PDR'
}

# ── Load and process each combo ───────────────────────────────────────────────
all_trends = []

for model in MODELS:
    for method in METHODS:
        csv_path = os.path.join(AOPC_DIR, f'grade_trend_{model}_{method}.csv')
        if not os.path.exists(csv_path):
            print(f'MISSING: {csv_path}')
            continue

        df = pd.read_csv(csv_path)
        df['grade_label'] = df['predicted_grade'].map(GRADE_LABELS)
        df['model']       = model
        df['method']      = method
        all_trends.append(df)

        print(f'{model}/{method}: {len(df)} grades, '
              f'deletion_auc range [{df["mean_deletion_auc"].min():.3f}, '
              f'{df["mean_deletion_auc"].max():.3f}]')

        # Overwrite summary CSV with grade labels added
        out_path = os.path.join(SUMMARY_DIR, f'grade_trend_{model}_{method}.csv')
        df.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')

combined = pd.concat(all_trends, ignore_index=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
# 2x2 grid: rows = models, cols = deletion/insertion AUC
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=False)
fig.suptitle('Grade-wise AOPC Trend — GradCAM vs LIME\n'
             '(Higher deletion AUC = better explanation; '
             'Lower insertion AUC = better explanation)',
             fontsize=13, y=1.01)

colors = {'gradcam': '#E63946', 'lime': '#457B9D'}
grades = [0, 1, 2, 3, 4]
x      = np.arange(len(grades))
grade_labels = [GRADE_LABELS[g] for g in grades]

for row, model in enumerate(MODELS):
    model_df = combined[combined['model'] == model]

    for col, metric in enumerate(['mean_deletion_auc', 'mean_insertion_auc']):
        ax = axes[row][col]

        for method in METHODS:
            sub = model_df[model_df['method'] == method].copy()
            sub = sub.set_index('predicted_grade').reindex(grades)
            vals = sub[metric].values

            ax.plot(x, vals, marker='o', linewidth=2,
                    label=method.upper(), color=colors[method])
            ax.fill_between(x, vals, alpha=0.1, color=colors[method])

            # Annotate n_images per grade
            if 'n_images' in sub.columns:
                for xi, (g, n) in enumerate(zip(grades, sub['n_images'].values)):
                    if not np.isnan(n):
                        ax.annotate(f'n={int(n)}', (xi, vals[xi]),
                                    textcoords='offset points',
                                    xytext=(0, 8), ha='center', fontsize=7,
                                    color=colors[method])

        metric_label = 'Deletion AUC ↑' if 'deletion' in metric else 'Insertion AUC ↓'
        ax.set_title(f'{model} — {metric_label}', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(grade_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('AUC', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

        # Highlight Grade 1 (Mild NPDR) — the known problem grade
        ax.axvspan(0.5, 1.5, alpha=0.08, color='orange',
                   label='Mild NPDR (known gap)')

plt.tight_layout()
plot_path = os.path.join(SUMMARY_DIR, 'aopc_grade_trend_plot.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nPlot saved: {plot_path}')

# ── Print summary table ───────────────────────────────────────────────────────
print('\n=== AOPC Grade Trend Summary ===')
pivot = combined.pivot_table(
    index=['model', 'method'],
    columns='predicted_grade',
    values='mean_deletion_auc'
).round(3)
pivot.columns = [GRADE_LABELS.get(c, c) for c in pivot.columns]
print(pivot.to_string())

print('\n=== Mild NPDR (Grade 1) specifically ===')
mild = combined[combined['predicted_grade'] == 1][
    ['model', 'method', 'mean_deletion_auc', 'mean_insertion_auc', 'n_images']
].sort_values('mean_deletion_auc', ascending=False)
print(mild.to_string(index=False))
