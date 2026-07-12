"""
eval/aopc_grade_trend.py

Computes and saves grade-wise AOPC trend for any model/method combination.
Reads raw aopc_{model}_{method}.csv (per-image AOPC scores),
aggregates by predicted_grade, writes grade_trend_{model}_{method}.csv.

Usage:
    # Single combo
    python eval/aopc_grade_trend.py --model efficientnetb4 --method gradcam

    # Multiple combos
    python eval/aopc_grade_trend.py \
        --model efficientnetb4 resnet50 vit_b16 \
        --method gradcam lime shap attention_rollout

    # All available combos (auto-discover from files on disk)
    python eval/aopc_grade_trend.py --all

Reads:
    results/scores/aopc/aopc_{model}_{method}.csv

Writes:
    results/scores/aopc/grade_trend_{model}_{method}.csv
    results/scores/summary/aopc_grade_trend_plot.png  (if --plot)
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PROJECT_ROOT = '/content/drive/MyDrive/Projects/diabetic-retinopathy-xai'
AOPC_DIR     = os.path.join(PROJECT_ROOT, 'results/scores/aopc')
SUMMARY_DIR  = os.path.join(PROJECT_ROOT, 'results/scores/summary')

GRADE_LABELS = {
    0: 'No DR', 1: 'Mild NPDR', 2: 'Moderate NPDR',
    3: 'Severe NPDR', 4: 'PDR',
}

METHOD_COLORS = {
    'gradcam': '#E63946', 'lime': '#457B9D',
    'shap': '#2A9D8F', 'attention_rollout': '#E9C46A',
}

KNOWN_MODELS = ['efficientnetb4', 'resnet50', 'vit_b16']


def compute_grade_trend(aopc_csv_path, model, method):
    df = pd.read_csv(aopc_csv_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    grade_col = next((c for c in df.columns if 'grade' in c or 'predicted' in c), None)
    del_col   = next((c for c in df.columns if 'deletion' in c and 'auc' in c), None)
    ins_col   = next((c for c in df.columns if 'insertion' in c and 'auc' in c), None)

    if not grade_col:
        raise ValueError(f"No grade column in {aopc_csv_path}. Cols: {df.columns.tolist()}")
    if not del_col or not ins_col:
        raise ValueError(f"No deletion/insertion AUC cols in {aopc_csv_path}. Cols: {df.columns.tolist()}")

    agg = (
        df.groupby(grade_col)
        .agg(
            mean_deletion_auc=(del_col, 'mean'),
            std_deletion_auc=(del_col, 'std'),
            mean_insertion_auc=(ins_col, 'mean'),
            std_insertion_auc=(ins_col, 'std'),
            n_images=(del_col, 'count'),
        )
        .reset_index()
        .rename(columns={grade_col: 'predicted_grade'})
    )
    agg['model']       = model
    agg['method']      = method
    agg['grade_label'] = agg['predicted_grade'].map(GRADE_LABELS)
    cols = ['model', 'method', 'predicted_grade', 'grade_label',
            'mean_deletion_auc', 'std_deletion_auc',
            'mean_insertion_auc', 'std_insertion_auc', 'n_images']
    return agg[[c for c in cols if c in agg.columns]]


def discover_combos(aopc_dir):
    combos = []
    for fpath in glob.glob(os.path.join(aopc_dir, 'aopc_*.csv')):
        fname  = os.path.basename(fpath).replace('aopc_', '').replace('.csv', '')
        model  = next((m for m in KNOWN_MODELS if fname.startswith(m)), None)
        if not model:
            print(f"  [SKIP] Cannot parse model from: {fpath}")
            continue
        method = fname[len(model) + 1:]
        combos.append((model, method))
    return combos


def plot_grade_trends(all_trends, output_path):
    models = sorted(all_trends['model'].unique())
    n_rows = len(models)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.5 * n_rows))
    if n_rows == 1:
        axes = [axes]
    fig.suptitle('Grade-wise AOPC Trend\nDeletion AUC higher = better; Insertion AUC lower = better',
                 fontsize=13, y=1.01)
    grades     = list(GRADE_LABELS.keys())
    grade_lbls = [GRADE_LABELS[g] for g in grades]
    x          = np.arange(len(grades))

    for row, model in enumerate(models):
        model_df = all_trends[all_trends['model'] == model]
        for col, metric in enumerate(['mean_deletion_auc', 'mean_insertion_auc']):
            ax = axes[row][col]
            for method in sorted(model_df['method'].unique()):
                sub   = model_df[model_df['method'] == method].set_index('predicted_grade').reindex(grades)
                vals  = sub[metric].values
                color = METHOD_COLORS.get(method, '#888888')
                ax.plot(x, vals, marker='o', linewidth=2, label=method.upper(), color=color)
                ax.fill_between(x, vals, alpha=0.08, color=color)
                if 'n_images' in sub.columns:
                    for xi, n in enumerate(sub['n_images'].values):
                        if not np.isnan(n):
                            ax.annotate(f'n={int(n)}', (xi, vals[xi]),
                                        textcoords='offset points', xytext=(0, 8),
                                        ha='center', fontsize=7, color=color)
            metric_label = 'Deletion AUC' if 'deletion' in metric else 'Insertion AUC'
            ax.set_title(f'{model} — {metric_label}', fontsize=11)
            ax.set_xticks(x)
            ax.set_xticklabels(grade_lbls, rotation=20, ha='right', fontsize=9)
            ax.set_ylabel('AUC', fontsize=9)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
            ax.axvspan(0.5, 1.5, alpha=0.07, color='orange')
            ax.text(1, ax.get_ylim()[0], 'Mild\nNPDR', ha='center',
                    fontsize=7, color='darkorange', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Plot saved: {output_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    nargs='+')
    parser.add_argument('--method',   nargs='+')
    parser.add_argument('--all',      action='store_true')
    parser.add_argument('--plot',     action='store_true', default=True)
    parser.add_argument('--no-plot',  dest='plot', action='store_false')
    args = parser.parse_args()

    os.makedirs(AOPC_DIR,    exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)

    if args.all:
        combos = discover_combos(AOPC_DIR)
        print(f'Discovered {len(combos)} combos: {combos}')
    elif args.model and args.method:
        combos = [(m, meth) for m in args.model for meth in args.method]
    else:
        parser.error('Provide --model + --method, or use --all')

    all_trends = []
    for model, method in combos:
        aopc_path = os.path.join(AOPC_DIR, f'aopc_{model}_{method}.csv')
        if not os.path.exists(aopc_path):
            print(f'[SKIP] Not found: {aopc_path}')
            continue
        print(f'\nProcessing: {model} / {method}')
        try:
            trend = compute_grade_trend(aopc_path, model, method)
        except ValueError as e:
            print(f'  [ERROR] {e}')
            continue
        out_path = os.path.join(AOPC_DIR, f'grade_trend_{model}_{method}.csv')
        trend.to_csv(out_path, index=False)
        print(f'  Saved: {out_path}')
        print(trend[['predicted_grade','grade_label','mean_deletion_auc','mean_insertion_auc','n_images']].to_string(index=False))
        all_trends.append(trend)

    if not all_trends:
        print('No results — check AOPC CSVs exist in:', AOPC_DIR)
        return

    combined = pd.concat(all_trends, ignore_index=True)

    print('\n=== Deletion AUC by grade ===')
    pivot = combined.pivot_table(
        index=['model','method'], columns='predicted_grade', values='mean_deletion_auc'
    ).round(3)
    pivot.columns = [GRADE_LABELS.get(c, str(c)) for c in pivot.columns]
    print(pivot.to_string())

    print('\n=== Mild NPDR (Grade 1) ===')
    mild = combined[combined['predicted_grade'] == 1][
        ['model','method','mean_deletion_auc','mean_insertion_auc','n_images']
    ].sort_values('mean_deletion_auc', ascending=False)
    print(mild.to_string(index=False) if not mild.empty else
          '  No Grade 1 predictions (confirms mild NPDR classification gap)')

    if args.plot:
        plot_grade_trends(combined, os.path.join(SUMMARY_DIR, 'aopc_grade_trend_plot.png'))


if __name__ == '__main__':
    main()
