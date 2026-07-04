import pandas as pd
import glob
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCORES_DIR = os.path.join(PROJECT_ROOT, "results", "scores")

# fidelity
fid = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(SCORES_DIR, "fidelity", "fidelity_scores_*.csv"))], ignore_index=True)
print("=== FIDELITY: mean IoU (model x method) ===")
print(fid.groupby(['model','method'])['iou'].mean().round(4).unstack())
print()
print("=== FIDELITY: mean IoU by lesion ===")
print(fid.groupby(['model','method','lesion_type'])['iou'].mean().round(4).unstack())
fid.groupby(['model','method'])['iou'].mean().round(4).unstack().to_csv(
    os.path.join(SCORES_DIR, "cross_fidelity_iou.csv")
)
fid.groupby(['model','method','lesion_type'])['iou'].mean().round(4).unstack().to_csv(
    os.path.join(SCORES_DIR, "cross_fidelity_by_lesion.csv")
)

# pointing game
pg = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(SCORES_DIR, "pointing_game", "pointing_game_*.csv"))], ignore_index=True)
print()
print("=== POINTING GAME: accuracy (model x method) ===")
print(pg.groupby(['model','method'])['hit'].mean().round(4).unstack())
print()
print("=== POINTING GAME: accuracy by lesion ===")
print(pg.groupby(['model','method','lesion_type'])['hit'].mean().round(4).unstack())
pg.groupby(['model','method'])['hit'].mean().round(4).unstack().to_csv(
    os.path.join(SCORES_DIR, "cross_pointing_game.csv")
)
pg.groupby(['model','method','lesion_type'])['hit'].mean().round(4).unstack().to_csv(
    os.path.join(SCORES_DIR, "cross_pointing_game_by_lesion.csv")
)

# aopc
aopc = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(SCORES_DIR, "aopc", "aopc_*.csv"))], ignore_index=True)
print()
print("=== AOPC (model x method) ===")
auc_cols = [c for c in aopc.columns if 'auc' in c.lower() or 'deletion' in c.lower() or 'insertion' in c.lower()]
print(aopc.groupby(['model','method'])[auc_cols].mean().round(4))
aopc.groupby(['model','method'])[auc_cols].mean().round(4).to_csv(
    os.path.join(SCORES_DIR, "cross_aopc.csv")
)

print("Saved all summary CSVs.")