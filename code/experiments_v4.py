"""
Fine-grained threshold sweep on Experiment B test set.
Exports F1/FPR/precision/recall curves over the threshold range so the paper
can honestly quantify how much detection is recoverable AND the false-alarm cost.
"""

# --- personal note ---
# Default decision thresholds assume P(attack) is not tiny. In this data it
# kind of is, and that is the whole point: RF needs t=0.01 to be usable at all,
# and even then you pay with false alarms. This sweep is the only experiment
# where "fixing" the model means changing the goalpost instead of the model.

import pandas as pd
import numpy as np
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'
DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'

df = pd.read_csv(DATA_PATH)
label_col = 'Label'
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
drop_cols = [c for c in df.columns if c != label_col and df[c].dtype == 'object']
feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]

MAX_ROWS = 50000
if len(df) > MAX_ROWS:
    sampled_dfs = []
    for label_val in df[label_col].unique():
        subset = df[df[label_col] == label_val]
        n_keep = int(MAX_ROWS * len(subset) / len(df))
        n_keep = max(n_keep, len(subset)) if label_val != 'BENIGN' else n_keep
        sampled_dfs.append(subset.sample(min(len(subset), n_keep), random_state=42))
    df = pd.concat(sampled_dfs).reset_index(drop=True)

df['binary_label'] = (df[label_col] != 'BENIGN').astype(int)
X = df[feature_cols].values.astype(np.float32)
y_bin = df['binary_label'].values
is_ddos = (df[label_col] == 'DDoS').values
is_benign = (df[label_col] == 'BENIGN').values
is_bot = (df[label_col] == 'Bot').values
is_portscan = (df[label_col] == 'PortScan').values

morning_mask = is_bot.copy()
afternoon_mask = is_ddos | is_portscan
benign_idx = np.where(is_benign)[0]
np.random.seed(42)
n_benign_morning = min(int(0.4 * is_benign.sum()), 30000)
n_benign_afternoon = min(int(0.4 * is_benign.sum()), 30000)
morning_benign = np.random.choice(benign_idx, size=n_benign_morning, replace=False)
remaining_benign = np.setdiff1d(benign_idx, morning_benign)
afternoon_benign = np.random.choice(remaining_benign, size=min(n_benign_afternoon, len(remaining_benign)), replace=False)
morning_mask[morning_benign] = True
afternoon_mask_full = afternoon_mask.copy()
afternoon_mask_full[afternoon_benign] = True

X_tr_b, y_tr_b = X[morning_mask], y_bin[morning_mask]
X_te_b, y_te_b = X[afternoon_mask_full], y_bin[afternoon_mask_full]

def get_models():
    return {
        'LR': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(
            n_estimators=80, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=50, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
    }

y_true = y_te_b
n_pos = (y_true == 1).sum()
n_neg = (y_true == 0).sum()
thresholds = np.load('thresholds.npz')['t'] if os.path.exists('thresholds.npz') else np.arange(0.0, 1.00001, 0.001)

out = {}
for name, model in get_models().items():
    t0 = time.time()
    scaler = StandardScaler()
    X_tra = np.nan_to_num(scaler.fit_transform(X_tr_b), nan=0.0)
    X_tes = np.nan_to_num(scaler.transform(X_te_b), nan=0.0)
    model.fit(X_tra, y_tr_b)
    p = model.predict_proba(X_tes)[:, 1]
    auc = roc_auc_score(y_true, p)

    # vectorized curve
    order = np.argsort(-p, kind='stable')
    ps = p[order]
    ys = y_true[order]
    cum_attack = np.cumsum(ys)
    idx = np.searchsorted(-ps, -thresholds, side='right')
    idx = np.clip(idx, 1, len(ys))
    tp = cum_attack[idx - 1]
    fp = idx - tp
    precision = tp / np.maximum(idx, 1)
    recall = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    frac_pred = idx / len(ys)

    best_i = int(np.argmax(f1))
    default_i = int(np.argmin(np.abs(thresholds - 0.5)))

    out[name] = {
        'roc_auc': float(auc),
        'default_f1': float(f1[default_i]),
        'default_fpr': float(fpr[default_i]),
        'best_threshold': float(thresholds[best_i]),
        'best_f1': float(f1[best_i]),
        'best_recall': float(recall[best_i]),
        'best_precision': float(precision[best_i]),
        'best_fpr': float(fpr[best_i]),
        'best_frac_pred': float(frac_pred[best_i]),
        'f1_curve': [float(x) for x in f1[::5]],
        'fpr_curve': [float(x) for x in fpr[::5]],
        'thresholds_plot': [float(x) for x in thresholds[::5]],
        'train_time_s': float(time.time() - t0),
    }
    print(f"  {name}: AUC={auc:.4f} default F1={out[name]['default_f1']:.4f} "
          f"FPR={out[name]['default_fpr']:.4f} | best F1={out[name]['best_f1']:.4f} "
          f"@t={out[name]['best_threshold']:.3f} FPR={out[name]['best_fpr']:.4f} "
          f"recall={out[name]['best_recall']:.3f} préc={out[name]['best_precision']:.3f} "
          f"frac_pred={out[name]['best_frac_pred']:.3f}")

with open(os.path.join(OUT_DIR, 'threshold_sweep_curve.json'), 'w') as f:
    json.dump(out, f, indent=2)
print("Saved threshold_sweep_curve.json")
print("COMPLETE")