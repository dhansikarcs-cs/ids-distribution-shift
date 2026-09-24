"""
Additional experiments to fix the train-size confound.

Experiment B2 (Matched-Size Control):
  - Train on 13K RANDOM samples from the full dataset
  - Test on the SAME temporal test set (DDoS + PortScan + Benign = 298K)
  - Purpose: Isolate whether RF/XGB collapse is due to distribution shift or small train size

Experiment B3 (Same-Distribution + Same-Size):
  - Train on 13K RANDOM samples
  - Test on 298K RANDOM samples from the same distribution
  - Purpose: What performance does a 13K-trained model get under ideal conditions?

Combined 2x2:
  A  (large train, same dist)     = baseline
  B  (small train, DIFFERENT dist) = temporal shift
  B2 (small train, SAME dist)     = control for train size
  B3 (small train, same dist, same test size) = full control
"""

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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from xgboost import XGBClassifier

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'
DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'

# ============ Load and prepare data (same as experiments.py) ============
df = pd.read_csv(DATA_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

label_col = 'Label'
MAX_ROWS = 50000
sampled_dfs = []
for label_val in df[label_col].unique():
    subset = df[df[label_col] == label_val]
    n_keep = int(MAX_ROWS * len(subset) / len(df))
    n_keep = max(n_keep, len(subset)) if label_val != 'BENIGN' else n_keep
    sampled_dfs.append(subset.sample(min(len(subset), n_keep), random_state=42))
df = pd.concat(sampled_dfs).reset_index(drop=True)
df['binary_label'] = (df[label_col] != 'BENIGN').astype(int)

drop_cols = [c for c in df.columns if c != label_col and df[c].dtype == 'object']
feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]

X = df[feature_cols].values.astype(np.float32)
y_bin = df['binary_label'].values

is_ddos = (df[label_col] == 'DDoS').values
is_portscan = (df[label_col] == 'PortScan').values
is_bot = (df[label_col] == 'Bot').values
is_benign = (df[label_col] == 'BENIGN').values

print(f"Dataset: {len(df):,} rows, {len(feature_cols)} features")

# ============ Recreate temporal test set (same seed) ============
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

X_te_temporal = X[afternoon_mask_full]
y_te_temporal = y_bin[afternoon_mask_full]
print(f"Temporal test set: {len(X_te_temporal):,} samples")

# Temporal train set size
n_temporal_train = morning_mask.sum()
print(f"Temporal train set size: {n_temporal_train:,}")

# ============ Define models ============
def get_models():
    return {
        'LR': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
        'RF': RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1),
        'XGB': XGBClassifier(
            n_estimators=80, max_depth=5, learning_rate=0.1,
            random_state=42, use_label_encoder=False, eval_metric='logloss',
            n_jobs=-1
        ),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=50, random_state=42,
            early_stopping=True, validation_fraction=0.1
        ),
    }

def evaluate(y_true, y_pred, y_prob=None, task='binary'):
    average = 'binary' if task == 'binary' else 'weighted'
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if task == 'binary':
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        metrics['fpr'] = float(fp / max(tn + fp, 1))
        if y_prob is not None:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_prob))
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
    return metrics

def run_single(model_name, model, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_te = np.nan_to_num(scaler.transform(X_test), nan=0.0)

    t0 = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - t0

    t_pred = time.time()
    y_pred = model.predict(X_te)
    infer_time = time.time() - t_pred

    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_te)[:, 1]
        except:
            pass

    metrics = evaluate(y_test, y_pred, y_prob, 'binary')
    metrics['inference_time_ms'] = float(infer_time * 1000 / len(X_te))
    metrics['train_time_s'] = float(train_time)
    return metrics

# ============================================================
# B2: Matched-Size Control
# Train on n_temporal_train RANDOM samples, test on temporal test set
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT B2: MATCHED-SIZE CONTROL (small random train -> temporal test)")
print("=" * 70)

results_b2 = {}
for model_name in ['LR', 'RF', 'XGB', 'MLP']:
    # Use same random_state for the subsample but different from temporal
    np.random.seed(123)  # Different seed from temporal
    all_indices = np.arange(len(X))
    train_indices = np.random.choice(all_indices, size=n_temporal_train, replace=False)

    X_train_b2 = X[train_indices]
    y_train_b2 = y_bin[train_indices]

    print(f"\n  {model_name}: train={len(X_train_b2):,} (random), test={len(X_te_temporal):,} (temporal)")
    model = get_models()[model_name]
    metrics = run_single(model_name, model, X_train_b2, X_te_temporal, y_train_b2, y_te_temporal)
    results_b2[model_name] = metrics
    print(f"    F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} ROC-AUC={metrics.get('roc_auc', 0):.4f}")

# ============================================================
# B3: Same-Distribution + Same-Size Control
# Train on 13K random, test on 298K random from same distribution
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT B3: SAME-DISTRIBUTION + SAME-SIZE CONTROL")
print("=" * 70)

results_b3 = {}
for model_name in ['LR', 'RF', 'XGB', 'MLP']:
    np.random.seed(42)
    X_tr_b3, X_te_b3, y_tr_b3, y_te_b3 = train_test_split(
        X, y_bin, test_size=len(X_te_temporal), random_state=42, stratify=y_bin
    )
    # Further subsample train to match temporal train size
    np.random.seed(123)
    train_idx = np.random.choice(len(X_tr_b3), size=n_temporal_train, replace=False)
    X_tr_b3_sub = X_tr_b3[train_idx]
    y_tr_b3_sub = y_tr_b3[train_idx]

    print(f"\n  {model_name}: train={len(X_tr_b3_sub):,} (random), test={len(X_te_b3):,} (random, same dist)")
    model = get_models()[model_name]
    metrics = run_single(model_name, model, X_tr_b3_sub, X_te_b3, y_tr_b3_sub, y_te_b3)
    results_b3[model_name] = metrics
    print(f"    F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f} ROC-AUC={metrics.get('roc_auc', 0):.4f}")

# ============================================================
# Save new results, merge with existing
# ============================================================
with open(os.path.join(OUT_DIR, 'all_results.json'), 'r') as f:
    all_results = json.load(f)

all_results['B2_matched_size_control'] = results_b2
all_results['B3_same_dist_same_size'] = results_b3

with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# ============================================================
# Summary comparison
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: Train-Size Confound Analysis")
print("=" * 70)
print(f"\n{'Model':<8} {'A-Random':<10} {'B-Temporal':<12} {'B2-Matched':<12} {'B3-SameDist':<12}")
print("-" * 60)
for m in ['LR', 'RF', 'XGB', 'MLP']:
    a_f1 = all_results['A_random_split'][m]['f1']
    b_f1 = all_results['B_temporal_split'][m]['f1']
    b2_f1 = results_b2[m]['f1']
    b3_f1 = results_b3[m]['f1']
    print(f"{m:<8} {a_f1:<10.4f} {b_f1:<12.4f} {b2_f1:<12.4f} {b3_f1:<12.4f}")

print("\nInterpretation:")
print("  A = Large train, same distribution (best case)")
print("  B = Small train, DIFFERENT distribution (temporal shift)")
print("  B2 = Small train, same size as B, SAME distribution")
print("  B3 = Small train + Large test, same distribution")
print()
print("  If B << B2: Distribution shift is the primary cause")
print("  If B2 << B3: Train/test size ratio matters")
print("  If B2 << A: Train size matters")
