"""
FIXED ML-Based Network Intrusion Detection Robustness Under Distribution Shift
===============================================================================
Fixes applied:
  1. B2: No train/test overlap (train from complement of temporal test)
  2. Experiment C: No benign leakage (benign test samples excluded from training)
  3. B renamed to 'session/attack-family distribution shift' (not pure temporal)
"""

import pandas as pd
import numpy as np
import time
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 70)
print("PHASE 1: Loading and preprocessing data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

label_col = 'Label'

df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"Dropped {before - len(df)} rows with NaN/Inf. Remaining: {len(df):,}")

drop_cols = []
for c in df.columns:
    if c != label_col and df[c].dtype == 'object':
        drop_cols.append(c)

feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]
print(f"Features: {len(feature_cols)}")

MAX_ROWS = 50000
if len(df) > MAX_ROWS:
    sampled_dfs = []
    for label_val in df[label_col].unique():
        subset = df[df[label_col] == label_val]
        n_keep = int(MAX_ROWS * len(subset) / len(df))
        n_keep = max(n_keep, len(subset)) if label_val != 'BENIGN' else n_keep
        sampled_dfs.append(subset.sample(min(len(subset), n_keep), random_state=42))
    df = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"Subsampled to: {len(df):,}")

df['binary_label'] = (df[label_col] != 'BENIGN').astype(int)
le = LabelEncoder()
df['multi_label'] = le.fit_transform(df[label_col])
class_names = list(le.classes_)
print(f"Classes: {class_names}")

print(f"\nClass distribution after subsampling:")
for cn in class_names:
    count = (df[label_col] == cn).sum()
    print(f"  {cn}: {count:,} ({100*count/len(df):.1f}%)")

X = df[feature_cols].values.astype(np.float32)
y_bin = df['binary_label'].values
y_multi = df['multi_label'].values

is_ddos = (df[label_col] == 'DDoS').values
is_portscan = (df[label_col] == 'PortScan').values
is_bot = (df[label_col] == 'Bot').values
is_benign = (df[label_col] == 'BENIGN').values

# ============================================================
# 2. SESSION-ATTACK-FAMILY SPLIT (renamed from "temporal")
# ============================================================
# CICIDS2017 Friday: Bot attacks in morning session, DDoS/PortScan in afternoon
# This is a session/attack-family shift, not a pure timestamp-level temporal shift
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

session_test_indices = np.where(afternoon_mask_full)[0]
n_temporal_train = morning_mask.sum()

print(f"\nSession-based split:")
print(f"  Morning session (train): {morning_mask.sum():,}")
print(f"  Afternoon session (test): {afternoon_mask_full.sum():,}")
print(f"  Complement pool (full - test): {len(X) - len(session_test_indices):,}")

# ============================================================
# 3. DEFINE MODELS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: Defining models")
print("=" * 70)

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

print("Models: LR, RF, XGB, MLP")

# ============================================================
# 4. EVALUATION HELPER
# ============================================================

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
    else:
        if y_prob is not None and y_prob.shape[1] > 2:
            try:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr', average=average))
                metrics['pr_auc'] = float(average_precision_score(y_true, y_prob, average=average))
            except:
                metrics['roc_auc'] = 0.0
                metrics['pr_auc'] = 0.0
    return metrics


def run_experiment(name, X_train, X_test, y_train, y_test, task='binary'):
    print(f"\n--- {name} ---")
    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    results = {}
    models = get_models()

    for model_name, model in models.items():
        t0 = time.time()
        scaler = StandardScaler()
        X_tr = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
        X_te = np.nan_to_num(scaler.transform(X_test), nan=0.0)

        model.fit(X_tr, y_train)

        t_pred_start = time.time()
        y_pred = model.predict(X_te)
        t_pred = time.time() - t_pred_start

        y_prob = None
        if hasattr(model, 'predict_proba'):
            try:
                y_prob = model.predict_proba(X_te)
                if task == 'binary' and y_prob.shape[1] == 2:
                    y_prob = y_prob[:, 1]
            except:
                y_prob = None

        metrics = evaluate(y_test, y_pred, y_prob, task=task)
        metrics['inference_time_ms'] = float(t_pred * 1000 / len(X_te))
        metrics['train_time_s'] = float(time.time() - t0)
        results[model_name] = metrics
        print(f"  {model_name}: F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} "
              f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
              f"FPR={metrics.get('fpr', 0):.4f} Infer={metrics['inference_time_ms']:.4f}ms")

    return results


# ============================================================
# 5. RUN ALL EXPERIMENTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 3: Running experiments")
print("=" * 70)

all_results = {}

# --- EXPERIMENT A: Conventional Random Split ---
print("\n" + "=" * 70)
print("EXPERIMENT A: Conventional Random Split (80/20)")
print("=" * 70)
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
all_results['A_random_split'] = run_experiment(
    'A - Random Split', X_tr_a, X_te_a, y_tr_a, y_te_a, 'binary'
)

# --- EXPERIMENT B: Session-Based Distribution Shift ---
print("\n" + "=" * 70)
print("EXPERIMENT B: Session-Based Distribution Shift (Morning -> Afternoon)")
print("=" * 70)
X_tr_b, y_tr_b = X[morning_mask], y_bin[morning_mask]
X_te_b, y_te_b = X[afternoon_mask_full], y_bin[afternoon_mask_full]
print(f"  Morning (train): {morning_mask.sum():,} | Afternoon (test): {afternoon_mask_full.sum():,}")

all_results['B_session_shift'] = run_experiment(
    'B - Session Shift', X_tr_b, X_te_b, y_tr_b, y_te_b, 'binary'
)

# --- EXPERIMENT B2: Matched-Size Control (FIXED: clean same-distribution subsample) ---
print("\n" + "=" * 70)
print("EXPERIMENT B2: MATCHED-SIZE CONTROL (13K from A's training pool)")
print("=" * 70)

# FIX: Use the same 80/20 random split as Experiment A, then subsample training to match B's size
# This ensures: (1) zero train/test overlap, (2) same distribution as A, (3) same train size as B
np.random.seed(123)
train_idx_b2 = np.random.choice(len(X_tr_a), size=n_temporal_train, replace=False)
X_tr_b2 = X_tr_a[train_idx_b2]
y_tr_b2 = y_tr_a[train_idx_b2]

print(f"  Train: {len(X_tr_b2):,} (random subsample of A's training pool)")
print(f"  Test: {len(X_te_a):,} (same held-out random split as A)")
print(f"  Train/test overlap: 0 (subsampled from disjoint training pool)")
print(f"  Train class dist: attack={int(y_tr_b2.sum())}/{len(y_tr_b2)} ({100*y_tr_b2.mean():.1f}%)")

all_results['B2_matched_size_control'] = run_experiment(
    'B2 - Matched Size Control', X_tr_b2, X_te_a, y_tr_b2, y_te_a, 'binary'
)

# --- EXPERIMENT B3: Same-Distribution + Same-Size Control ---
print("\n" + "=" * 70)
print("EXPERIMENT B3: SAME-DISTRIBUTION + SAME-SIZE CONTROL")
print("=" * 70)

X_tr_b3, X_te_b3, y_tr_b3, y_te_b3 = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
np.random.seed(123)
train_idx_b3 = np.random.choice(len(X_tr_b3), size=n_temporal_train, replace=False)
X_tr_b3_sub = X_tr_b3[train_idx_b3]
y_tr_b3_sub = y_tr_b3[train_idx_b3]

print(f"  Train: {len(X_tr_b3_sub):,} (random from 80%) | Test: {len(X_te_b3):,} (random 20%)")

all_results['B3_same_dist_same_size'] = run_experiment(
    'B3 - Same Distribution Same Size', X_tr_b3_sub, X_te_b3, y_tr_b3_sub, y_te_b3, 'binary'
)

# --- EXPERIMENT C: Attack-Family Shift (FIX 2: no benign leakage) ---
print("\n" + "=" * 70)
print("EXPERIMENT C: Attack-Family Shift (Leave-One-Out, FIXED)")
print("=" * 70)

attack_classes = ['Bot', 'DDoS', 'PortScan']
results_c = {}

for held_out in attack_classes:
    print(f"\n  Held-out: {held_out}")

    # Test: held-out attack + selected benign
    test_attack_mask = (df[label_col] == held_out).values
    n_attack = test_attack_mask.sum()
    benign_indices_all = np.where(is_benign)[0]
    np.random.seed(42)
    # FIX: Cap benign test to leave at least 1000 benign in training
    n_benign_test = min(n_attack, len(benign_indices_all) - 1000)
    benign_test_indices = np.random.choice(benign_indices_all, size=n_benign_test, replace=False)

    # FIX 2: Remove benign test samples from training pool
    train_mask = (df[label_col] != held_out).values.copy()
    train_mask[benign_test_indices] = False  # exclude benign test samples from training

    test_mask = test_attack_mask.copy()
    test_mask[benign_test_indices] = True

    n_benign_train = (train_mask & is_benign).sum()
    n_benign_test = benign_test_indices.shape[0]
    print(f"  Train: benign={n_benign_train} (excluded {n_benign_test} test benign)")
    print(f"  Test: {held_out}={n_attack}, benign={n_benign_test}")

    X_tr, y_tr = X[train_mask], y_bin[train_mask]
    X_te, y_te = X[test_mask], y_bin[test_mask]
    results_c[held_out] = run_experiment(
        f'C - Leave-One-Out ({held_out})', X_tr, X_te, y_tr, y_te, 'binary'
    )

all_results['C_attack_family_shift'] = results_c

# --- EXPERIMENT D: Feature Distribution Shift ---
print("\n" + "=" * 70)
print("EXPERIMENT D: Feature Distribution Shift")
print("=" * 70)

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

all_results['D_no_shift_control'] = run_experiment(
    'D - No Shift (Control)', X_tr_d, X_te_d, y_tr_d, y_te_d, 'binary'
)

np.random.seed(42)
X_shifted = X_te_d.copy()
duration_col = feature_cols.index('Flow Duration') if 'Flow Duration' in feature_cols else 1
shift_factors = np.random.uniform(2.0, 5.0, size=len(X_shifted))
X_shifted[:, duration_col] = X_shifted[:, duration_col] * shift_factors

for col_name in ['Total Fwd Packets', 'Total Backward Packets']:
    if col_name in feature_cols:
        ci = feature_cols.index(col_name)
        noise = np.random.normal(1.5, 0.5, size=len(X_shifted))
        X_shifted[:, ci] = X_shifted[:, ci] * noise

X_shifted = np.clip(X_shifted, -1e6, 1e6)
all_results['D_feature_shift'] = run_experiment(
    'D - Feature Distribution Shift', X_tr_d, X_shifted, y_tr_d, y_te_d, 'binary'
)

# ============================================================
# 6. MULTI-CLASS EXPERIMENT
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT: Multi-Class Classification")
print("=" * 70)

X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)
results_mc = {}
for model_name, model in get_models().items():
    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_tr_mc), nan=0.0)
    X_te = np.nan_to_num(scaler.transform(X_te_mc), nan=0.0)

    t0 = time.time()
    model.fit(X_tr, y_tr_mc)
    train_time = time.time() - t0

    t_pred = time.time()
    y_pred = model.predict(X_te)
    infer_time = time.time() - t_pred

    y_prob = None
    if hasattr(model, 'predict_proba'):
        try:
            y_prob = model.predict_proba(X_te)
        except:
            pass

    metrics = evaluate(y_te_mc, y_pred, y_prob, task='multiclass')
    metrics['inference_time_ms'] = float(infer_time * 1000 / len(X_te))
    metrics['train_time_s'] = float(train_time)
    results_mc[model_name] = metrics
    print(f"  {model_name}: F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f}")

all_results['multiclass'] = results_mc

# ============================================================
# 7. SAVE RESULTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 4: Saving results")
print("=" * 70)

with open(os.path.join(OUT_DIR, 'all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=2, default=str)

# Summary table
rows = []
for exp_name, exp_results in all_results.items():
    if isinstance(exp_results, dict):
        for model_name, metrics in exp_results.items():
            if isinstance(metrics, dict) and 'f1' in metrics:
                row = {'experiment': exp_name, 'model': model_name}
                row.update(metrics)
                rows.append(row)

summary_df = pd.DataFrame(rows)
summary_df.to_csv(os.path.join(OUT_DIR, 'summary_table.csv'), index=False)
print(f"Saved summary: {len(summary_df)} rows")
print("\n" + summary_df[['experiment', 'model', 'accuracy', 'f1', 'precision', 'recall', 'fpr']].to_string())

# ============================================================
# 8. VERIFY CONTROLS
# ============================================================
print("\n" + "=" * 70)
print("VERIFICATION: Control Experiments")
print("=" * 70)
print(f"\n{'Model':<8} {'A-Random':<10} {'B-Session':<12} {'B2-Matched':<12} {'B3-SameDist':<12}")
print("-" * 60)
for m in ['LR', 'RF', 'XGB', 'MLP']:
    a_f1 = all_results['A_random_split'][m]['f1']
    b_f1 = all_results['B_session_shift'][m]['f1']
    b2_f1 = all_results['B2_matched_size_control'][m]['f1']
    b3_f1 = all_results['B3_same_dist_same_size'][m]['f1']
    print(f"{m:<8} {a_f1:<10.4f} {b_f1:<12.4f} {b2_f1:<12.4f} {b3_f1:<12.4f}")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUT_DIR}")
