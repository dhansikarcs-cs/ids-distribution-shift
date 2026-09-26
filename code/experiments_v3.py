"""
SUPPLEMENTARY EXPERIMENTS for reviewer z-round z (journal readiness).
1. Experiment B4: single-class covariate-shift control (isolates unseen-class
   shift from pure covariate shift). DDoS present in BOTH train and test;
   only the test feature distribution shifts.
2. Threshold sweep on Experiment B test set: quantify how much F1 can be
   recovered by tuning the decision threshold away from 0.5.
3. Wasserstein distance quantification of the B and D shifts.
Replicates the exact preprocessing of experiments_v2.py so pools align.
"""

# --- personal note (July-Aug 2026) ---
# This file grew organically: B4 came after B2, and the threshold sweep came
# after I saw Random Forest still had ROC-AUC 0.81 while its F1 sat at exactly
# zero. That gap bugged me until I swept the threshold - hence experiments_v4.
# The Wasserstein numbers here also answer the obvious reviewer question,
# "how big is the shift, really?" If you re-run this, the pools MUST match
# experiments_v2.py, otherwise B4 stops being a control and becomes nothing.

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
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier

try:
    from scipy.stats import wasserstein_distance
except ImportError:
    def wasserstein_distance(a, b):
        a = np.sort(a); b = np.sort(b)
        n, m = len(a), len(b)
        # linear-interpolation style 1D W1 over common quantile grid
        q = np.linspace(0, 1, max(n, m))
        qa = np.quantile(a, q, method='linear')
        qb = np.quantile(b, q, method='linear')
        return float(np.mean(np.abs(qa - qb)))

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'
DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'
# NOTE: full run is ~90 min on my laptop (RF + XGB on the 50K pool). if you
# only need the numbers, read results/supplementary_results.json instead.

# ============================================================
# 0. IDENTICAL PREPROCESSING TO experiments_v2.py
# ============================================================
df = pd.read_csv(DATA_PATH)
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
X = df[feature_cols].values.astype(np.float32)
y_bin = df['binary_label'].values
is_ddos = (df[label_col] == 'DDoS').values
is_benign = (df[label_col] == 'BENIGN').values

# Rebuild Experiment B pools exactly as in experiments_v2.py
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

# Experiment D test pool (same random split as A)
_, X_te_d_pool, _, y_te_d_pool = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

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

def fit_predict_proba(model, X_train, y_train, X_test):
    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_train), nan=0.0)
    X_te = np.nan_to_num(scaler.transform(X_test), nan=0.0)
    model.fit(X_tr, y_train)
    p = model.predict_proba(X_te)
    if p.shape[1] == 2:
        return p[:, 1]
    return p

def threshold_sweep(y_true, proba):
    """Vectorized F1 over decision thresholds. Returns best threshold + max F1."""
    y_true = np.asarray(y_true)
    order = np.argsort(-proba, kind='stable')
    ps = proba[order]
    ys = y_true[order]
    cum_attack = np.cumsum(ys)
    n_pos = cum_attack[-1]
    n = len(ys)
    thresholds = np.arange(0.001, 1.0, 0.001)
    # k = number of positives predicted for each threshold
    idx = np.searchsorted(-ps, -thresholds, side='right')
    idx = np.clip(idx, 1, n)
    tp = cum_attack[idx - 1]
    fp = idx - tp
    precision = tp / np.maximum(idx, 1)
    recall = tp / max(n_pos, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best_i = int(np.argmax(f1))
    return {
        'best_threshold': float(thresholds[best_i]),
        'max_f1': float(f1[best_i]),
        'f1_at_0.50': float(f1[int(np.searchsorted(thresholds, 0.5)) - 1]),
        'proba_frac_above_05': float((proba > 0.5).mean()),
    }

# ============================================================
# 1. EXPERIMENT B4: SINGLE-CLASS COVARIATE-SHIFT CONTROL
# ============================================================
print("\n" + "=" * 70)
print("EXPERIMENT B4: Single-class covariate shift (DDoS only)")
print("=" * 70)

b4_mask = is_ddos | is_benign
idx_b4 = np.where(b4_mask)[0]
X_b4 = X[idx_b4]
y_b4 = (df[label_col].values[idx_b4] == 'DDoS').astype(int)

X_tr_b4, X_te_b4, y_tr_b4, y_te_b4 = train_test_split(
    X_b4, y_b4, test_size=0.2, random_state=42, stratify=y_b4
)
print(f"B4 pool: {len(X_b4):,} (DDoS + BENIGN) | train: {len(X_tr_b4):,} test: {len(X_te_b4):,}")

# B4a: no-shift control (same distribution train/test)
print("\n-- B4a: same-distribution control (DDoS in train and test) --")
b4a = {}
for name, model in get_models().items():
    t0 = time.time()
    proba = fit_predict_proba(model, X_tr_b4, y_tr_b4, X_te_b4)
    pred = (proba >= 0.5).astype(int)
    b4a[name] = {
        'f1': float(f1_score(y_te_b4, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_te_b4, proba)),
        'proba_frac_above_05': float((proba > 0.5).mean()),
        'train_time_s': float(time.time() - t0),
    }
    print(f"  {name}: F1={b4a[name]['f1']:.4f} AUC={b4a[name]['roc_auc']:.4f}")

# B4b: covariate shift - same attack type, features perturbed on test DDoS
np.random.seed(42)
X_te_b4_shift = X_te_b4.copy()
# make the test DDoS look different: 2-5x longer flows, ~1.5x more packets.
# train features stay original, so only the feature distribution moves.
dur_col = feature_cols.index('Flow Duration') if 'Flow Duration' in feature_cols else 1
shift_factors = np.random.uniform(2.0, 5.0, size=len(X_te_b4_shift))
X_te_b4_shift[:, dur_col] = X_te_b4_shift[:, dur_col] * shift_factors
for col_name in ['Total Fwd Packets', 'Total Backward Packets']:
    if col_name in feature_cols:
        ci = feature_cols.index(col_name)
        noise = np.random.normal(1.5, 0.5, size=len(X_te_b4_shift))
        X_te_b4_shift[:, ci] = X_te_b4_shift[:, ci] * noise
X_te_b4_shift = np.clip(X_te_b4_shift, -1e6, 1e6)

print("\n-- B4b: covariate shift (DDoS present, features perturbed) --")
b4b = {}
for name, model in get_models().items():
    t0 = time.time()
    proba = fit_predict_proba(model, X_tr_b4, y_tr_b4, X_te_b4_shift)
    pred = (proba >= 0.5).astype(int)
    b4b[name] = {
        'f1': float(f1_score(y_te_b4, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_te_b4, proba)),
        'proba_frac_above_05': float((proba > 0.5).mean()),
        'train_time_s': float(time.time() - t0),
    }
    print(f"  {name}: F1={b4b[name]['f1']:.4f} AUC={b4b[name]['roc_auc']:.4f}")

# ============================================================
# 2. THRESHOLD SWEEP ON EXPERIMENT B TEST SET
# ============================================================
print("\n" + "=" * 70)
print("THRESHOLD SWEEP: Experiment B test set")
print("=" * 70)

sweep = {}
for name, model in get_models().items():
    t0 = time.time()
    proba = fit_predict_proba(model, X_tr_b, y_tr_b, X_te_b)
    sw = threshold_sweep(y_te_b, proba)
    sw['roc_auc'] = float(roc_auc_score(y_te_b, proba))
    sw['time_s'] = float(time.time() - t0)
    sweep[name] = sw
    print(f"  {name}: default F1={sw['f1_at_0.50']:.4f} -> max F1={sw['max_f1']:.4f} "
          f"at t={sw['best_threshold']:.3f} (AUC={sw['roc_auc']:.4f})")

# ============================================================
# 3. WASSERSTEIN DISTANCE QUANTIFICATION
# ============================================================
print("\n" + "=" * 70)
print("WASSERSTEIN DISTANCES")
print("=" * 70)

def wasserstein_profile(a, b, rng_seed=0, n_max=25000):
    rng = np.random.RandomState(rng_seed)
    n = min(len(a), n_max)
    ia = rng.choice(len(a), size=n, replace=False)
    ib = rng.choice(len(b), size=n, replace=False)
    a_s, b_s = a[ia], b[ib]
    norms = []
    for f in range(a_s.shape[1]):
        x = a_s[:, f]
        y = b_s[:, f]
        pooled_std = np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)
        if pooled_std < 1e-9:
            continue
        norms.append(wasserstein_distance(x, y) / pooled_std)
    norms = np.array(norms)
    return {
        'mean_w1_normalized': float(norms.mean()),
        'median_w1_normalized': float(np.median(norms)),
        'n_features': int(len(norms)),
    }

def top_shifted_features(a, b, k=5, rng_seed=0, n_max=25000):
    rng = np.random.RandomState(rng_seed)
    n = min(len(a), n_max)
    ia = rng.choice(len(a), size=n, replace=False)
    ib = rng.choice(len(b), size=n, replace=False)
    a_s, b_s = a[ia], b[ib]
    out = []
    for f in range(a_s.shape[1]):
        x = a_s[:, f]
        y = b_s[:, f]
        pooled_std = np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)
        if pooled_std < 1e-9:
            continue
        out.append((feature_cols[f], float(wasserstein_distance(x, y) / pooled_std)))
    out.sort(key=lambda t: -t[1])
    return out[:k]

# B shift: morning-session training features vs afternoon-session test features
w_b = wasserstein_profile(X_tr_b, X_te_b, rng_seed=42)
top_b = top_shifted_features(X_tr_b, X_te_b, k=5)
print(f"B shift: mean W1/norm = {w_b['mean_w1_normalized']:.3f}")
print("  top shifted features:", [(n, round(v, 2)) for n, v in top_b])

# D shift: unshifted test features vs shifted test features
np.random.seed(42)
X_shifted = X_te_d_pool.copy()
shift_factors = np.random.uniform(2.0, 5.0, size=len(X_shifted))
X_shifted[:, dur_col] = X_shifted[:, dur_col] * shift_factors
for col_name in ['Total Fwd Packets', 'Total Backward Packets']:
    if col_name in feature_cols:
        ci = feature_cols.index(col_name)
        noise = np.random.normal(1.5, 0.5, size=len(X_shifted))
        X_shifted[:, ci] = X_shifted[:, ci] * noise
X_shifted = np.clip(X_shifted, -1e6, 1e6)

w_d = wasserstein_profile(X_te_d_pool, X_shifted, rng_seed=42)
top_d = top_shifted_features(X_te_d_pool, X_shifted, k=5)
print(f"D shift: mean W1/norm = {w_d['mean_w1_normalized']:.3f}")
print("  top shifted features:", [(n, round(v, 2)) for n, v in top_d])

# B4 shift: B4 test unshifted vs B4 test shifted
w_b4 = wasserstein_profile(X_te_b4, X_te_b4_shift, rng_seed=42)
print(f"B4 shift: mean W1/norm = {w_b4['mean_w1_normalized']:.3f}")

# ============================================================
# 4. SAVE
# ============================================================
supp = {
    'B4_covariate_shift_control': {
        'pool': {'n': int(len(X_b4)), 'n_train': int(len(X_tr_b4)), 'n_test': int(len(X_te_b4))},
        'B4a_same_distribution': b4a,
        'B4b_covariate_shift': b4b,
    },
    'threshold_sweep_B': sweep,
    'wasserstein': {
        'B_session_shift': {**w_b, 'top_features': top_b},
        'D_feature_shift': {**w_d, 'top_features': top_d},
        'B4_covariate_shift': {**w_b4},
    },
}

with open(os.path.join(OUT_DIR, 'supplementary_results.json'), 'w') as f:
    json.dump(supp, f, indent=2, default=str)
print(f"\nSaved: {os.path.join(OUT_DIR, 'supplementary_results.json')}")
print("COMPLETE")