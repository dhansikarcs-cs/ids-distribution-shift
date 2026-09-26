"""
ML-Based Network Intrusion Detection Robustness Under Distribution Shift
=========================================================================
Experiments: Random split, Temporal split, Attack-family shift, Feature shift
Models: Logistic Regression, Random Forest, XGBoost, MLP
Dataset: CICIDS2017 (Friday subset from HuggingFace mirror)
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'
# NOTE: point this at the CLEANED Friday sheet (save_data.py), not the raw
# download. the raw csv is a few million rows with malformed lines and the
# whole paper's numbers are tied to the cleaned + subsampled version.

# ============================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================
print("=" * 70)
print("PHASE 1: Loading and preprocessing data")
print("=" * 70)

df = pd.read_csv(DATA_PATH)
print(f"Raw shape: {df.shape}")

label_col = 'Label'
print(f"Label column: {label_col}")

# Replace infinities and drop NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)
before = len(df)
df.dropna(inplace=True)
print(f"Dropped {before - len(df)} rows with NaN/Inf. Remaining: {len(df):,}")

# Drop non-numeric columns except Label
# (timestamps and a few hex-text columns read as strings - they can't go in)
drop_cols = []
for c in df.columns:
    if c != label_col and df[c].dtype == 'object':
        drop_cols.append(c)

feature_cols = [c for c in df.columns if c not in drop_cols and c != label_col]
print(f"Features: {len(feature_cols)}")

# Subsample to manageable size
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

# Create labels AFTER subsampling
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

# Masks for temporal split - morning is Bot + some Benign, afternoon is
# DDoS + PortScan + the rest. the model trains on one and tests on the other.
is_ddos = (df[label_col] == 'DDoS').values
is_portscan = (df[label_col] == 'PortScan').values
is_bot = (df[label_col] == 'Bot').values
is_benign = (df[label_col] == 'BENIGN').values

print(f"\nTemporal split plan:")
print(f"  Morning session (Bot + Benign from morning): {(is_bot | is_benign).sum():,}")
print(f"  Afternoon session (DDoS + PortScan + Benign): {(is_ddos | is_portscan | is_benign).sum():,}")

# ============================================================
# 2. DEFINE MODELS
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
# 3. EVALUATION HELPER
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
# 4. RUN ALL EXPERIMENTS
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

# --- EXPERIMENT B: Temporal Split ---
print("\n" + "=" * 70)
print("EXPERIMENT B: Temporal Split (Morning -> Afternoon)")
print("=" * 70)
# Train on morning (Bot + Benign from morning), test on afternoon (DDoS + PortScan + their Benign)
# Approximate: morning = all Bot + ~50% Benign, afternoon = DDoS + PortScan + ~50% Benign
# Temporal: train on Bot+Benign morning, test on DDoS+PortScan+Benign afternoon
# This tests: can a model trained on one type of traffic generalize to different traffic?
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
print(f"  Morning (train): {morning_mask.sum():,} | Afternoon (test): {afternoon_mask_full.sum():,}")

all_results['B_temporal_split'] = run_experiment(
    'B - Temporal Split', X_tr_b, X_te_b, y_tr_b, y_te_b, 'binary'
)

# --- EXPERIMENT C: Attack-Family Shift ---
print("\n" + "=" * 70)
print("EXPERIMENT C: Attack-Family Shift (Leave-One-Out)")
print("=" * 70)

attack_classes = ['Bot', 'DDoS', 'PortScan']
results_c = {}

for held_out in attack_classes:
    print(f"\n  Held-out: {held_out}")
    train_mask = (df[label_col] != held_out).values
    test_attack_mask = (df[label_col] == held_out).values
    test_benign_mask = (df[label_col] == 'BENIGN').values

    n_attack = test_attack_mask.sum()
    benign_indices = np.where(test_benign_mask)[0]
    np.random.seed(42)
    selected = np.random.choice(benign_indices, size=min(n_attack, len(benign_indices)), replace=False)
    test_mask = test_attack_mask.copy()
    test_mask[selected] = True

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

np.random.seed(42)
X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)

# Control: no shift
all_results['D_no_shift_control'] = run_experiment(
    'D - No Shift (Control)', X_tr_d, X_te_d, y_tr_d, y_te_d, 'binary'
)

# Feature-shifted test set
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
# 5. MULTI-CLASS EXPERIMENT
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
# 6. SAVE RESULTS
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
# 7. GENERATE PLOTS
# ============================================================
print("\n" + "=" * 70)
print("PHASE 5: Generating plots")
print("=" * 70)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
model_colors = {'LR': '#e74c3c', 'RF': '#3498db', 'XGB': '#2ecc71', 'MLP': '#9b59b6'}

exp_keys = ['A_random_split', 'B_temporal_split', 'D_no_shift_control', 'D_feature_shift']
exp_labels = ['A: Random Split', 'B: Temporal Split', 'D: No Shift', 'D: Feature Shift']

def get_metric_val(ek, model, metric):
    data = all_results.get(ek, {})
    return data.get(model, {}).get(metric, 0)

# --- Plot 1: Performance comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric in zip(axes, ['accuracy', 'f1', 'recall']):
    x = np.arange(len(exp_keys))
    width = 0.2
    for i, (model, color) in enumerate(model_colors.items()):
        vals = [get_metric_val(ek, model, metric) for ek in exp_keys]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.upper()} Across Scenarios')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(exp_labels, rotation=20, ha='right', fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig1_performance.png")

# --- Plot 2: F1 Heatmap ---
fig, ax = plt.subplots(figsize=(10, 4))
heat_data = {}
for model in model_colors:
    heat_data[model] = [get_metric_val(ek, model, 'f1') for ek in exp_keys]

# Add attack-family shift results
for held_out in attack_classes:
    key = f'C_family_{held_out}'
    exp_keys_c = f'C_attack_family_shift'
    val = all_results.get(exp_keys_c, {}).get(held_out, {})
    for model in model_colors:
        heat_data[model].append(val.get(model, {}).get('f1', 0))
    exp_labels.append(f'C: {held_out} LOO')

heat_df = pd.DataFrame(heat_data, index=exp_labels)
sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax, linewidths=0.5)
ax.set_title('F1 Score: Models x Evaluation Scenarios')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig2_heatmap.png")

# --- Plot 3: Degradation from baseline ---
fig, ax = plt.subplots(figsize=(10, 5))
baseline_f1 = {m: get_metric_val('A_random_split', m, 'f1') for m in model_colors}
deg_labels = exp_labels[1:] + [f'C: {ho} LOO' for ho in attack_classes]
deg_keys = exp_keys[1:] + ['C_attack_family_shift'] * 3

x = np.arange(len(deg_labels))
width = 0.2
for i, (model, color) in enumerate(model_colors.items()):
    vals = []
    for j, ek in enumerate(deg_keys):
        if j >= len(exp_keys) - 1:
            held_out = attack_classes[j - (len(exp_keys) - 1)]
            v = all_results.get('C_attack_family_shift', {}).get(held_out, {}).get(model, {}).get('f1', 0)
        else:
            v = get_metric_val(ek, model, 'f1')
        deg = ((baseline_f1[model] - v) / baseline_f1[model]) * 100 if baseline_f1[model] > 0 else 0
        vals.append(max(deg, 0))
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)

ax.set_ylabel('F1 Degradation from Random Split Baseline (%)')
ax.set_title('Performance Degradation Under Distribution Shift')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(deg_labels, rotation=25, ha='right', fontsize=8)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_degradation.png")

# --- Plot 4: Inference Time ---
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(exp_keys))
width = 0.2
for i, (model, color) in enumerate(model_colors.items()):
    vals = [get_metric_val(ek, model, 'inference_time_ms') for ek in exp_keys]
    ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
ax.set_ylabel('Inference Time (ms/sample)')
ax.set_title('Computational Cost Across Scenarios')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(exp_labels[:4], rotation=20, ha='right', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_inference_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_inference_time.png")

# --- Plot 5: Multi-class ROC ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (model_name, model) in enumerate(get_models().items()):
    scaler = StandardScaler()
    X_tr = np.nan_to_num(scaler.fit_transform(X_tr_mc), nan=0.0)
    X_te = np.nan_to_num(scaler.transform(X_te_mc), nan=0.0)
    model.fit(X_tr, y_tr_mc)
    y_prob = model.predict_proba(X_te)
    ax = axes.flatten()[idx]
    for i, cn in enumerate(class_names):
        y_true_bin = (y_te_mc == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
        auc = roc_auc_score(y_true_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=f'{cn} (AUC={auc:.3f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{model_name} - Multi-class ROC')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_multiclass_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_multiclass_roc.png")

# --- Plot 6: Attack-family shift detail ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, held_out in enumerate(attack_classes):
    results = all_results['C_attack_family_shift'].get(held_out, {})
    ax = axes[idx]
    models = list(model_colors.keys())
    f1_vals = [results.get(m, {}).get('f1', 0) for m in models]
    colors = [model_colors[m] for m in models]
    bars = ax.bar(models, f1_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_title(f'When {held_out} Unseen in Training')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.1)
plt.suptitle('Attack-Family Shift: F1 When Specific Attack Type Excluded from Training', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_family_shift.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_family_shift.png")

# --- Plot 7: Confusion matrices for best model under random vs temporal ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
best_model_name = 'RF'
scaler = StandardScaler()

# Random split
X_tr = np.nan_to_num(scaler.fit_transform(X_tr_a), nan=0.0)
X_te = np.nan_to_num(scaler.transform(X_te_a), nan=0.0)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_tr, y_tr_a)
cm_a = confusion_matrix(y_te_a, rf.predict(X_te))
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Random Split (Baseline)')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Temporal split
scaler2 = StandardScaler()
X_tr2 = np.nan_to_num(scaler2.fit_transform(X_tr_b), nan=0.0)
X_te2 = np.nan_to_num(scaler2.transform(X_te_b), nan=0.0)
rf2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf2.fit(X_tr2, y_tr_b)
cm_b = confusion_matrix(y_te_b, rf2.predict(X_te2))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False)
axes[1].set_title('Temporal Split')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.suptitle('Confusion Matrices: Random Forest Under Random vs Temporal Split', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig7_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig7_confusion_matrices.png")

print("\n" + "=" * 70)
print("ALL EXPERIMENTS COMPLETE")
print("=" * 70)
print(f"Results saved to: {OUT_DIR}")
