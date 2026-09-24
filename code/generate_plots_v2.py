"""Regenerate all plots with corrected experiment results (v2)"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'

with open(os.path.join(OUT_DIR, 'all_results.json'), 'r') as f:
    all_results = json.load(f)

DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'
df = pd.read_csv(DATA_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

MAX_ROWS = 50000
label_col = 'Label'
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

le = LabelEncoder()
y_multi = le.fit_transform(df[label_col])
class_names = list(le.classes_)

is_ddos = (df[label_col] == 'DDoS').values
is_portscan = (df[label_col] == 'PortScan').values
is_bot = (df[label_col] == 'Bot').values
is_benign = (df[label_col] == 'BENIGN').values

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

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
model_colors = {'LR': '#e74c3c', 'RF': '#3498db', 'XGB': '#2ecc71', 'MLP': '#9b59b6'}
attack_classes = ['Bot', 'DDoS', 'PortScan']

# Updated experiment keys and labels
A_KEY = 'A_random_split'
B_KEY = 'B_session_shift'
B2_KEY = 'B2_matched_size_control'
B3_KEY = 'B3_same_dist_same_size'
C_KEY = 'C_attack_family_shift'
D_CTRL_KEY = 'D_no_shift_control'
D_FEAT_KEY = 'D_feature_shift'

# ============================================================
# FIGURE 1: Main 4-scenario comparison
# ============================================================
print("Generating fig1...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
exp_keys = [A_KEY, B_KEY, B2_KEY, D_CTRL_KEY]
exp_labels = ['A: Random\nSplit', 'B: Session\nShift', 'B2: Matched\nSize Control', 'D: No Shift\n(Control)']

for ax, metric in zip(axes, ['accuracy', 'f1', 'recall']):
    x = np.arange(len(exp_keys))
    width = 0.2
    for i, (model, color) in enumerate(model_colors.items()):
        vals = [all_results.get(ek, {}).get(model, {}).get(metric, 0) for ek in exp_keys]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11)
    ax.set_title(f'{metric.upper()}', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(exp_labels, fontsize=9)
    ax.legend(fontsize=9, loc='lower left')
    ax.set_ylim(0, 1.1)
plt.suptitle('Model Performance Across Evaluation Scenarios', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 2: F1 Heatmap (full)
# ============================================================
print("Generating fig2...")
all_exp_labels = ['A: Random', 'B: Session', 'B2: Matched Size', 'D: No Shift', 'D: Feature Shift']
all_exp_keys = [A_KEY, B_KEY, B2_KEY, D_CTRL_KEY, D_FEAT_KEY]

heat_data = {}
for model in model_colors:
    heat_data[model] = [all_results.get(ek, {}).get(model, {}).get('f1', 0) for ek in all_exp_keys]

for held_out in attack_classes:
    for model in model_colors:
        v = all_results.get(C_KEY, {}).get(held_out, {}).get(model, {}).get('f1', 0)
        heat_data[model].append(v)
    all_exp_labels.append(f'C: {held_out} LOO')

fig, ax = plt.subplots(figsize=(12, 5))
heat_df = pd.DataFrame(heat_data, index=all_exp_labels)
sns.heatmap(heat_df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1, ax=ax, linewidths=0.5, annot_kws={'size': 9})
ax.set_title('F1 Score: Models x Evaluation Scenarios', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 3: KEY ISOLATION FIGURE (B vs B2 vs A)
# ============================================================
print("Generating fig3 (KEY FIGURE)...")
fig, ax = plt.subplots(figsize=(10, 6))
models = list(model_colors.keys())
x = np.arange(len(models))
width = 0.25

scenarios = [
    (A_KEY, 'Random Split (254K)', '#2ecc71'),
    (B_KEY, 'Session Shift (13K)', '#e74c3c'),
    (B2_KEY, 'Matched Size (13K)', '#f39c12'),
]

for i, (ek, label, color) in enumerate(scenarios):
    vals = [all_results.get(ek, {}).get(m, {}).get('f1', 0) for m in models]
    bars = ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Isolating the Effect of Distribution Shift\n(B vs B2: Same 13K Train Size, Different Distribution)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(models, fontsize=11)
ax.legend(fontsize=10, loc='lower left')
ax.set_ylim(0, 1.15)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_key_isolation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 4: Degradation from baseline
# ============================================================
print("Generating fig4...")
baseline = {m: all_results[A_KEY][m]['f1'] for m in model_colors}

deg_pairs = [
    ('B: Session\nShift', all_results.get(B_KEY, {})),
    ('B2: Matched\nSize', all_results.get(B2_KEY, {})),
    ('D: Feature\nShift', all_results.get(D_FEAT_KEY, {})),
]
for ho in attack_classes:
    deg_pairs.append((f'C: {ho}\nLOO', all_results.get(C_KEY, {}).get(ho, {})))

deg_labels = [p[0] for p in deg_pairs]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(deg_labels))
width = 0.2
for i, (model, color) in enumerate(model_colors.items()):
    vals = []
    for _, model_results in deg_pairs:
        v = model_results.get(model, {}).get('f1', 0)
        deg = max(((baseline[model] - v) / baseline[model]) * 100, 0) if baseline[model] > 0 else 0
        vals.append(deg)
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_ylabel('F1 Degradation from Baseline (%)', fontsize=11)
ax.set_title('Performance Degradation Across Shift Types', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(deg_labels, fontsize=9)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 5: Inference time
# ============================================================
print("Generating fig5...")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(exp_keys))
width = 0.2
for i, (model, color) in enumerate(model_colors.items()):
    vals = [all_results.get(ek, {}).get(model, {}).get('inference_time_ms', 0) for ek in exp_keys]
    ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
ax.set_ylabel('Inference Time (ms/sample)', fontsize=11)
ax.set_title('Computational Cost', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(exp_labels, fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_inference_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 6: Multi-class ROC
# ============================================================
print("Generating fig6...")
X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(X, y_multi, test_size=0.2, random_state=42, stratify=y_multi)
scaler = StandardScaler()
X_tr_s = np.nan_to_num(scaler.fit_transform(X_tr_mc), nan=0.0)
X_te_s = np.nan_to_num(scaler.transform(X_te_mc), nan=0.0)

models_mc = {
    'LR': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    'RF': RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1),
    'XGB': XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=50, random_state=42, early_stopping=True),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for idx, (model_name, model) in enumerate(models_mc.items()):
    model.fit(X_tr_s, y_tr_mc)
    y_prob = model.predict_proba(X_te_s)
    ax = axes.flatten()[idx]
    for i, cn in enumerate(class_names):
        y_true_bin = (y_te_mc == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_prob[:, i])
        auc = roc_auc_score(y_true_bin, y_prob[:, i])
        ax.plot(fpr, tpr, label=f'{cn} (AUC={auc:.3f})', linewidth=1.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR')
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower right')
plt.suptitle('Multi-Class ROC Curves (Random Split)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_multiclass_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 7: Attack-family shift
# ============================================================
print("Generating fig7...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for idx, held_out in enumerate(attack_classes):
    results = all_results[C_KEY].get(held_out, {})
    ax = axes[idx]
    models_l = list(model_colors.keys())
    f1_vals = [results.get(m, {}).get('f1', 0) for m in models_l]
    colors = [model_colors[m] for m in models_l]
    bars = ax.bar(models_l, f1_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_title(f'When {held_out} Unseen', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score')
    ax.set_ylim(0, 1.1)
plt.suptitle('Attack-Family Shift: Leave-One-Out F1', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig7_family_shift.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 8: Confusion matrices (Random Forest: A vs B vs B2)
# ============================================================
print("Generating fig8...")
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# A: Random split
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X, y_bin, test_size=0.2, random_state=42, stratify=y_bin)
sc1 = StandardScaler()
rf_a = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
rf_a.fit(np.nan_to_num(sc1.fit_transform(X_tr_a)), y_tr_a)
cm_a = confusion_matrix(y_te_a, rf_a.predict(np.nan_to_num(sc1.transform(X_te_a))))
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'], annot_kws={'size': 14})
axes[0].set_title('A: Random Split', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# B: Session shift
X_tr_b = X[morning_mask]
y_tr_b = y_bin[morning_mask]
X_te_b = X[afternoon_mask_full]
y_te_b = y_bin[afternoon_mask_full]
sc2 = StandardScaler()
rf_b = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
rf_b.fit(np.nan_to_num(sc2.fit_transform(X_tr_b)), y_tr_b)
cm_b = confusion_matrix(y_te_b, rf_b.predict(np.nan_to_num(sc2.transform(X_te_b))))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Reds', ax=axes[1], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'], annot_kws={'size': 14})
axes[1].set_title('B: Session Shift', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

# B2: Matched-size control (correct: subsample of A's training pool)
np.random.seed(123)
train_idx_b2 = np.random.choice(len(X_tr_a), size=morning_mask.sum(), replace=False)
sc3 = StandardScaler()
rf_b2 = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
rf_b2.fit(np.nan_to_num(sc3.fit_transform(X_tr_a[train_idx_b2])), y_tr_a[train_idx_b2])
cm_b2 = confusion_matrix(y_te_a, rf_b2.predict(np.nan_to_num(sc3.transform(X_te_a))))
sns.heatmap(cm_b2, annot=True, fmt='d', cmap='Greens', ax=axes[2], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'], annot_kws={'size': 14})
axes[2].set_title('B2: Matched Size Control', fontsize=12, fontweight='bold')
axes[2].set_ylabel('True')
axes[2].set_xlabel('Predicted')

plt.suptitle('Random Forest: Distribution Shift vs Train Size\n(B = same train size, different distribution; B2 = same size, same distribution)', fontsize=13, fontweight='bold', y=1.06)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig8_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 9: Radar chart (A vs B vs B2)
# ============================================================
print("Generating fig9...")
metrics_to_radar = ['accuracy', 'f1', 'precision', 'recall']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
exp_configs = [
    (A_KEY, 'A: Random Split'),
    (B_KEY, 'B: Session Shift'),
    (B2_KEY, 'B2: Matched Size'),
]

for ax, (ek, title) in zip(axes, exp_configs):
    angles = np.linspace(0, 2 * np.pi, len(metrics_to_radar), endpoint=False).tolist()
    angles += angles[:1]
    for model, color in model_colors.items():
        vals = [all_results.get(ek, {}).get(model, {}).get(m, 0) for m in metrics_to_radar]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=color)
        ax.fill(angles, vals, alpha=0.1, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.title() for m in metrics_to_radar])
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')

plt.suptitle('Model Performance Radar: Distribution Shift Effect', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig9_radar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 10: ROC-AUC comparison
# ============================================================
print("Generating fig10...")
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(exp_keys))
width = 0.2
for i, (model, color) in enumerate(model_colors.items()):
    vals = [all_results.get(ek, {}).get(model, {}).get('roc_auc', 0) for ek in exp_keys]
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('ROC-AUC', fontsize=11)
ax.set_title('ROC-AUC Across Scenarios\n(F1 can collapse while ROC-AUC remains partially informative)', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(exp_labels, fontsize=9)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig10_rocauc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

print("\nAll plots regenerated!")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")
