"""Generate all remaining plots from saved results"""

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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'

with open(os.path.join(OUT_DIR, 'all_results.json'), 'r') as f:
    all_results = json.load(f)

DATA_PATH = r'C:\Users\dhans\Desktop\research\tech\cicids2017_friday.csv'
df = pd.read_csv(DATA_PATH)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Subsample same as experiments
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

is_ddos = (df[label_col] == 'DDoS').values
is_portscan = (df[label_col] == 'PortScan').values
is_bot = (df[label_col] == 'Bot').values
is_benign = (df[label_col] == 'BENIGN').values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_multi = le.fit_transform(df[label_col])
class_names = list(le.classes_)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
model_colors = {'LR': '#e74c3c', 'RF': '#3498db', 'XGB': '#2ecc71', 'MLP': '#9b59b6'}

# ---- PLOT 3: Performance Degradation (fixed) ----
print("Generating fig3...")
# Only use the 4 main experiments (no family shift here)
exp_keys_main = ['A_random_split', 'B_temporal_split', 'D_no_shift_control', 'D_feature_shift']
exp_labels_main = ['A: Random\nSplit', 'B: Temporal\nSplit', 'D: No Shift\n(Control)', 'D: Feature\nShift']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric in zip(axes, ['accuracy', 'f1', 'recall']):
    x = np.arange(len(exp_keys_main))
    width = 0.2
    for i, (model, color) in enumerate(model_colors.items()):
        vals = [all_results.get(ek, {}).get(model, {}).get(metric, 0) for ek in exp_keys_main]
        ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.upper()} Across Scenarios')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(exp_labels_main, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_performance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig3_performance.png")

# ---- PLOT 4: Degradation Rate ----
print("Generating fig4...")
baseline = {m: all_results['A_random_split'][m]['f1'] for m in model_colors}

deg_labels = ['B: Temporal', 'D: Feature Shift']
deg_keys = ['B_temporal_split', 'D_feature_shift']

x = np.arange(len(deg_labels))
width = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
for i, (model, color) in enumerate(model_colors.items()):
    vals = []
    for ek in deg_keys:
        f1_val = all_results.get(ek, {}).get(model, {}).get('f1', 0)
        deg = ((baseline[model] - f1_val) / baseline[model]) * 100 if baseline[model] > 0 else 0
        vals.append(max(deg, 0))
    bars = ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
    for bar, val in zip(bars, vals):
        if val > 1:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_ylabel('F1 Degradation from Baseline (%)')
ax.set_title('Performance Degradation Under Distribution Shift')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(deg_labels, fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_degradation.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig4_degradation.png")

# ---- PLOT 5: Inference Time ----
print("Generating fig5...")
x = np.arange(len(exp_keys_main))
width = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
for i, (model, color) in enumerate(model_colors.items()):
    vals = [all_results.get(ek, {}).get(model, {}).get('inference_time_ms', 0) for ek in exp_keys_main]
    ax.bar(x + i * width, vals, width, label=model, color=color, alpha=0.85)
ax.set_ylabel('Inference Time (ms/sample)')
ax.set_title('Computational Cost Across Scenarios')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(exp_labels_main, fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_inference_time.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig5_inference_time.png")

# ---- PLOT 6: Multi-class ROC ----
print("Generating fig6...")
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

X_tr_mc, X_te_mc, y_tr_mc, y_te_mc = train_test_split(
    X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
)
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
    ax.set_title(f'{model_name} - Multi-class ROC')
    ax.legend(fontsize=7, loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig6_multiclass_roc.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig6_multiclass_roc.png")

# ---- PLOT 7: Attack-family shift ----
print("Generating fig7...")
attack_classes = ['Bot', 'DDoS', 'PortScan']
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
plt.savefig(os.path.join(OUT_DIR, 'fig7_family_shift.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig7_family_shift.png")

# ---- PLOT 8: Confusion Matrices ----
print("Generating fig8...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random split
X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y_bin
)
scaler1 = StandardScaler()
X_tr1 = np.nan_to_num(scaler1.fit_transform(X_tr_a), nan=0.0)
X_te1 = np.nan_to_num(scaler1.transform(X_te_a), nan=0.0)
rf1 = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
rf1.fit(X_tr1, y_tr_a)
cm_a = confusion_matrix(y_te_a, rf1.predict(X_te1))
sns.heatmap(cm_a, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[0].set_title('Random Split (Baseline)')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

# Temporal split
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

X_tr_b = X[morning_mask]
y_tr_b = y_bin[morning_mask]
X_te_b = X[afternoon_mask_full]
y_te_b = y_bin[afternoon_mask_full]

scaler2 = StandardScaler()
X_tr2 = np.nan_to_num(scaler2.fit_transform(X_tr_b), nan=0.0)
X_te2 = np.nan_to_num(scaler2.transform(X_te_b), nan=0.0)
rf2 = RandomForestClassifier(n_estimators=80, max_depth=15, random_state=42, n_jobs=-1)
rf2.fit(X_tr2, y_tr_b)
cm_b = confusion_matrix(y_te_b, rf2.predict(X_te2))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Oranges', ax=axes[1], cbar=False,
            xticklabels=['Benign', 'Attack'], yticklabels=['Benign', 'Attack'])
axes[1].set_title('Temporal Split')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.suptitle('Random Forest Confusion Matrices: Random vs Temporal Split', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig8_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig8_confusion_matrices.png")

# ---- PLOT 9: Radar/Spider chart summary ----
print("Generating fig9...")
metrics_to_radar = ['accuracy', 'f1', 'precision', 'recall']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
exp_configs = [
    ('A_random_split', 'Random Split'),
    ('B_temporal_split', 'Temporal Split'),
    ('D_feature_shift', 'Feature Shift'),
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
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')

plt.suptitle('Model Performance Radar: Random Split vs Distribution Shift', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig9_radar.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: fig9_radar.png")

print("\nAll plots generated successfully!")
print(f"Output directory: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f} ({size:.0f} KB)")
