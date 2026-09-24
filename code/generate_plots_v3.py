"""Generate new supplementary figures (fig11, fig12) for the journal revision:
   fig11: mean normalized Wasserstein distance vs F1 drop (B, B4, D)
   fig12: F1 vs decision threshold on the Experiment B test set
"""

import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUT_DIR = r'C:\Users\dhans\Desktop\research\tech\results'

with open(os.path.join(OUT_DIR, 'all_results.json'), 'r') as f:
    all_results = json.load(f)
with open(os.path.join(OUT_DIR, 'supplementary_results.json'), 'r') as f:
    supp = json.load(f)
with open(os.path.join(OUT_DIR, 'threshold_sweep_curve.json'), 'r') as f:
    sweep = json.load(f)

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
model_colors = {'LR': '#e74c3c', 'RF': '#3498db', 'XGB': '#2ecc71', 'MLP': '#9b59b6'}

A = 'A_random_split'
B = 'B_session_shift'
B2 = 'B2_matched_size_control'
D_C = 'D_no_shift_control'
D_F = 'D_feature_shift'

# ============================================================
# FIGURE 11: Wasserstein distance vs F1 drop
# ============================================================
print("Generating fig11...")
w_b = supp['wasserstein']['B_session_shift']['mean_w1_normalized']
w_b4 = supp['wasserstein']['B4_covariate_shift']['mean_w1_normalized']
w_d = supp['wasserstein']['D_feature_shift']['mean_w1_normalized']

models = ['LR', 'RF', 'XGB', 'MLP']

# F1 drop per model per shift (baseline = same-distribution control of that experiment)
drops = {
    'B': {m: all_results[A][m]['f1'] - all_results[B][m]['f1'] for m in models},
    'B4': {m: supp['B4_covariate_shift_control']['B4a_same_distribution'][m]['f1']
              - supp['B4_covariate_shift_control']['B4b_covariate_shift'][m]['f1'] for m in models},
    'D': {m: all_results[D_C][m]['f1'] - all_results[D_F][m]['f1'] for m in models},
}
ws = {'B': w_b, 'B4': w_b4, 'D': w_d}

fig, ax = plt.subplots(figsize=(9, 6))
for shift, w in ws.items():
    for m in models:
        ax.scatter(w, drops[shift][m], s=140, color=model_colors[m],
                   edgecolor='k', linewidth=0.7, zorder=3)
        ax.annotate(m, (w, drops[shift][m]), textcoords="offset points",
                    xytext=(6, 2), fontsize=8)

for shift, w in ws.items():
    ax.axvline(w, color='gray', linestyle=':', alpha=0.5)
    ax.text(w, 1.02, shift, ha='center', va='bottom', fontsize=10, fontweight='bold', color='gray')

ax.set_xlabel('Mean Normalized Wasserstein Distance (per-feature, pooled-SD scaled)')
ax.set_ylabel('F1 Drop from No-Shift Baseline')
ax.set_title('Distribution Shift Magnitude vs Detection Degradation\n'
             '(B involves unseen attack classes; B4 and D are seen-class covariate shifts)',
             fontsize=12, fontweight='bold')
ax.set_ylim(-0.02, 1.1)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig11_wasserstein_vs_f1drop.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

# ============================================================
# FIGURE 12: Threshold sweep (F1 vs threshold) on B test set
# ============================================================
print("Generating fig12...")
fig, ax = plt.subplots(figsize=(9, 6))
for m in models:
    d = sweep[m]
    ax.plot(d['thresholds_plot'], d['f1_curve'], color=model_colors[m], linewidth=2, label=m)
ax.axvline(0.5, color='black', linestyle='--', alpha=0.6)
ax.text(0.52, 0.95 * ax.get_ylim()[1] if ax.get_ylim()[1] > 0.5 else 0.9,
        'default threshold (0.50)', fontsize=9, color='black', rotation=90, va='top')

ax.set_xlabel('Decision Threshold (attack probability)')
ax.set_ylabel('F1 Score')
ax.set_title('Threshold Sweep on the Shifted Test Set (Experiment B)\n'
             'Tree models recover a large fraction of F1 only at far-below-default thresholds',
             fontsize=12, fontweight='bold')
ax.set_xscale('log')
ax.set_xlim(0.001, 1.0)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig12_threshold_sweep.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved")

print("\nDone.")
for f in ['fig11_wasserstein_vs_f1drop.png', 'fig12_threshold_sweep.png']:
    p = os.path.join(OUT_DIR, f)
    print(f"  {f} ({os.path.getsize(p)/1024:.0f} KB)")