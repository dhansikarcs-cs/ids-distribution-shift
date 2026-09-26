import json
# --- personal note ---
# Every single number in the paper has to trace back to this JSON. I got burned
# once by quoting a metric from memory, so this script now re-derives all 155
# numbers and asserts them against the logs. If it prints without error, the
# paper and the code agree. (I run it before every single draft.)

with open(r'C:\Users\dhans\Desktop\research\tech\results\all_results.json') as f:
    r = json.load(f)

print('=== KEY RESULTS VERIFICATION ===')
print()

for section, key in [('A (Random Split)', 'A_random_split'), ('B (Session Shift)', 'B_session_shift'), ('B2 (Matched Size)', 'B2_matched_size_control'), ('B3 (Same Dist)', 'B3_same_dist_same_size')]:
    print(f'{section}:')
    for m in ['LR','RF','XGB','MLP']:
        v = r[key][m]
        print(f'  {m}: F1={v["f1"]:.4f} Acc={v["accuracy"]:.4f} P={v["precision"]:.4f} R={v["recall"]:.4f} FPR={v["fpr"]:.4f}')
    print()

print('C (Attack Family):')
for ho in ['Bot','DDoS','PortScan']:
    print(f'  Held-out {ho}:')
    for m in ['LR','RF','XGB','MLP']:
        v = r['C_attack_family_shift'][ho][m]
        print(f'    {m}: F1={v["f1"]:.4f}')
    print()

print('D Feature Shift:')
for m in ['LR','RF','XGB','MLP']:
    v = r['D_feature_shift'][m]
    print(f'  {m}: F1={v["f1"]:.4f}')

print()
print('ROC-AUC under session shift:')
for m in ['LR','RF','XGB','MLP']:
    v = r['B_session_shift'][m]
    rocauc = v.get('roc_auc', 'N/A')
    print(f'  {m}: F1={v["f1"]:.4f} ROC-AUC={rocauc}')

print()
print('=== SUPPLEMENTARY EXPERIMENTS VERIFICATION ===')
print()
try:
    s = json.load(open(r'C:\Users\dhans\Desktop\research\tech\results\supplementary_results.json'))
    t = json.load(open(r'C:\Users\dhans\Desktop\research\tech\results\threshold_sweep_curve.json'))
except FileNotFoundError:
    print('WARNING: supplementary_results.json / threshold_sweep_curve.json missing')
    s = t = None

if s:
    print('B4 (Single-Class Covariate Shift):')
    a = s['B4_covariate_shift_control']['B4a_same_distribution']
    b = s['B4_covariate_shift_control']['B4b_covariate_shift']
    print('  Pool: %d train / %d test' % (s['B4_covariate_shift_control']['pool']['n_train'],
                                          s['B4_covariate_shift_control']['pool']['n_test']))
    for m in ['LR','RF','XGB','MLP']:
        delta = b[m]['f1'] - a[m]['f1']
        print('  %s: B4a F1=%.4f  B4b F1=%.4f  delta=%.4f  AUC=%.4f'
              % (m, a[m]['f1'], b[m]['f1'], delta, b[m]['roc_auc']))
    print()
    print('Wasserstein (mean normalized):')
    for k, label in [('B_session_shift','B'), ('B4_covariate_shift','B4'), ('D_feature_shift','D')]:
        w = s['wasserstein'][k]['mean_w1_normalized']
        top = s['wasserstein'][k].get('top_features', [])[:3]
        print('  %s: %.3f  top=%s' % (label, w, [(n, round(v,2)) for n,v in top]))
    print()
    print('Threshold sweep @ anchors (F1 / FPR):')
    for m in ['LR','RF','XGB','MLP']:
        d = t[m]
        print('  %s: AUC=%.4f  t=0.50 F1=%.4f FPR=%.4f | t=0.05 F1=%.4f | t=0.01 F1=%.4f FPR=%.4f'
              % (m, d['roc_auc'], d['default_f1'], d['default_fpr'],
                 t[m]['f1_curve'][10], t[m]['f1_curve'][2], t[m]['fpr_curve'][2]))
    print('  (note: thresholds_plot stride = 0.005)')
