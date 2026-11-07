"""Verify every number quoted in paper/paper.md against the saved results.

The paper makes quantitative claims in its tables and prose. This script loads
the experiment outputs, recomputes the derived quantities (drops, shares,
ratios), and asserts that each value rounded the way the paper rounds it is
still correct. It exits non-zero on any mismatch, so it can be run before a
release.

Usage:
    python code/verify_numbers.py
"""

import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')

MODELS = ('LR', 'RF', 'XGB', 'MLP')

_passed = 0
_failures = []


def load(name):
    path = os.path.join(RES, name)
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def section(title):
    print('\n== %s ==' % title)


def check(condition, message):
    global _passed
    if condition:
        _passed += 1
    else:
        _failures.append(message)


def check_int(actual, expected, label):
    check(actual == expected, '%s: got %r, expected %r' % (label, actual, expected))


def check_close(actual, expected, label, tol=1e-9):
    check(abs(float(actual) - float(expected)) <= tol,
          '%s: got %r, expected %r' % (label, actual, expected))


def check_round(actual, expected, digits, label):
    got = round(float(actual), digits)
    check(got == expected, '%s: %r rounds to %r, expected %r'
          % (label, actual, got, expected))


def share(part, total):
    return 100.0 * float(part) / float(total)


def main():
    results = load('all_results.json')
    summary = load('dataset_summary.json')
    supplementary = load('supplementary_results.json')
    curves = load('threshold_sweep_curve.json')

    # ---------------------------------------------------------------
    # Dataset construction (Section 3.1 and 3.4)
    # ---------------------------------------------------------------
    section('Dataset counts and shares')

    raw = summary['raw_class_counts']
    proc = summary['processed_class_counts']

    check_int(summary['raw_rows'], 703245, 'raw rows')
    check_int(summary['rows_dropped'], 527, 'rows dropped')
    check_int(summary['rows_after_nan_drop'], 702718, 'rows after NaN drop')
    check_int(summary['n_features'], 78, 'numeric feature count')
    check_int(summary['n_features_with_nonzero_pooled_std'], 68,
              'features with non-zero pooled standard deviation')

    check_int(sum(raw.values()), 702718, 'raw class counts sum')
    check_int(raw['BENIGN'], 413933, 'raw benign count')
    check_int(raw['BENIGN'] + raw['Bot'] + raw['DDoS'] + raw['PortScan'],
              702718, 'raw class total')
    check_int(summary['processed_rows'], 318237, 'processed rows')
    check_int(sum(proc.values()), 318237, 'processed class counts sum')
    check_int(summary['processed_attack_rows'], 288785, 'processed attack rows')
    check_int(proc['BENIGN'] + proc['Bot'] + proc['DDoS'] + proc['PortScan'],
              318237, 'processed class total')
    check_int(proc['BENIGN'], 29452, 'processed benign count')
    check_int(proc['DDoS'], 128025, 'processed DDoS count')
    check_int(proc['PortScan'], 158804, 'processed PortScan count')
    check_int(proc['Bot'], 1956, 'processed Bot count')

    check_round(share(summary['rows_dropped'], summary['raw_rows']), 0.07, 2,
                'dropped-row percentage')
    check_round(share(raw['BENIGN'], 702718), 58.9, 1, 'raw benign share')
    check_round(share(proc['BENIGN'], 318237), 9.3, 1, 'processed benign share')
    check_round(share(proc['Bot'], 318237), 0.6, 1, 'processed Bot share')
    check_round(share(proc['DDoS'], 318237), 40.2, 1, 'processed DDoS share')
    check_round(share(proc['PortScan'], 318237), 49.9, 1,
                'processed PortScan share')
    check_round(share(proc['DDoS'] + proc['PortScan'], 318237), 90.1, 1,
                'DDoS plus PortScan share')

    # Experiment B construction.
    check_int(summary['experiment_b_train_rows'], 13736, 'B training rows')
    check_int(summary['experiment_b_test_rows'], 298609, 'B test rows')
    check_int(summary['experiment_b_train_benign'], 11780, 'B training benign')
    check_int(summary['experiment_b_test_benign'], 11780, 'B test benign')
    check_int(summary['experiment_b_train_benign'] + proc['Bot'], 13736,
              'B training rows from benign plus Bot')
    check_int(11780 + proc['DDoS'] + proc['PortScan'], 298609,
              'B test rows from benign plus attacks')
    check_round(share(11780, 13736), 85.8, 1, 'B training benign share')
    check_round(share(11780, 298609), 3.9, 1, 'B test benign share')
    check_round(share(298609 - 11780, 298609), 96.1, 1, 'B test attack share')

    # The two 40% benign halves and the unused 20%.
    check_int(2 * 11780 + (29452 - 2 * 11780), 29452,
              'two benign halves plus unused remainder')
    check_round(share(11780, 29452), 40.0, 1, 'benign share per B split')
    check_round(share(29452 - 2 * 11780, 29452), 20.0, 1,
                'unused benign share')

    # B2 training sample composition.
    check_int(summary['b2_train_attack'], 12471, 'B2 training attacks')
    check_int(summary['b2_train_benign'], 1265, 'B2 training benign')
    check_int(summary['b2_train_attack'] + summary['b2_train_benign'], 13736,
              'B2 training rows')
    check_round(share(summary['b2_train_attack'], 13736), 90.8, 1,
                'B2 training attack share')

    # B4 construction.
    check_int(summary['b4_pool'], 157477, 'B4 pool rows')
    check_int(summary['b4_train'], 125981, 'B4 training rows')
    check_int(summary['b4_test'], 31496, 'B4 test rows')
    check_int(summary['b4_train'] + summary['b4_test'], 157477,
              'B4 train plus test rows')
    check_int(summary['b4_test_benign'] + summary['b4_test_ddos'], 31496,
              'B4 test class total')
    check_int(summary['b4_test_ddos'], 25605, 'B4 test DDoS rows')
    check_int(summary['b4_test_benign'], 5891, 'B4 test benign rows')

    # ---------------------------------------------------------------
    # Table 2: Experiment A
    # ---------------------------------------------------------------
    section('Table 2: random split (Experiment A)')

    a = results['A_random_split']
    table_a = {
        'LR': (0.9911, 0.9951, 0.9938, 0.9965, 0.0611),
        'RF': (0.9995, 0.9997, 0.9997, 0.9998, 0.0029),
        'XGB': (0.9995, 0.9997, 0.9996, 0.9998, 0.0037),
        'MLP': (0.9981, 0.9989, 0.9985, 0.9994, 0.0148),
    }
    for model, expected in table_a.items():
        got = (round(a[model]['accuracy'], 4), round(a[model]['f1'], 4),
               round(a[model]['precision'], 4), round(a[model]['recall'], 4),
               round(a[model]['fpr'], 4))
        check(got == expected, 'Table 2 %s values: %r != %r'
              % (model, got, expected))

    check_round(a['RF']['inference_time_ms'], 0.0014, 4, 'A RF inference time')
    check_round(a['XGB']['inference_time_ms'], 0.0004, 4, 'A XGB inference time')
    check_round(a['MLP']['inference_time_ms'], 0.0009, 4, 'A MLP inference time')
    check(a['LR']['inference_time_ms'] < 0.0001, 'A LR inference time below 0.0001')

    check(min(a[m]['f1'] for m in MODELS) > 0.99,
          'all four random-split F1 scores exceed 0.99')
    check_round(min(a[m]['precision'] for m in MODELS), 0.9938, 4,
                'Experiment A minimum precision')
    check_round(max(a[m]['precision'] for m in MODELS), 0.9997, 4,
                'Experiment A maximum precision')
    check_round(min(a[m]['recall'] for m in MODELS), 0.9965, 4,
                'Experiment A minimum recall')
    check_round(max(a[m]['recall'] for m in MODELS), 0.9998, 4,
                'Experiment A maximum recall')
    check_round(min(a[m]['fpr'] for m in MODELS), 0.0029, 4,
                'Experiment A minimum FPR')
    check_round(max(a[m]['fpr'] for m in MODELS), 0.0611, 4,
                'Experiment A maximum FPR')

    # ---------------------------------------------------------------
    # Table 3: Experiment B
    # ---------------------------------------------------------------
    section('Table 3: session and attack-family shift (Experiment B)')

    b = results['B_session_shift']
    table_b = {
        'LR': (0.3021, 0.4307, 0.9949, 0.2748, 0.0343),
        'RF': (0.0393, 0.0000, 0.0000, 0.0000, 0.0040),
        'XGB': (0.0395, 0.0003, 0.6282, 0.0002, 0.0025),
        'MLP': (0.2833, 0.4059, 0.9960, 0.2549, 0.0249),
    }
    for model, expected in table_b.items():
        got = (round(b[model]['accuracy'], 4), round(b[model]['f1'], 4),
               round(b[model]['precision'], 4), round(b[model]['recall'], 4),
               round(b[model]['fpr'], 4))
        check(got == expected, 'Table 3 %s values: %r != %r'
              % (model, got, expected))

    check_round(b['LR']['inference_time_ms'], 0.0001, 4, 'B LR inference time')
    check_round(b['RF']['inference_time_ms'], 0.0014, 4, 'B RF inference time')
    check_round(b['XGB']['inference_time_ms'], 0.0003, 4, 'B XGB inference time')
    check_round(b['MLP']['inference_time_ms'], 0.0007, 4, 'B MLP inference time')

    check(b['RF']['f1'] == 0.0, 'RF F1 collapses to exactly 0.0 under B')
    check(b['RF']['recall'] == 0.0 and b['RF']['precision'] == 0.0,
          'RF predicts no positive flows under B')
    check(b['XGB']['recall'] < 0.001, 'XGB recall is negligible under B')
    check(0.70 < 1.0 - b['LR']['recall'] < 0.76,
          'LR misses roughly 73 percent of attacks')
    check(0.70 < 1.0 - b['MLP']['recall'] < 0.76,
          'MLP misses roughly 73 percent of attacks')

    # Losses relative to the random-split baseline.
    losses = {m: a[m]['f1'] - b[m]['f1'] for m in MODELS}
    check(0.56 <= min(losses.values()) < 0.57,
          'smallest B F1 loss is about 0.56')
    check(max(losses.values()) > 0.99, 'largest B F1 loss is about 0.99')
    check_round(a['RF']['f1'] - b['RF']['f1'], 0.9997, 4,
                'RF random-split to B F1 drop')

    # ---------------------------------------------------------------
    # Table 4: B versus B2
    # ---------------------------------------------------------------
    section('Table 4: matched-size control (B2)')

    b2 = results['B2_matched_size_control']
    b3 = results['B3_same_dist_same_size']
    table_b2 = {'LR': 0.9942, 'RF': 0.9992, 'XGB': 0.9991, 'MLP': 0.9979}
    for model, expected in table_b2.items():
        check_round(b2[model]['f1'], expected, 4, 'Table 4 %s F1' % model)
        check_round(b2[model]['f1'], expected, 4,
                    'B3 %s F1 matches B2 (same split and subsample)' % model)
        check_close(b3[model]['f1'], b2[model]['f1'],
                    'B3 %s F1 equals B2 exactly' % model)
        check(b2[model]['f1'] > b[model]['f1'],
              'B2 %s beats B under the same training size' % model)

    check_round(b2['RF']['f1'] - b['RF']['f1'], 0.9992, 4,
                'RF B2 minus B F1 gap')
    check(b2['RF']['f1'] > 0.999, 'B2 RF F1 exceeds 0.999')

    # ---------------------------------------------------------------
    # Table 5: B4 covariate-shift control
    # ---------------------------------------------------------------
    section('Table 5: seen-class covariate shift (B4)')

    control = supplementary['B4_covariate_shift_control']
    b4a = control['B4a_same_distribution']
    b4b = control['B4b_covariate_shift']

    check_int(control['pool']['n'], 157477, 'B4 pool size')
    check_int(control['pool']['n_train'], 125981, 'B4 training size')
    check_int(control['pool']['n_test'], 31496, 'B4 test size')

    table_b4 = {
        'LR': (0.9990, 0.9982, 0.0007),
        'RF': (0.9999, 0.9997, 0.0002),
        'XGB': (0.9999, 0.9997, 0.0003),
        'MLP': (0.9998, 0.9995, 0.0004),
    }
    for model, (f1_a, f1_b, delta) in table_b4.items():
        check_round(b4a[model]['f1'], f1_a, 4, 'Table 5 %s B4a F1' % model)
        check_round(b4b[model]['f1'], f1_b, 4, 'Table 5 %s B4b F1' % model)
        check_round(b4a[model]['f1'] - b4b[model]['f1'], delta, 4,
                    'Table 5 %s F1 change' % model)
        check(b4a[model]['f1'] - b4b[model]['f1'] < 0.001,
              'B4 %s change is negligible' % model)

    # ---------------------------------------------------------------
    # Table 6: attack-family leave-one-out
    # ---------------------------------------------------------------
    section('Table 6: attack-family leave-one-out (Experiment C)')

    c = results['C_attack_family_shift']
    table_c = {
        'Bot': (0.384, 0.000, 0.000, 0.000),
        'DDoS': (0.516, 0.624, 0.495, 0.639),
        'PortScan': (0.654, 0.510, 0.028, 0.682),
    }
    for held_out, expected in table_c.items():
        got = tuple(round(c[held_out][m]['f1'], 3) for m in MODELS)
        check(got == expected, 'Table 6 %s F1 values: %r != %r'
              % (held_out, got, expected))

    check(all(c['Bot'][m]['f1'] == 0.0 for m in ('RF', 'XGB', 'MLP')),
          'Bot leave-one-out collapses all non-linear models to zero')
    check(c['Bot']['LR']['f1'] > 0.30,
          'LR retains partial Bot detection')

    # ---------------------------------------------------------------
    # Table 7: synthetic feature perturbation
    # ---------------------------------------------------------------
    section('Table 7: synthetic feature perturbation (Experiment D)')

    d0 = results['D_no_shift_control']
    d1 = results['D_feature_shift']
    table_d = {
        'LR': (0.9951, 0.9084),
        'RF': (0.9997, 0.9997),
        'XGB': (0.9997, 0.9997),
        'MLP': (0.9989, 0.9983),
    }
    for model, (f1_before, f1_after) in table_d.items():
        check_round(d0[model]['f1'], f1_before, 4,
                    'Table 7 %s F1 without shift' % model)
        check_round(d1[model]['f1'], f1_after, 4,
                    'Table 7 %s F1 under feature shift' % model)
        check_close(d0[model]['f1'], a[model]['f1'],
                    'D no-shift %s F1 equals random split A' % model)

    d_delta = {m: d0[m]['f1'] - d1[m]['f1'] for m in MODELS}
    check_round(d_delta['LR'], 0.0868, 4, 'D LR F1 drop')
    check_round(d_delta['RF'], 0.0001, 4, 'D RF F1 drop')
    check_round(d_delta['XGB'], 0.0001, 4, 'D XGB F1 drop')
    check_round(d_delta['MLP'], 0.0006, 4, 'D MLP F1 drop')
    check(max(d_delta, key=d_delta.get) == 'LR', 'D largest drop is LR')

    # ---------------------------------------------------------------
    # Table 8: Wasserstein distances
    # ---------------------------------------------------------------
    section('Table 8: shift magnitude (Wasserstein distances)')

    w = supplementary['wasserstein']
    means = {
        'B_session_shift': 0.293,
        'B4_covariate_shift': 0.152,
        'D_feature_shift': 0.106,
    }
    for key, expected in means.items():
        check_round(w[key]['mean_w1_normalized'], expected, 3,
                    'Table 8 %s mean normalised W1' % key)
        check_int(w[key]['n_features'], 68, 'Table 8 %s feature count' % key)

    top_expected = {
        'B_session_shift': (
            ('PSH Flag Count', 1.01),
            ('Min Packet Length', 0.81),
            ('Bwd Packet Length Min', 0.77),
        ),
        'B4_covariate_shift': (
            ('Packet Length Variance', 0.94),
            ('Flow IAT Std', 0.71),
            ('Flow Duration', 0.71),
        ),
        'D_feature_shift': (
            ('Packet Length Variance', 0.59),
            ('Flow Duration', 0.47),
            ('Flow IAT Std', 0.47),
        ),
    }
    for key, expected in top_expected.items():
        top = {name: value for name, value in w[key]['top_features']}
        for name, value in expected:
            check(name in top, 'Table 8 %s contains %s' % (key, name))
            if name in top:
                check_round(top[name], value, 2,
                            'Table 8 %s %s distance' % (key, name))

    check_int(summary['n_features'] - w['B_session_shift']['n_features'], 10,
              'features skipped because pooled standard deviation is zero')

    b_drop = max(a[m]['f1'] - b[m]['f1'] for m in MODELS)
    b4_drop = max(b4a[m]['f1'] - b4b[m]['f1'] for m in MODELS)
    d_max_drop = max(d_delta.values())
    check_round(b_drop, 0.9997, 4, 'Table 8 maximum B F1 drop')
    check_round(b4_drop, 0.0007, 4, 'Table 8 maximum B4 F1 drop')
    check_round(d_max_drop, 0.0868, 4, 'Table 8 maximum D F1 drop')
    check(max(((a[m]['f1'] - b[m]['f1']), m) for m in MODELS)[1] == 'RF',
          'Table 8 B largest drop is RF')
    check(max(((b4a[m]['f1'] - b4b[m]['f1']), m) for m in MODELS)[1] == 'LR',
          'Table 8 B4 largest drop is LR')

    check(0.45 < w['B4_covariate_shift']['mean_w1_normalized']
          / w['B_session_shift']['mean_w1_normalized'] < 0.60,
          'B4 displacement is roughly half of B displacement')
    check(500 < b_drop / b4_drop < 3000,
          'B F1 drop is about a thousand times the B4 drop')

    # ---------------------------------------------------------------
    # Table 9: threshold sweep and Section 4.8 prose
    # ---------------------------------------------------------------
    section('Table 9: threshold sweep on the B test set')

    table_9 = {
        'LR': (0.4307, 0.0343, 0.7104, 0.0581, 0.7115, 0.1602),
        'RF': (0.0000, 0.0040, 0.7114, 0.0399, 0.8262, 0.0654),
        'XGB': (0.0003, 0.0025, 0.4635, 0.0081, 0.6037, 0.0219),
        'MLP': (0.4059, 0.0249, 0.4923, 0.0368, 0.6614, 0.0492),
    }
    best_expected = {
        'LR': (0.001, 0.7240),
        'RF': (0.009, 0.8263),
        'XGB': (0.002, 0.6731),
        'MLP': (0.001, 0.7067),
    }

    for model, expected in table_9.items():
        curve = curves[model]
        check_round(curve['default_f1'], expected[0], 4,
                    'Table 9 %s F1 at 0.50' % model)
        check_round(curve['default_fpr'], expected[1], 4,
                    'Table 9 %s FPR at 0.50' % model)

        thresholds = curve['thresholds_plot']
        f1_curve = curve['f1_curve']
        fpr_curve = curve['fpr_curve']
        for anchor, f1_expected, fpr_expected in (
                (0.05, expected[2], expected[3]),
                (0.01, expected[4], expected[5])):
            index = min(range(len(thresholds)),
                        key=lambda j: abs(thresholds[j] - anchor))
            check(abs(thresholds[index] - anchor) < 1e-9,
                  'anchor threshold %s present for %s' % (anchor, model))
            check_round(f1_curve[index], f1_expected, 4,
                        'Table 9 %s F1 at %s' % (model, anchor))
            check_round(fpr_curve[index], fpr_expected, 4,
                        'Table 9 %s FPR at %s' % (model, anchor))

        best_threshold, best_f1 = best_expected[model]
        check_close(curve['best_threshold'], best_threshold,
                    'Section 4.8 %s best threshold' % model, tol=1e-6)
        check_round(curve['best_f1'], best_f1, 4,
                    'Section 4.8 %s best F1' % model)
        check_close(curve['best_f1'],
                    supplementary['threshold_sweep_B'][model]['max_f1'],
                    'Section 4.8 %s best F1 matches supplementary sweep'
                    % model)
        check(curve['best_threshold'] >= 0.001,
              '%s best threshold excludes the degenerate zero' % model)
        check(curve['best_threshold'] < 0.5,
              '%s best threshold lies below the default' % model)

        # Threshold zero predicts every flow as attack (degenerate).
        check_round(f1_curve[0], 0.98, 2,
                    'Section 4.8 %s degenerate F1 at threshold zero' % model)
        check_close(fpr_curve[0], 1.0,
                    'Section 4.8 %s degenerate FPR at threshold zero' % model)

        check(curve['best_f1'] < b2[model]['f1'],
              '%s threshold recovery remains below the B2 baseline' % model)
        check(abs(curve['default_f1'] - b[model]['f1']) < 1e-8,
              '%s sweep default F1 matches the B result' % model)
        check(abs(curve['default_fpr'] - b[model]['fpr']) < 1e-8,
              '%s sweep default FPR matches the B result' % model)

    check_round(curves['RF']['best_f1'], 0.826, 3,
                'RF recovery rounds to 0.826')
    check_round(curves['RF']['default_fpr'] * 100, 0.4, 1,
                'RF default FPR is 0.4 percent')
    check_round(0.0654 * 100, 6.5, 1, 'RF FPR at 0.01 is 6.5 percent')
    check_round(curves['LR']['best_fpr'] * 100, 30.0, 1,
                'LR FPR at its best threshold is 30.0 percent')

    # Prose in Section 4.8 reports the false-positive rate and the fraction of
    # all test flows flagged at each model's best operating point.
    best_fpr_expected = {'LR': 0.300, 'RF': 0.065, 'XGB': 0.054, 'MLP': 0.076}
    best_frac_expected = {'LR': 0.564, 'RF': 0.681, 'XGB': 0.491, 'MLP': 0.530}
    for model, expected in best_fpr_expected.items():
        check_round(curves[model]['best_fpr'], expected, 3,
                    'Section 4.8 %s best-point FPR' % model)
    for model, expected in best_frac_expected.items():
        check_round(curves[model]['best_frac_pred'], expected, 3,
                    'Section 4.8 %s flagged share' % model)
    flagged = {m: curves[m]['best_frac_pred'] for m in MODELS}
    check(min(flagged, key=flagged.get) == 'XGB',
          'XGB flags the smallest share of flows at its best point')
    check(max(flagged, key=flagged.get) == 'RF',
          'RF flags the largest share of flows at its best point')
    check_round(100 * min(flagged.values()), 49.1, 1,
                'lowest flagged share is 49.1 percent')
    check_round(100 * max(flagged.values()), 68.1, 1,
                'highest flagged share is 68.1 percent')

    # ---------------------------------------------------------------
    # Table 10: ROC-AUC under shift
    # ---------------------------------------------------------------
    section('Table 10: ROC-AUC under shift')

    table_10 = {'LR': 0.6081, 'RF': 0.8185, 'XGB': 0.8452, 'MLP': 0.6485}
    for model, expected in table_10.items():
        check_round(b[model]['roc_auc'], expected, 4,
                    'Table 10 %s ROC-AUC' % model)
        check(b[model]['roc_auc'] > 0.5,
              'Table 10 %s ROC-AUC exceeds chance' % model)

    check_round(min(b[m]['roc_auc'] for m in MODELS), 0.61, 2,
                'ROC-AUC range lower bound 0.61')
    check_round(max(b[m]['roc_auc'] for m in MODELS), 0.85, 2,
                'ROC-AUC range upper bound 0.85')
    check_round(b['RF']['roc_auc'], 0.819, 3,
                'abstract RF ROC-AUC rounds to 0.819')

    # ---------------------------------------------------------------
    # Section 4.10: multiclass weighted F1
    # ---------------------------------------------------------------
    section('Section 4.10: multiclass weighted F1')

    mc = results['multiclass']
    table_mc = {'LR': 0.9956, 'RF': 0.9995, 'XGB': 0.9996, 'MLP': 0.9986}
    for model, expected in table_mc.items():
        check_round(mc[model]['f1'], expected, 4,
                    'Section 4.10 %s weighted F1' % model)
        check(mc[model]['f1'] > 0.99,
              'Section 4.10 %s remains above 0.99' % model)

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------
    print('\n' + '=' * 60)
    if _failures:
        print('%d checks passed, %d FAILED' % (_passed, len(_failures)))
        for failure in _failures:
            print('  FAIL: ' + failure)
        return 1
    print('OK: %d assertions passed.' % _passed)
    return 0


if __name__ == '__main__':
    sys.exit(main())
