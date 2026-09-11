"""
Two follow-up tests on the Modern-Hopfield/SDM/DAM comparison
(hopfield_modern_sdm_dam_comparison.py), which found a large gap at M=800, sigma=0.3
(Modern Hopfield 99.0%, DAM 96.0%, ours 57.5%):

  1. Does a bigger engram (H) close that gap for OUR network specifically? The existing
     heteroassoc_H_extends_capacity.py already showed doubling H (1000->2000, K scaled to
     keep density=0.15) leaves the zero-noise-overlap / sigma=0.1-exact numbers essentially
     unchanged at every M tested (e.g. M=800: 0.8532 vs 0.8530 zero-noise overlap) --
     consistent with the image-side effective-rank ceiling being architecture-independent.
     But that test used sigma=0.1, where recovery was already near-ceiling (1.0) at both H
     values -- it couldn't reveal a difference even if one existed. This script re-tests
     specifically at M=800, sigma=0.3 (the exact condition with the big gap), across
     H in {1000, 1500, 2000, 3000} (K=0.15H throughout, matching the established
     density convention), N=200 trials/point.

  2. A noise-sweep at M=800 (high-memory regime) for all 7 methods, alongside the
     existing M=50 (low-memory) sweep -- since the M-sweep already showed pinv/modern/dam/
     ours are all close together up to M~200 and diverge sharply after, a noise-sweep at
     M=50 alone can't show whether that divergence is noise-level-dependent or a pure
     capacity effect. Same sigma grid and N=200/point as the M=50 sweep, for direct
     comparison.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import N, SEED, relation, generate_unique, drive_vectors, conflict_matrix
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk
from hopfield_vs_our_network_comparison import (
    bipolarize, train_hopfield_outer, train_hopfield_covariance, train_hopfield_pinv, hopfield_recall,
)
from hopfield_modern_sdm_dam_comparison import (
    train_modern_hopfield, modern_hopfield_recall, train_sdm, sdm_recall, train_dam, dam_recall,
    METHODS, LABELS, wilson_ci,
)

RIDGE_LAMBDA = 1.0
N_TRIALS = 200
DENSITY_RATIO = 0.15  # K/H, matching heteroassoc_H_extends_capacity.py's convention

M_FOR_H_TEST = 800
SIGMA_FOR_H_TEST = 0.3
H_SWEEP = [1000, 1500, 2000, 3000]

SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
M_HIGH = 800


# ---------- Experiment 1: does bigger H help OUR network at M=800, sigma=0.3? ----------

def run_ours_at_H(M_local, H_val, sigma, n_trials):
    K_val = int(round(DENSITY_RATIO * H_val))
    rng0 = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M_local, N)
    conf = conflict_matrix()
    images, img_dim = load_dataset('mnist', M_local)
    rng = np.random.default_rng(SEED)
    W_SH = make_projection_hk(rng, H_val)
    H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)

    trng = np.random.default_rng(SEED + 999)
    correct = []
    for _ in range(n_trials):
        m = trng.integers(M_local)
        x_noisy = add_noise(images[m:m + 1], sigma, trng)[0]
        h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
        h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
            H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
        correct.append(int(np.all(digits_final == digits_true_all[m])))
    lo, hi = wilson_ci(sum(correct), len(correct))
    return K_val, np.mean(correct), lo, hi


def experiment_1_H_scaling():
    print(f'{"=" * 90}\nEXPERIMENT 1: does bigger H help "ours" at M={M_FOR_H_TEST}, '
          f'sigma={SIGMA_FOR_H_TEST}? (N={N_TRIALS}/point)\n{"=" * 90}')
    rows = []
    for H_val in H_SWEEP:
        t0 = time.time()
        K_val, acc, lo, hi = run_ours_at_H(M_FOR_H_TEST, H_val, SIGMA_FOR_H_TEST, N_TRIALS)
        rows.append({'H': H_val, 'K': K_val, 'M': M_FOR_H_TEST, 'sigma': SIGMA_FOR_H_TEST,
                     'exact': acc, 'ci_lo': lo, 'ci_hi': hi})
        print(f'  H={H_val:5d} K={K_val:4d}  exact_recovery={acc:.3f} [{lo:.3f},{hi:.3f}]  '
              f'[{time.time()-t0:.1f}s]')
    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_H_scaling_at_M800_sigma03.csv', index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(df.H, df.exact, yerr=[df.exact - df.ci_lo, df.ci_hi - df.exact],
                fmt='o-', color='tab:green', capsize=4)
    ax.axhline(0.990, color='tab:blue', linestyle='--', label='Modern Hopfield (H=1000 comparison)')
    ax.axhline(0.960, color='violet', linestyle='--', label='DAM (H=1000 comparison)')
    ax.set_xlabel('H (engram dimension, K=0.15H throughout)')
    ax.set_ylabel('Exact recovery rate')
    ax.set_title(f'Does bigger H close the gap? M={M_FOR_H_TEST}, sigma={SIGMA_FOR_H_TEST}')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig('heteroassoc_H_scaling_at_M800_sigma03.png', dpi=150)
    print('Saved heteroassoc_H_scaling_at_M800_sigma03.csv/.png')
    return df


# ---------- Experiment 2: noise-sweep at HIGH M (M=800), all 7 methods ----------

def build_all_stores(M_local):
    rng0 = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M_local, N)
    conf = conflict_matrix()
    images, img_dim = load_dataset('mnist', M_local)
    rng = np.random.default_rng(SEED)
    H_val, K_val = 1000, 150
    W_SH = make_projection_hk(rng, H_val)
    H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)
    return dict(images=images, H_keys=H_keys, B_drive=B_drive, W_SH=W_SH, conf=conf,
                digits_true_all=digits_true_all, W_SI=W_SI, H_val=H_val, K_val=K_val)


def noise_sweep_at_M(M_local, sigma_levels, n_trials):
    t0 = time.time()
    store = build_all_stores(M_local)
    images = store['images']
    images_bp_clean = bipolarize(images)
    W_outer = train_hopfield_outer(images_bp_clean)
    W_cov = train_hopfield_covariance(images_bp_clean)
    W_pinv = train_hopfield_pinv(images_bp_clean, ridge=1.0)
    X_modern = train_modern_hopfield(images)
    rng_sdm = np.random.default_rng(SEED + 12345)
    sdm_store = train_sdm(images_bp_clean, rng_sdm)
    X_dam = train_dam(images_bp_clean)
    H_keys, B_drive, W_SH, conf = store['H_keys'], store['B_drive'], store['W_SH'], store['conf']
    W_SI, digits_true_all = store['W_SI'], store['digits_true_all']
    H_val, K_val = store['H_val'], store['K_val']

    rows = []
    for sigma in sigma_levels:
        rng = np.random.default_rng(SEED + 999)
        results = {k: [] for k in METHODS}
        for _ in range(n_trials):
            m = rng.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], sigma, rng)[0]
            cue_bp = bipolarize(x_noisy[None, :])[0]
            results['outer'].append(int(np.array_equal(hopfield_recall(W_outer, cue_bp), images_bp_clean[m])))
            results['cov'].append(int(np.array_equal(hopfield_recall(W_cov, cue_bp), images_bp_clean[m])))
            results['pinv'].append(int(np.array_equal(hopfield_recall(W_pinv, cue_bp), images_bp_clean[m])))
            out_modern = modern_hopfield_recall(X_modern, x_noisy)
            nn_idx = int(np.argmin(np.sum((X_modern - out_modern[None, :]) ** 2, axis=1)))
            results['modern'].append(int(nn_idx == m))
            out_sdm, activated = sdm_recall(sdm_store, cue_bp)
            results['sdm'].append(int(activated and np.array_equal(out_sdm, images_bp_clean[m])))
            results['dam'].append(int(np.array_equal(dam_recall(X_dam, cue_bp), images_bp_clean[m])))
            h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
                H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
            results['ours'].append(int(np.all(digits_final == digits_true_all[m])))
        row = {'M': M_local, 'sigma': sigma}
        for key in METHODS:
            arr = results[key]
            lo, hi = wilson_ci(sum(arr), len(arr))
            row[f'{key}_exact'] = np.mean(arr)
            row[f'{key}_ci_lo'] = lo
            row[f'{key}_ci_hi'] = hi
        rows.append(row)
        print(f"  M={M_local} sigma={sigma:.2f}  " +
              "  ".join(f"{k}={row[f'{k}_exact']:.3f}" for k in METHODS) +
              f"  [{time.time()-t0:.1f}s]")
    return rows


def experiment_2_highM_noise_sweep():
    print(f'\n{"=" * 90}\nEXPERIMENT 2: noise-sweep at M={M_HIGH} (high-memory regime), '
          f'N={N_TRIALS}/point\n{"=" * 90}')
    rows = noise_sweep_at_M(M_HIGH, SIGMA_LEVELS, N_TRIALS)
    df = pd.DataFrame(rows)
    df.to_csv(f'hopfield_modern_sdm_dam_noise_sweep_M{M_HIGH}.csv', index=False)
    print(f'Saved hopfield_modern_sdm_dam_noise_sweep_M{M_HIGH}.csv')
    return df


def main():
    df_h = experiment_1_H_scaling()
    df_noise_high = experiment_2_highM_noise_sweep()

    # combined low-M vs. high-M noise-sweep figure, reusing the M=50 CSV already on disk
    df_noise_low = pd.read_csv('hopfield_modern_sdm_dam_noise_sweep.csv')
    colors = {'outer': 'tab:red', 'cov': 'tab:purple', 'pinv': 'tab:orange',
              'modern': 'tab:blue', 'sdm': 'tab:brown', 'dam': 'tab:pink', 'ours': 'tab:green'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for key in METHODS:
        axes[0].errorbar(df_noise_low.sigma, df_noise_low[f'{key}_exact'],
                          yerr=[df_noise_low[f'{key}_exact'] - df_noise_low[f'{key}_ci_lo'],
                                df_noise_low[f'{key}_ci_hi'] - df_noise_low[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=LABELS[key], capsize=3)
        axes[1].errorbar(df_noise_high.sigma, df_noise_high[f'{key}_exact'],
                          yerr=[df_noise_high[f'{key}_exact'] - df_noise_high[f'{key}_ci_lo'],
                                df_noise_high[f'{key}_ci_hi'] - df_noise_high[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=LABELS[key], capsize=3)
    axes[0].set_title('Low-memory regime: M=50'); axes[0].set_xlabel('Gaussian pixel noise sigma')
    axes[0].set_ylabel('Exact recovery rate'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_title(f'High-memory regime: M={M_HIGH}'); axes[1].set_xlabel('Gaussian pixel noise sigma')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('hopfield_modern_sdm_dam_noise_sweep_low_vs_high_M.png', dpi=150)
    print('\nSaved hopfield_modern_sdm_dam_noise_sweep_low_vs_high_M.png')


if __name__ == '__main__':
    main()
