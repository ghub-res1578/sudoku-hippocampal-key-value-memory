"""
Redoes the Hopfield comparison (Table 8) fixing two issues at once:

  (1) STATISTICAL RIGOR: N=200 trials/point with Wilson 95% CIs, matching Table 7's bar,
      not the N=20-60 used previously.

  (2) MATCHED NOISE MODEL: previously the three auto-associative baselines were corrupted
      via random BIT-FLIPS on the bipolarized image, while our network was corrupted via
      additive GAUSSIAN noise on the continuous [0,1] image -- two different corruption
      models with no shared severity axis. Here every method sees the SAME corrupted
      continuous image per trial (paired): our network consumes it directly; the three
      Hopfield variants receive the SAME noisy image bipolarized (thresholded at 0.5) --
      i.e. bit-flips now arise naturally from Gaussian pixel noise crossing the threshold,
      rather than being imposed as a separate, disconnected corruption process. This gives
      one shared x-axis (Gaussian sigma) for all four methods.

Reuses the training routines from hopfield_vs_our_network_comparison.py unchanged.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, TAU,
    relation, generate_unique, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk
from hopfield_vs_our_network_comparison import (
    bipolarize, train_hopfield_outer, train_hopfield_covariance, train_hopfield_pinv, hopfield_recall,
)

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0

M_SWEEP = [100, 200, 400, 800]
SIGMA_FOR_M_SWEEP = 0.3
N_TRIALS = 200

NOISE_SWEEP_M = 50
SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - halfwidth, center + halfwidth)


def build_our_pipeline(M_local):
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
    return dict(images=images, H_keys=H_keys, B_drive=B_drive, W_SH=W_SH, conf=conf,
                digits_true_all=digits_true_all, W_SI=W_SI)


def run_one_M(M_local, sigma_levels, n_trials, label_prefix):
    t0 = time.time()
    store = build_our_pipeline(M_local)
    images = store['images']
    images_bp_clean = bipolarize(images)
    W_outer = train_hopfield_outer(images_bp_clean)
    W_cov = train_hopfield_covariance(images_bp_clean)
    W_pinv = train_hopfield_pinv(images_bp_clean, ridge=1.0)

    H_keys, B_drive, W_SH, conf = store['H_keys'], store['B_drive'], store['W_SH'], store['conf']
    W_SI, digits_true_all = store['W_SI'], store['digits_true_all']

    rows = []
    for sigma in sigma_levels:
        rng = np.random.default_rng(SEED + 999)
        results = {'outer': [], 'cov': [], 'pinv': [], 'ours': []}
        flip_fracs = []
        for _ in range(n_trials):
            m = rng.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], sigma, rng)[0]
            cue_bp = bipolarize(x_noisy[None, :])[0]
            flip_fracs.append(np.mean(cue_bp != images_bp_clean[m]))

            results['outer'].append(int(np.array_equal(hopfield_recall(W_outer, cue_bp), images_bp_clean[m])))
            results['cov'].append(int(np.array_equal(hopfield_recall(W_cov, cue_bp), images_bp_clean[m])))
            results['pinv'].append(int(np.array_equal(hopfield_recall(W_pinv, cue_bp), images_bp_clean[m])))

            h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
                H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
            results['ours'].append(int(np.all(digits_final == digits_true_all[m])))

        row = {'M': M_local, 'sigma': sigma, 'mean_effective_flip_frac': np.mean(flip_fracs)}
        for key in ['outer', 'cov', 'pinv', 'ours']:
            arr = results[key]
            lo, hi = wilson_ci(sum(arr), len(arr))
            row[f'{key}_exact'] = np.mean(arr)
            row[f'{key}_ci_lo'] = lo
            row[f'{key}_ci_hi'] = hi
        rows.append(row)
        print(f'  {label_prefix} sigma={sigma:.2f} (eff.flip={np.mean(flip_fracs):.1%})  '
              f'outer={row["outer_exact"]:.3f}  cov={row["cov_exact"]:.3f}  '
              f'pinv={row["pinv_exact"]:.3f}  ours={row["ours_exact"]:.3f}  [{time.time()-t0:.1f}s]')
    return rows


def main():
    print(f'{"=" * 78}\nM-SWEEP at fixed sigma={SIGMA_FOR_M_SWEEP}, N={N_TRIALS}/point\n{"=" * 78}')
    m_rows = []
    for M_local in M_SWEEP:
        m_rows.extend(run_one_M(M_local, [SIGMA_FOR_M_SWEEP], N_TRIALS, f'M={M_local}'))
    df_m = pd.DataFrame(m_rows)
    df_m.to_csv('hopfield_matched_noise_M_sweep.csv', index=False)

    print(f'\n{"=" * 78}\nNOISE-SWEEP at fixed M={NOISE_SWEEP_M}, N={N_TRIALS}/point\n{"=" * 78}')
    noise_rows = run_one_M(NOISE_SWEEP_M, SIGMA_LEVELS, N_TRIALS, f'M={NOISE_SWEEP_M}')
    df_n = pd.DataFrame(noise_rows)
    df_n.to_csv('hopfield_matched_noise_noise_sweep.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = {'outer': 'tab:red', 'cov': 'tab:purple', 'pinv': 'tab:orange', 'ours': 'tab:green'}
    labels = {'outer': 'Classical Hopfield', 'cov': 'Covariance-rule Hopfield',
              'pinv': 'Pseudo-inverse Hopfield', 'ours': 'Our network'}
    for key in ['outer', 'cov', 'pinv', 'ours']:
        axes[0].errorbar(df_m.M, df_m[f'{key}_exact'],
                          yerr=[df_m[f'{key}_exact'] - df_m[f'{key}_ci_lo'], df_m[f'{key}_ci_hi'] - df_m[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=labels[key], capsize=3)
        axes[1].errorbar(df_n.sigma, df_n[f'{key}_exact'],
                          yerr=[df_n[f'{key}_exact'] - df_n[f'{key}_ci_lo'], df_n[f'{key}_ci_hi'] - df_n[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=labels[key], capsize=3)
    axes[0].set_xlabel('M (number of stored images)'); axes[0].set_ylabel('Exact recovery rate')
    axes[0].set_title(f'Exact recovery vs. M (shared Gaussian noise, sigma={SIGMA_FOR_M_SWEEP})')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Gaussian pixel noise sigma (shared across all 4 methods)')
    axes[1].set_ylabel('Exact recovery rate')
    axes[1].set_title(f'Exact recovery vs. noise, fixed M={NOISE_SWEEP_M} (shared noise model)')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('hopfield_matched_noise_rigorous.png', dpi=150)
    print('\nSaved hopfield_matched_noise_rigorous.png')


if __name__ == '__main__':
    main()
