"""
Tests whether OUR OWN readout has its own hidden memory cliff, masked by the iterative
Sudoku-legality cleanup loop -- the direct, symmetric counterpart to
modern_hopfield_memory_cliff_test.py, which found a genuine fixed-beta cliff for Modern
Hopfield. Our own attention readout, p = softmax(sim/tau), is mathematically the identical
form to Modern Hopfield's softmax(beta*sim) with beta=1/tau; every comparison against Modern
Hopfield in this project used our own default tau=2.0 (beta=0.5) UNCHANGED and UNSCALED across
the entire M range -- exactly the kind of naive, un-validated carryover that failed almost
completely when first tried on Modern Hopfield itself.

This script isolates the SINGLE-SHOT attention readout (image -> W_SI -> k-WTA -> ONE
readout_attention call -> decode_digits), with NO iterative Sudoku-legality cleanup pass at
all, across the identical extended M-grid (100-4000) and sigma=0.3 already used for the full
pipeline (full_memory_vs_accuracy_consolidated.py) and for Modern Hopfield
(modern_hopfield_memory_cliff_test.py). If single-shot accuracy collapses sharply while the
full pipeline (already measured) declines gracefully, that tells us the cleanup loop -- not an
inherently robust readout -- is doing the real work of avoiding a cliff. If single-shot
accuracy ALSO declines gracefully, that says our readout's fixed beta=0.5 genuinely doesn't
need Modern-Hopfield-style re-tuning at this task's similarity scale (engram overlap, bounded
0-K), unlike raw image dot products.

Also runs a beta-rescue sweep at high M (mirroring modern_hopfield_memory_cliff_test.py's own
beta-rescue check) to see whether the single-shot readout's degradation, if any, responds to
increasing beta the way Modern Hopfield's did.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, TAU, relation, generate_unique, drive_vectors, conflict_matrix,
    readout_attention, decode_digits,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk
from hopfield_modern_sdm_dam_comparison import wilson_ci

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
M_SWEEP = [100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000]
SIGMA_LEVELS = [0.0, 0.3]
N_TRIALS = 200
SEED_TRIALS = SEED + 999

RESCUE_M_VALUES = [2000, 4000]
RESCUE_TAUS = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]  # smaller tau = larger effective beta = sharper


def build_pipeline(M_local):
    rng0 = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M_local, N)
    images, img_dim = load_dataset('mnist', M_local)
    rng = np.random.default_rng(SEED)
    W_SH = make_projection_hk(rng, H_val)
    H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)
    return images, H_keys, B_drive, W_SI, digits_true_all


def single_shot_recall(H_keys, B_drive, W_SI, x_query, tau=TAU):
    h_initial = k_wta_hk(W_SI @ x_query, H_val, K_val)
    b_hat = readout_attention(H_keys, B_drive, h_initial, tau=tau)
    return decode_digits(b_hat)


def evaluate(images, H_keys, B_drive, W_SI, digits_true_all, M_local, sigma, tau, n_trials, seed):
    rng = np.random.default_rng(seed)
    correct = []
    for _ in range(n_trials):
        m = rng.integers(M_local)
        x_noisy = add_noise(images[m:m + 1], sigma, rng)[0]
        digits_out = single_shot_recall(H_keys, B_drive, W_SI, x_noisy, tau=tau)
        correct.append(int(np.all(digits_out == digits_true_all[m])))
    lo, hi = wilson_ci(sum(correct), n_trials)
    return np.mean(correct), lo, hi


def main():
    print(f'{"=" * 90}\nTEST 1: single-shot readout (NO cleanup), fixed tau={TAU} (beta={1/TAU:g}), '
          f'extended M-sweep, N={N_TRIALS}/point\n{"=" * 90}')
    rows = []
    for M_local in M_SWEEP:
        t0 = time.time()
        images, H_keys, B_drive, W_SI, digits_true_all = build_pipeline(M_local)
        for sigma in SIGMA_LEVELS:
            acc, lo, hi = evaluate(images, H_keys, B_drive, W_SI, digits_true_all,
                                    M_local, sigma, TAU, N_TRIALS, SEED_TRIALS)
            rows.append({'M': M_local, 'sigma': sigma, 'tau': TAU, 'exact': acc,
                         'ci_lo': lo, 'ci_hi': hi})
            print(f'  M={M_local:5d} sigma={sigma:.1f}  exact={acc:.3f} [{lo:.3f},{hi:.3f}]  '
                  f'[{time.time()-t0:.1f}s]')
    df_sweep = pd.DataFrame(rows)
    df_sweep.to_csv('ours_single_shot_readout_M_sweep.csv', index=False)

    print(f'\n{"=" * 90}\nTEST 2: beta-rescue (via tau) at M={RESCUE_M_VALUES}, sigma=0.3, '
          f'N={N_TRIALS}/point\n{"=" * 90}')
    rescue_rows = []
    for M_local in RESCUE_M_VALUES:
        t0 = time.time()
        images, H_keys, B_drive, W_SI, digits_true_all = build_pipeline(M_local)
        for tau in RESCUE_TAUS:
            acc, lo, hi = evaluate(images, H_keys, B_drive, W_SI, digits_true_all,
                                    M_local, 0.3, tau, N_TRIALS, SEED_TRIALS)
            rescue_rows.append({'M': M_local, 'tau': tau, 'beta': 1 / tau, 'exact': acc,
                                'ci_lo': lo, 'ci_hi': hi})
            print(f'  M={M_local:5d} tau={tau:5.2f} (beta={1/tau:6.2f})  exact={acc:.3f} '
                  f'[{lo:.3f},{hi:.3f}]  [{time.time()-t0:.1f}s]')
    df_rescue = pd.DataFrame(rescue_rows)
    df_rescue.to_csv('ours_single_shot_readout_beta_rescue.csv', index=False)

    # overlay against the already-established FULL PIPELINE numbers (with cleanup)
    try:
        df_full = pd.read_csv('full_memory_vs_accuracy_consolidated.csv')
        full_03 = df_full[['M', 'ours_exact', 'ours_ci_lo', 'ours_ci_hi']]
    except FileNotFoundError:
        full_03 = None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for sigma, color, label in [(0.0, 'tab:blue', 'single-shot, sigma=0.0'),
                                  (0.3, 'tab:orange', 'single-shot, sigma=0.3')]:
        sub = df_sweep[df_sweep.sigma == sigma]
        axes[0].errorbar(sub.M, sub.exact, yerr=[sub.exact - sub.ci_lo, sub.ci_hi - sub.exact],
                          fmt='o-', color=color, label=label, capsize=3)
    if full_03 is not None:
        axes[0].errorbar(full_03.M, full_03.ours_exact,
                          yerr=[full_03.ours_exact - full_03.ours_ci_lo,
                                full_03.ours_ci_hi - full_03.ours_exact],
                          fmt='s--', color='tab:green', label='FULL pipeline (with cleanup), sigma=0.3',
                          capsize=3)
    axes[0].set_xlabel('M (number of stored images)'); axes[0].set_ylabel('Exact recovery rate')
    axes[0].set_title(f'Single-shot readout (beta=1/tau={1/TAU:g}) vs. full pipeline')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    colors_m = {2000: 'tab:purple', 4000: 'tab:red'}
    for M_local in RESCUE_M_VALUES:
        sub = df_rescue[df_rescue.M == M_local]
        axes[1].errorbar(sub.beta, sub.exact, yerr=[sub.exact - sub.ci_lo, sub.ci_hi - sub.exact],
                          fmt='o-', color=colors_m[M_local], label=f'M={M_local}', capsize=3)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('beta = 1/tau (log scale)'); axes[1].set_ylabel('Exact recovery (sigma=0.3)')
    axes[1].set_title('Does more beta (sharper tau) rescue single-shot recovery?')
    axes[1].legend(); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('ours_single_shot_readout_cliff_test.png', dpi=150)
    print('\nSaved ours_single_shot_readout_M_sweep.csv, ours_single_shot_readout_beta_rescue.csv, '
          'ours_single_shot_readout_cliff_test.png')


if __name__ == '__main__':
    main()
