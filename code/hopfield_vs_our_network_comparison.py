"""
The definitive comparison: classical Hopfield network vs. our heteroassociation +
Sudoku-attractor pipeline, on the SAME MNIST images (not random patterns), across the
SAME range of stored-item counts.

Three methods, all tested on identical images:
  1. Classic Hopfield (outer-product/Hebbian rule) -- auto-associative: image is both
     the cue and the target. The textbook memory-cliff architecture.
  2. Pseudo-inverse Hopfield (projection rule) -- SAME learning-rule family as our own
     network (dual-form ridge pseudo-inverse), but still auto-associative with no
     separate error-correcting scaffold. Isolates "is it the rule or the architecture
     that avoids the cliff?"
  3. Our network -- bidirectional pseudo-inverse heteroassociation (image <-> engram)
     PLUS the separate Sudoku-legality attractor for cleanup (the actual architecture
     difference from #2).

Two comparisons:
  A. Exact-recovery vs. M (number of stored images), at a fixed, moderate noise level
     for each method's own natural corruption type.
  B. Exact-recovery vs. noise level, at a fixed, SMALL M=50 (chosen to sit below the
     classic Hopfield's cliff at zero noise, so all three methods have "room to fail"
     as noise increases, rather than one already being collapsed at the starting point).

Hopfield noise = fraction of bipolar bits flipped (the standard corruption model for
{-1,+1} networks). Our network's noise = additive Gaussian noise on [0,1] pixels (the
model used throughout this project). These are NOT the same units -- plotted on
separate x-axes for the noise-sweep panel, with the same corruption-severity ordering,
not a claim of numerical equivalence.
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
    N, SEED, TAU,
    relation, generate_unique, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
MAX_HOPFIELD_ITERS = 15

M_SWEEP = [100, 200, 300, 400, 500, 600, 700, 800]
OUR_SIGMA_FOR_M_SWEEP = 0.5
HOPFIELD_FLIP_FOR_M_SWEEP = 0.05
N_TRIALS_M_SWEEP = 40

NOISE_SWEEP_M = 50
HOPFIELD_FLIP_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
OUR_SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N_TRIALS_NOISE_SWEEP = 60


def bipolarize(images):
    return np.where(images > 0.5, 1.0, -1.0).astype(np.float64)


def train_hopfield_outer(patterns_bipolar):
    Xb = patterns_bipolar
    N_local = Xb.shape[1]
    W = (Xb.T @ Xb) / N_local
    np.fill_diagonal(W, 0.0)
    return W


def train_hopfield_pinv(patterns_bipolar, ridge=RIDGE_LAMBDA):
    # W (N x N): reuses the SAME dual-form pseudo-inverse solver used for our own
    # network's heteroassociation, with target = input itself (auto-association).
    return train_pseudo_inverse_dual(patterns_bipolar, patterns_bipolar, ridge)


def train_hopfield_covariance(patterns_bipolar):
    """Tsodyks-Feigelman covariance rule: subtracts each NEURON's (pixel's) own mean
    activity level -- not a single global scalar -- before the outer-product sum.
    Real images have strong per-pixel structure (corner pixels are background in
    every image, center pixels vary) that a shared global bias correction misses;
    per-neuron centering is the standard, correct form of this rule."""
    a = patterns_bipolar.mean(axis=0)
    Xc = patterns_bipolar - a[None, :]
    N_local = Xc.shape[1]
    W = (Xc.T @ Xc) / N_local
    np.fill_diagonal(W, 0.0)
    return W


def hopfield_recall(W, cue, max_iters=MAX_HOPFIELD_ITERS):
    """Synchronous update, sign(W @ s). Ties (net input exactly 0) keep the neuron's
    CURRENT state rather than forcing +1 -- with mean-centered patterns, many pixels
    are constant across every stored image and get exactly-zero weights, so a fixed
    "tie -> +1" convention would systematically flip all of them wrong on step one."""
    s = cue.copy()
    for _ in range(max_iters):
        net = W @ s
        s_new = np.where(net == 0, s, np.sign(net))
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


def flip_bits(pattern, fraction, rng):
    n = len(pattern)
    nflip = int(round(fraction * n))
    if nflip == 0:
        return pattern.copy()
    cue = pattern.copy()
    idx = rng.choice(n, size=nflip, replace=False)
    cue[idx] *= -1
    return cue


def build_our_pipeline(M_local, rng_seed=SEED):
    rng0 = np.random.default_rng(rng_seed)
    pool = generate_unique(M_local, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M_local, N)
    conf = conflict_matrix()
    images, img_dim = load_dataset('mnist', M_local)

    rng = np.random.default_rng(rng_seed)
    W_SH = make_projection_hk(rng, H_val)
    H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images, RIDGE_LAMBDA)
    return dict(images=images, H_keys=H_keys, B_drive=B_drive, W_SH=W_SH, conf=conf,
                digits_true_all=digits_true_all, W_SI=W_SI, W_IS=W_IS)


def test_our_pipeline(store, M_local, sigma, n_trials, rng_test):
    images, H_keys, B_drive, W_SH, conf = (
        store['images'], store['H_keys'], store['B_drive'], store['W_SH'], store['conf'])
    W_SI, digits_true_all = store['W_SI'], store['digits_true_all']
    results = []
    for _ in range(n_trials):
        m = rng_test.integers(M_local)
        x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]
        h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
        h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
            H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
        results.append(int(np.all(digits_final == digits_true_all[m])))
    return np.mean(results)


def main():
    # =====================================================================
    # PANEL A: exact-recovery vs. M, fixed moderate noise
    # =====================================================================
    print(f'\n{"=" * 78}\nPANEL A: exact-recovery vs. M\n{"=" * 78}')
    rows_A = []
    for M_local in M_SWEEP:
        t0 = time.time()
        images, _ = load_dataset('mnist', M_local)
        images_bp = bipolarize(images)

        W_outer = train_hopfield_outer(images_bp)
        W_pinv = train_hopfield_pinv(images_bp)
        W_cov = train_hopfield_covariance(images_bp)

        rng_test = np.random.default_rng(SEED + 999)
        outer_results, pinv_results, cov_results = [], [], []
        for _ in range(N_TRIALS_M_SWEEP):
            m = rng_test.integers(M_local)
            cue = flip_bits(images_bp[m], HOPFIELD_FLIP_FOR_M_SWEEP, rng_test)
            outer_results.append(int(np.array_equal(hopfield_recall(W_outer, cue), images_bp[m])))
            pinv_results.append(int(np.array_equal(hopfield_recall(W_pinv, cue), images_bp[m])))
            cov_results.append(int(np.array_equal(hopfield_recall(W_cov, cue), images_bp[m])))
        outer_exact = np.mean(outer_results)
        pinv_exact = np.mean(pinv_results)
        cov_exact = np.mean(cov_results)

        our_store = build_our_pipeline(M_local)
        rng_test2 = np.random.default_rng(SEED + 999)
        our_exact = test_our_pipeline(our_store, M_local, OUR_SIGMA_FOR_M_SWEEP,
                                       N_TRIALS_M_SWEEP, rng_test2)

        rows_A.append({'M': M_local, 'hopfield_outer_exact': outer_exact,
                        'hopfield_pinv_exact': pinv_exact, 'hopfield_covariance_exact': cov_exact,
                        'our_network_exact': our_exact})
        print(f'  M={M_local:4d}  hopfield_outer={outer_exact:.2%}  '
              f'hopfield_pinv={pinv_exact:.2%}  hopfield_covariance={cov_exact:.2%}  '
              f'our_network={our_exact:.2%}  [{time.time()-t0:.1f}s]')

    df_A = pd.DataFrame(rows_A)
    df_A.to_csv('hopfield_vs_our_network_M_sweep.csv', index=False)

    # =====================================================================
    # PANEL B: exact-recovery vs. noise, fixed small M=50
    # =====================================================================
    print(f'\n{"=" * 78}\nPANEL B: exact-recovery vs. noise, M={NOISE_SWEEP_M}\n{"=" * 78}')
    images, _ = load_dataset('mnist', NOISE_SWEEP_M)
    images_bp = bipolarize(images)
    W_outer = train_hopfield_outer(images_bp)
    W_pinv = train_hopfield_pinv(images_bp)
    W_cov = train_hopfield_covariance(images_bp)

    rows_B_hop = []
    rng_test = np.random.default_rng(SEED + 999)
    for flip_frac in HOPFIELD_FLIP_LEVELS:
        outer_results, pinv_results, cov_results = [], [], []
        for _ in range(N_TRIALS_NOISE_SWEEP):
            m = rng_test.integers(NOISE_SWEEP_M)
            cue = flip_bits(images_bp[m], flip_frac, rng_test)
            outer_results.append(int(np.array_equal(hopfield_recall(W_outer, cue), images_bp[m])))
            pinv_results.append(int(np.array_equal(hopfield_recall(W_pinv, cue), images_bp[m])))
            cov_results.append(int(np.array_equal(hopfield_recall(W_cov, cue), images_bp[m])))
        rows_B_hop.append({'flip_fraction': flip_frac, 'hopfield_outer_exact': np.mean(outer_results),
                            'hopfield_pinv_exact': np.mean(pinv_results),
                            'hopfield_covariance_exact': np.mean(cov_results)})
        print(f'  flip={flip_frac:.2f}  hopfield_outer={np.mean(outer_results):.2%}  '
              f'hopfield_pinv={np.mean(pinv_results):.2%}  hopfield_covariance={np.mean(cov_results):.2%}')

    our_store = build_our_pipeline(NOISE_SWEEP_M)
    rows_B_ours = []
    rng_test2 = np.random.default_rng(SEED + 999)
    for sigma in OUR_SIGMA_LEVELS:
        exact = test_our_pipeline(our_store, NOISE_SWEEP_M, sigma, N_TRIALS_NOISE_SWEEP, rng_test2)
        rows_B_ours.append({'sigma': sigma, 'our_network_exact': exact})
        print(f'  sigma={sigma:.2f}  our_network={exact:.2%}')

    df_B_hop = pd.DataFrame(rows_B_hop)
    df_B_ours = pd.DataFrame(rows_B_ours)
    df_B_hop.to_csv('hopfield_vs_our_network_noise_sweep_hopfield.csv', index=False)
    df_B_ours.to_csv('hopfield_vs_our_network_noise_sweep_ours.csv', index=False)

    # =====================================================================
    # Plots
    # =====================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].plot(df_A.M, df_A.hopfield_outer_exact, 'o-', color='tab:red',
                 label=f'Classic Hopfield (outer-product), {HOPFIELD_FLIP_FOR_M_SWEEP:.0%} bit-flip')
    axes[0].plot(df_A.M, df_A.hopfield_covariance_exact, 'd-', color='tab:purple',
                 label=f'Covariance-rule Hopfield (bias-corrected), {HOPFIELD_FLIP_FOR_M_SWEEP:.0%} bit-flip')
    axes[0].plot(df_A.M, df_A.hopfield_pinv_exact, 's-', color='tab:orange',
                 label=f'Pseudo-inverse Hopfield, {HOPFIELD_FLIP_FOR_M_SWEEP:.0%} bit-flip')
    axes[0].plot(df_A.M, df_A.our_network_exact, '^-', color='tab:green',
                 label=f'Our network, sigma={OUR_SIGMA_FOR_M_SWEEP}')
    axes[0].set_xlabel('M (number of stored images)'); axes[0].set_ylabel('Exact recovery rate')
    axes[0].set_title('Exact recovery vs. M (fixed moderate noise)')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df_B_hop.flip_fraction * 100, df_B_hop.hopfield_outer_exact, 'o-', color='tab:red',
             label='Classic Hopfield (bottom axis: % bits flipped)')
    ax2.plot(df_B_hop.flip_fraction * 100, df_B_hop.hopfield_covariance_exact, 'd-', color='tab:purple',
             label='Covariance-rule Hopfield, bias-corrected (bottom axis)')
    ax2.plot(df_B_hop.flip_fraction * 100, df_B_hop.hopfield_pinv_exact, 's-', color='tab:orange',
             label='Pseudo-inverse Hopfield (bottom axis)')
    ax2.set_xlabel('Hopfield: % bits flipped'); ax2.set_ylabel('Exact recovery rate')
    ax3 = ax2.twiny()
    ax3.plot(df_B_ours.sigma, df_B_ours.our_network_exact, '^-', color='tab:green',
             label='Our network (top axis: Gaussian sigma)')
    ax3.set_xlabel('Our network: Gaussian noise sigma (NOT the same units as bottom axis)')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    ax2.set_title(f'Exact recovery vs. noise, fixed M={NOISE_SWEEP_M}\n(two different noise models -- see axis labels)')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('hopfield_vs_our_network_comparison.png', dpi=150)
    print('\nSaved hopfield_vs_our_network_comparison.png')


if __name__ == '__main__':
    main()
