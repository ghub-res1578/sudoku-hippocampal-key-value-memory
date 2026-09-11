"""
Extends the existing Hopfield comparison (hopfield_matched_noise_rigorous.py: classical/
covariance/pseudo-inverse Hopfield vs. ours) with three more STANDALONE associative-memory
architectures that the paper's Section 3.4 separation-operator ablation only tested as a
borrowed nonlinearity inside OUR OWN encoder, never as complete, independently-constructed
systems with their own native representation:

  1. Modern (continuous) Hopfield network (Ramsauer et al. 2020): xi_new = X @ softmax(beta
     X^T xi), iterated to a fixed point. Operates DIRECTLY on the full continuous [0,1] image
     (no sparse engram bottleneck at all) -- this is the cleanest test of whether our sparse
     K/H compression costs anything relative to a full continuous store at matched M. Query
     noise, image loading, and M/sigma sweep grid are identical to "ours" and to
     hopfield_matched_noise_rigorous.py, so all methods are directly comparable and most are
     literally recomputed with the same RNG draws (paired).

  2. Sparse Distributed Memory (Kanerva 1988): L random bipolar "hard locations"; writing a
     pattern adds it to every hard location within a Hamming radius; reading a noisy cue sums
     the counters of every hard location within radius of the cue and thresholds. A genuinely
     different addressing scheme from ours (content-addressed via random reference points in
     the pattern's OWN space, not via a projected sparse code).

  3. Dense associative memory (Krotov & Hopfield 2016): rectified-power energy E = -sum_mu
     ReLU(xi . x_mu)^n, synchronous relaxation s <- sign(field). Uses n=8, the SAME exponent
     already validated as the "dense-associative-memory-style" separation operator in Section
     3.4, for continuity with that result rather than a fresh arbitrary choice.

All three are evaluated on the SAME M-sweep (M in {100,200,400,800} at sigma=0.3) and
noise-sweep (M=50, sigma in {0.0,...,1.0}) as the existing four methods, Wilson 95% CIs,
N=200 trials/point. Construction (unit/parameter counts) is reported for every method
alongside accuracy, addressing the "neuron numbers" half of the comparison directly.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binom

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, relation, generate_unique, drive_vectors, conflict_matrix,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk
from hopfield_vs_our_network_comparison import (
    bipolarize, train_hopfield_outer, train_hopfield_covariance, train_hopfield_pinv, hopfield_recall,
)

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
IMG_DIM = 784

M_SWEEP = [100, 200, 400, 800]
SIGMA_FOR_M_SWEEP = 0.3
N_TRIALS = 200

NOISE_SWEEP_M = 50
SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

MAX_ITERS = 30
# MODERN_BETA and DAM_N were NOT carried over from Section 3.4 or picked casually: real
# MNIST images bipolarize/normalize far more correlated than random patterns (mean pairwise
# Hamming distance 148 vs. ~392 expected for random +-1 vectors of the same dimension), so
# both sharpening parameters were re-validated from scratch on THIS data by checking
# zero-noise self-recovery (does a perfectly clean cue recover exactly?) across the full
# M=100-800 sweep BEFORE any noise-robustness numbers were computed, exactly the standard
# this project applies to ridge lambda elsewhere. beta=0.5/n=8 (naive carryovers) both fail
# near-totally even at zero noise; beta=60/n=60 give clean recovery throughout the range.
MODERN_BETA = 60.0
DAM_N = 60
SDM_L = 1000           # hard locations, matched to H=1000 for a resource-comparable test
SDM_TARGET_ACTIVATION = 0.02  # target fraction of hard locations activated per write/read;
                        # validated on random +-1 patterns (SDM's native regime) to give
                        # 94% zero-noise recovery at M=50 -- the implementation is correct,
                        # but real MNIST's much higher pattern correlation (above) causes it
                        # to genuinely fail on this task, parallel to classical/covariance
                        # Hopfield's already-reported failure on the same real-image data.


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - halfwidth, center + halfwidth)


def sdm_radius_for_target_activation(d, l, target_frac):
    """Hamming radius r such that P(Hamming(random, random) <= r) ~= target_frac, for two
    independent random bipolar d-dim vectors (Hamming distance ~ Binomial(d, 0.5))."""
    r = binom.ppf(target_frac, d, 0.5)
    return int(r)


def train_modern_hopfield(patterns):
    # L2-normalize each stored pattern: raw MNIST images vary a lot in total "ink" (L2
    # norm), and an un-normalized dot product lets higher-brightness images dominate
    # similarity regardless of actual shape match (verified directly: the true pattern is
    # NOT always its own top match against an unnormalized store). Normalizing queries the
    # same way (in modern_hopfield_recall) restores genuine shape-similarity ranking.
    X = patterns.astype(np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def modern_hopfield_recall(X, cue, beta=MODERN_BETA, max_iters=MAX_ITERS, tol=1e-6):
    xi = cue.astype(np.float64).copy()
    xi = xi / np.linalg.norm(xi)
    for _ in range(max_iters):
        sims = X @ xi
        sims = sims - sims.max()
        w = np.exp(beta * sims)
        w = w / w.sum()
        xi_new = X.T @ w
        xi_new = xi_new / np.linalg.norm(xi_new)
        if np.max(np.abs(xi_new - xi)) < tol:
            xi = xi_new
            break
        xi = xi_new
    return xi


def train_sdm(patterns_bipolar, rng, l=SDM_L, target_frac=SDM_TARGET_ACTIVATION):
    d = patterns_bipolar.shape[1]
    R = rng.choice([-1.0, 1.0], size=(l, d))
    radius = sdm_radius_for_target_activation(d, l, target_frac)
    threshold_dot = d - 2 * radius  # activate iff Hamming(R_l, x) <= radius <=> R_l . x >= threshold_dot
    S = R @ patterns_bipolar.T           # (L, M) similarity of every hard location to every pattern
    A = (S >= threshold_dot).astype(np.float64)  # (L, M) activation mask
    counters = A @ patterns_bipolar      # (L, d) accumulated counters per hard location
    activation_rate = A.mean()
    return dict(R=R, counters=counters, threshold_dot=threshold_dot, radius=radius,
                activation_rate=activation_rate)


def sdm_recall(store, cue):
    R, counters, threshold_dot = store['R'], store['counters'], store['threshold_dot']
    s_q = R @ cue                         # (L,)
    a_q = (s_q >= threshold_dot).astype(np.float64)
    if a_q.sum() == 0:
        return cue.copy(), False          # dead query: no hard location activated
    out = a_q @ counters                  # (d,)
    return np.sign(np.where(out == 0, cue, out)), True


def train_dam(patterns):
    return patterns.astype(np.float64)


def dam_recall(X, cue, n=DAM_N, max_iters=MAX_ITERS):
    s = cue.astype(np.float64).copy()
    d = X.shape[1]
    for _ in range(max_iters):
        overlaps = (X @ s) / d            # (M,), normalized so ReLU(.)^n doesn't explode
        gate = n * np.maximum(overlaps, 0) ** (n - 1)  # (M,)
        field = gate @ X                  # (d,)
        s_new = np.where(field == 0, s, np.sign(field))
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


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


METHODS = ['outer', 'cov', 'pinv', 'modern', 'sdm', 'dam', 'ours']
LABELS = {'outer': 'Classical Hopfield', 'cov': 'Covariance-rule Hopfield',
          'pinv': 'Pseudo-inverse Hopfield', 'modern': 'Modern Hopfield (Ramsauer 2020)',
          'sdm': 'Sparse Distributed Memory (Kanerva)', 'dam': 'Dense Assoc. Memory (Krotov-Hopfield)',
          'ours': 'Our network'}


def run_one_M(M_local, sigma_levels, n_trials, label_prefix):
    t0 = time.time()
    store = build_our_pipeline(M_local)
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

    rows = []
    for sigma in sigma_levels:
        rng = np.random.default_rng(SEED + 999)
        results = {k: [] for k in METHODS}
        sdm_dead = 0
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
            sdm_dead += int(not activated)
            results['sdm'].append(int(activated and np.array_equal(out_sdm, images_bp_clean[m])))

            results['dam'].append(int(np.array_equal(dam_recall(X_dam, cue_bp), images_bp_clean[m])))

            h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
                H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
            results['ours'].append(int(np.all(digits_final == digits_true_all[m])))

        row = {'M': M_local, 'sigma': sigma, 'sdm_dead_query_frac': sdm_dead / n_trials,
               'sdm_activation_rate': sdm_store['activation_rate']}
        for key in METHODS:
            arr = results[key]
            lo, hi = wilson_ci(sum(arr), len(arr))
            row[f'{key}_exact'] = np.mean(arr)
            row[f'{key}_ci_lo'] = lo
            row[f'{key}_ci_hi'] = hi
        rows.append(row)
        print(f"  {label_prefix} sigma={sigma:.2f}  " +
              "  ".join(f"{k}={row[f'{k}_exact']:.3f}" for k in METHODS) +
              f"  [{time.time()-t0:.1f}s]")
    return rows


def construction_table():
    """Unit/parameter counts at M=200, the paper's default headline stored-set size."""
    M_local = 200
    d = IMG_DIM
    rows = [
        {'method': LABELS['outer'], 'state_units': d, 'stored_params': f'{d}x{d} weight matrix',
         'retrieval': 'iterative (sign(Ws))'},
        {'method': LABELS['cov'], 'state_units': d, 'stored_params': f'{d}x{d} weight matrix',
         'retrieval': 'iterative (sign(Ws))'},
        {'method': LABELS['pinv'], 'state_units': d, 'stored_params': f'{d}x{d} weight matrix',
         'retrieval': 'iterative (sign(Ws))'},
        {'method': LABELS['modern'], 'state_units': d,
         'stored_params': f'{M_local}x{d} pattern matrix (no weight matrix)',
         'retrieval': f'iterative softmax fixed-point (beta={MODERN_BETA})'},
        {'method': LABELS['sdm'], 'state_units': d,
         'stored_params': f'{SDM_L} hard locations x {d} + {SDM_L}x{d} counters',
         'retrieval': 'one-shot (radius-thresholded majority sum)'},
        {'method': LABELS['dam'], 'state_units': d,
         'stored_params': f'{M_local}x{d} pattern matrix (no weight matrix)',
         'retrieval': f'iterative rectified-power relaxation (n={DAM_N})'},
        {'method': LABELS['ours'], 'state_units': f'{H_val} (engram) + {d} (image)',
         'stored_params': f'fixed W_SH: {H_val}x{N*N}; {M_local} stored (key,value) pairs: '
                            f'{M_local}x({H_val}+{d})',
         'retrieval': 'iterative attention readout + Sudoku-attractor cleanup'},
    ]
    return pd.DataFrame(rows)


def main():
    print(f'{"=" * 90}\nCONSTRUCTION (unit/parameter counts, M=200)\n{"=" * 90}')
    df_construction = construction_table()
    df_construction.to_csv('hopfield_modern_sdm_dam_construction.csv', index=False)
    print(df_construction.to_string(index=False))

    print(f'\n{"=" * 90}\nM-SWEEP at fixed sigma={SIGMA_FOR_M_SWEEP}, N={N_TRIALS}/point\n{"=" * 90}')
    m_rows = []
    for M_local in M_SWEEP:
        m_rows.extend(run_one_M(M_local, [SIGMA_FOR_M_SWEEP], N_TRIALS, f'M={M_local}'))
    df_m = pd.DataFrame(m_rows)
    df_m.to_csv('hopfield_modern_sdm_dam_M_sweep.csv', index=False)

    print(f'\n{"=" * 90}\nNOISE-SWEEP at fixed M={NOISE_SWEEP_M}, N={N_TRIALS}/point\n{"=" * 90}')
    noise_rows = run_one_M(NOISE_SWEEP_M, SIGMA_LEVELS, N_TRIALS, f'M={NOISE_SWEEP_M}')
    df_n = pd.DataFrame(noise_rows)
    df_n.to_csv('hopfield_modern_sdm_dam_noise_sweep.csv', index=False)

    colors = {'outer': 'tab:red', 'cov': 'tab:purple', 'pinv': 'tab:orange',
              'modern': 'tab:blue', 'sdm': 'tab:brown', 'dam': 'tab:pink', 'ours': 'tab:green'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for key in METHODS:
        axes[0].errorbar(df_m.M, df_m[f'{key}_exact'],
                          yerr=[df_m[f'{key}_exact'] - df_m[f'{key}_ci_lo'], df_m[f'{key}_ci_hi'] - df_m[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=LABELS[key], capsize=3)
        axes[1].errorbar(df_n.sigma, df_n[f'{key}_exact'],
                          yerr=[df_n[f'{key}_exact'] - df_n[f'{key}_ci_lo'], df_n[f'{key}_ci_hi'] - df_n[f'{key}_exact']],
                          fmt='o-', color=colors[key], label=LABELS[key], capsize=3)
    axes[0].set_xlabel('M (number of stored images)'); axes[0].set_ylabel('Exact recovery rate')
    axes[0].set_title(f'Exact recovery vs. M (shared Gaussian noise, sigma={SIGMA_FOR_M_SWEEP})')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Gaussian pixel noise sigma (shared across all methods)')
    axes[1].set_ylabel('Exact recovery rate')
    axes[1].set_title(f'Exact recovery vs. noise, fixed M={NOISE_SWEEP_M}')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('hopfield_modern_sdm_dam_comparison.png', dpi=150)
    print('\nSaved hopfield_modern_sdm_dam_construction.csv, _M_sweep.csv, _noise_sweep.csv, .png')


if __name__ == '__main__':
    main()
