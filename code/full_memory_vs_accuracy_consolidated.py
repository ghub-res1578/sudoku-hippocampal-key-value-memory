"""
One consolidated memory (M) vs. accuracy comparison, ALL ten methods, on a single extended
M-grid (100-4000), single sigma=0.3, single trial protocol, FULLY PAIRED per trial (every
method sees the identical noisy query on a given trial) -- the strongest form of consistency,
stronger than merely sharing an M-grid. Earlier comparisons in this project computed subsets
of these methods on M=100-800 only (hopfield_modern_sdm_dam_comparison.py,
vectorhash_modular_scaffold_comparison.py, knn_at_sigma03_matched.py) or extended a single
method (ours, Modern Hopfield) to M=4000 separately. This script reruns everything together
on the identical extended grid so one plot can honestly show every method side by side with
no gaps and no risk of a subtle mismatch (different RNG order, different image draw) between
scripts run at different times.

Methods: classical/covariance/pseudo-inverse Hopfield, Modern Hopfield (beta=60, validated),
sparse distributed memory, dense associative memory (n=60, validated), k-NN, our network,
and the Vector-HaSH-style modular scaffold (one-shot and recurrent cleanup).
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
    wilson_ci, MODERN_BETA, DAM_N,
)
from vectorhash_modular_scaffold_comparison import (
    grid_codes_all, clean_grid_code, recurrent_attractor_decode, MODULI, ATTRACTOR_BETA,
)

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
M_SWEEP = [100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000]
SIGMA = 0.3
N_TRIALS = 200
SEED_TRIALS = SEED + 999

METHODS = ['outer', 'cov', 'pinv', 'modern', 'sdm', 'dam', 'knn', 'ours', 'vh_oneshot', 'vh_recurrent']
LABELS = {
    'outer': 'Classical Hopfield', 'cov': 'Covariance-rule Hopfield',
    'pinv': 'Pseudo-inverse Hopfield', 'modern': f'Modern Hopfield (beta={MODERN_BETA:g})',
    'sdm': 'Sparse Distributed Memory', 'dam': f'Dense Assoc. Memory (n={DAM_N:g})',
    'knn': 'k-NN (raw pixels)', 'ours': 'Our network',
    'vh_oneshot': 'Modular scaffold (one-shot)', 'vh_recurrent': 'Modular scaffold (recurrent)',
}


def build_all(M_local):
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

    images_bp_clean = bipolarize(images)
    W_outer = train_hopfield_outer(images_bp_clean)
    W_cov = train_hopfield_covariance(images_bp_clean)
    W_pinv = train_hopfield_pinv(images_bp_clean, ridge=1.0)
    X_modern = train_modern_hopfield(images)
    rng_sdm = np.random.default_rng(SEED + 12345)
    sdm_store = train_sdm(images_bp_clean, rng_sdm)
    X_dam = train_dam(images_bp_clean)

    codes = grid_codes_all(M_local, MODULI)
    codes_norm = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    W_SI_vh = train_pseudo_inverse_dual(images, codes, RIDGE_LAMBDA)

    return dict(images=images, images_bp_clean=images_bp_clean, H_keys=H_keys, B_drive=B_drive,
                W_SH=W_SH, conf=conf, digits_true_all=digits_true_all, W_SI=W_SI,
                W_outer=W_outer, W_cov=W_cov, W_pinv=W_pinv, X_modern=X_modern,
                sdm_store=sdm_store, X_dam=X_dam, codes=codes, codes_norm=codes_norm,
                W_SI_vh=W_SI_vh)


def run_one_M(M_local, sigma, n_trials):
    t0 = time.time()
    store = build_all(M_local)
    images, images_bp_clean = store['images'], store['images_bp_clean']
    H_keys, B_drive, W_SH, conf = store['H_keys'], store['B_drive'], store['W_SH'], store['conf']
    W_SI, digits_true_all = store['W_SI'], store['digits_true_all']
    codes, codes_norm, W_SI_vh = store['codes'], store['codes_norm'], store['W_SI_vh']

    rng = np.random.default_rng(SEED_TRIALS)
    results = {k: [] for k in METHODS}
    for _ in range(n_trials):
        m = rng.integers(M_local)
        x_noisy = add_noise(images[m:m + 1], sigma, rng)[0]
        cue_bp = bipolarize(x_noisy[None, :])[0]

        results['outer'].append(int(np.array_equal(hopfield_recall(store['W_outer'], cue_bp), images_bp_clean[m])))
        results['cov'].append(int(np.array_equal(hopfield_recall(store['W_cov'], cue_bp), images_bp_clean[m])))
        results['pinv'].append(int(np.array_equal(hopfield_recall(store['W_pinv'], cue_bp), images_bp_clean[m])))

        out_modern = modern_hopfield_recall(store['X_modern'], x_noisy, beta=MODERN_BETA)
        nn_idx = int(np.argmin(np.sum((store['X_modern'] - out_modern[None, :]) ** 2, axis=1)))
        results['modern'].append(int(nn_idx == m))

        out_sdm, activated = sdm_recall(store['sdm_store'], cue_bp)
        results['sdm'].append(int(activated and np.array_equal(out_sdm, images_bp_clean[m])))

        results['dam'].append(int(np.array_equal(dam_recall(store['X_dam'], cue_bp, n=DAM_N), images_bp_clean[m])))

        dists = np.sum((images - x_noisy[None, :]) ** 2, axis=1)
        results['knn'].append(int(np.argmin(dists) == m))

        h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
        h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
            H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
        results['ours'].append(int(np.all(digits_final == digits_true_all[m])))

        raw_vh = W_SI_vh @ x_noisy
        results['vh_oneshot'].append(int(np.array_equal(clean_grid_code(raw_vh, MODULI), codes[m])))
        m_decoded = recurrent_attractor_decode(codes_norm, raw_vh, ATTRACTOR_BETA)
        results['vh_recurrent'].append(int(m_decoded == m))

    row = {'M': M_local, 'sigma': sigma}
    for key in METHODS:
        arr = results[key]
        lo, hi = wilson_ci(sum(arr), len(arr))
        row[f'{key}_exact'] = np.mean(arr)
        row[f'{key}_ci_lo'] = lo
        row[f'{key}_ci_hi'] = hi
    print(f'M={M_local:5d}  ' + '  '.join(f'{k}={row[f"{k}_exact"]:.3f}' for k in METHODS) +
          f'  [{time.time()-t0:.1f}s]')
    return row


def main():
    print(f'{"=" * 100}\nFULL MEMORY-VS-ACCURACY, {len(METHODS)} methods, sigma={SIGMA}, '
          f'N={N_TRIALS}/point, M={M_SWEEP}\n{"=" * 100}')
    rows = [run_one_M(M_local, SIGMA, N_TRIALS) for M_local in M_SWEEP]
    df = pd.DataFrame(rows)
    df.to_csv('full_memory_vs_accuracy_consolidated.csv', index=False)
    print('\nSaved full_memory_vs_accuracy_consolidated.csv')

    colors = {
        'outer': 'tab:red', 'cov': 'tab:purple', 'pinv': 'tab:orange', 'modern': 'tab:blue',
        'sdm': 'tab:brown', 'dam': 'tab:pink', 'knn': 'black', 'ours': 'tab:green',
        'vh_oneshot': 'gray', 'vh_recurrent': 'tab:cyan',
    }
    linestyles = {k: ('--' if k == 'vh_oneshot' else '-') for k in METHODS}

    fig, ax = plt.subplots(figsize=(11, 7.5))
    for key in METHODS:
        ax.errorbar(df.M, df[f'{key}_exact'],
                    yerr=[df[f'{key}_exact'] - df[f'{key}_ci_lo'], df[f'{key}_ci_hi'] - df[f'{key}_exact']],
                    fmt='o' + linestyles[key], color=colors[key], label=LABELS[key], capsize=3,
                    markersize=5)
    ax.axvline(610, color='gray', linestyle=':', alpha=0.6, label='MNIST effective rank (~610)')
    ax.set_xlabel('M (number of stored images)')
    ax.set_ylabel('Exact recovery rate')
    ax.set_title(f'Memory (M) vs. accuracy, all {len(METHODS)} methods, shared Gaussian noise '
                 f'sigma={SIGMA}, N={N_TRIALS}/point, fully paired trials')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('full_memory_vs_accuracy_consolidated.png', dpi=150, bbox_inches='tight')
    print('Saved full_memory_vs_accuracy_consolidated.png')


if __name__ == '__main__':
    main()
