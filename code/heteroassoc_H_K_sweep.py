"""
Does noise-robust recall improve with a bigger engram (H) or a denser/sparser active
code (K)? Disentangles the two by testing H and K independently (and their ratio,
density = K/H, which classic sparse-coding theory says is the mechanistically relevant
quantity for pattern separation, not either value alone).

H and K are module-level constants in hippocampus_drive_readout.py, baked into several
functions' internals (make_projection, make_engrams, clean_engram) and into a frozen
default argument in heteroassoc_vectorhash_generalized.k_wta. Monkey-patching those
globals is fragile (default-argument values are bound at function-definition time, not
re-read later), so this script reimplements the H/K-dependent pieces as small, explicitly
parameterized local functions instead of importing the frozen versions, to guarantee
correctness at each (H, K) setting -- everything else (relation, generate_unique,
drive_vectors, readout_attention, decode_digits, conflict_matrix, coincidence) has no H/K
dependency and is safely imported as-is.
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, TAU, RIDGE_LAMBDA, DENSITY, INPUT_DIM,
    relation, generate_unique, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise

M = 200
MAX_ITERS = 20
NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N_TRIALS_PER_SIGMA = 80

CONFIGS = [
    (1000, 150, 'H=1000,K=150 (baseline, density=0.15)'),
    (1000, 300, 'H=1000,K=300 (same H, denser, density=0.30)'),
    (2000, 150, 'H=2000,K=150 (same K, sparser, density=0.075)'),
    (2000, 300, 'H=2000,K=300 (double both, same density=0.15)'),
    (500, 150, 'H=500,K=150 (smaller H, denser, density=0.30)'),
]


def make_projection_hk(rng, H_val):
    mask = rng.random((H_val, INPUT_DIM)) < DENSITY
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H_val, INPUT_DIM), dtype=np.float32)


def make_engrams_hk(C_all, W_SH, H_val, k):
    a = C_all @ W_SH.T
    a = a.toarray() if sparse.issparse(a) else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C_all.shape[0], H_val), np.uint8)
    Hs[np.arange(C_all.shape[0])[:, None], idx] = 1
    return Hs


def k_wta_hk(a, H_val, k):
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H_val, np.uint8); h[idx] = 1
    return h


def clean_engram_hk(C_masked, W_SH, H_val, k):
    a = np.asarray(W_SH @ C_masked.astype(np.float32).ravel()).ravel()
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H_val, np.uint8); h[idx] = 1
    return h


def iterate_sudoku_attractor_hk(H_keys, B_drive, W_SH, conf, h_query, H_val, k, max_iters=MAX_ITERS):
    h_current = h_query.copy()
    visited = {h_current.tobytes(): 0}
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_drive, h_current, tau=TAU)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram_hk(C_clean, W_SH, H_val, k)
        key = h_next.tobytes()
        if key in visited:
            return h_next, digits, it, True
        visited[key] = it
        h_current = h_next
    return h_current, digits, max_iters, False


def main():
    print(f'Building shared M={M} MNIST store (pool + images, reused across all H/K configs)...')
    rng0 = np.random.default_rng(SEED)
    pool = generate_unique(M, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M, N)
    conf = conflict_matrix()
    images, img_dim = load_dataset('mnist', M)

    all_rows = []
    for H_val, k, label in CONFIGS:
        print(f'\n{"=" * 78}\nCONFIG: {label}\n{"=" * 78}')
        t0 = time.time()
        rng = np.random.default_rng(SEED)  # same seed -> same Sudoku-relational structure, only W_SH size differs
        W_SH = make_projection_hk(rng, H_val)
        H_keys = make_engrams_hk(C_all, W_SH, H_val, k)

        W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32))
        W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images)

        h_clean_pred = np.array([k_wta_hk(W_SI @ images[m], H_val, k) for m in range(M)])
        zero_ov = np.mean([np.sum(h_clean_pred[m] * H_keys[m]) / k for m in range(M)])
        print(f'  built [{time.time()-t0:.1f}s], zero-noise image->engram overlap={zero_ov:.3f}')

        rng_test = np.random.default_rng(SEED + 999)
        for sigma in NOISE_SIGMAS:
            trial_rows = []
            for _ in range(N_TRIALS_PER_SIGMA):
                m = rng_test.integers(M)
                x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]
                h_initial = k_wta_hk(W_SI @ x_noisy, H_val, k)
                h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
                    H_keys, B_drive, W_SH, conf, h_initial, H_val, k)
                exact = int(np.all(digits_final == digits_true_all[m]))
                cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / k
                trial_rows.append((exact, cleaned_overlap))
            exact_rate = np.mean([t[0] for t in trial_rows])
            ov_rate = np.mean([t[1] for t in trial_rows])
            all_rows.append({'config': label, 'H': H_val, 'K': k, 'density': k / H_val,
                              'sigma': sigma, 'exact': exact_rate, 'cleaned_overlap': ov_rate})
            print(f'  sigma={sigma:.2f}  exact={exact_rate:.2%}  cleaned_ov={ov_rate:.3f}  '
                  f'[{time.time()-t0:.1f}s]')

    df = pd.DataFrame(all_rows)
    df.to_csv('heteroassoc_H_K_sweep.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for H_val, k, label in CONFIGS:
        sub = df[(df.H == H_val) & (df.K == k)]
        axes[0].plot(sub.sigma, sub.exact, 'o-', label=label)
        axes[1].plot(sub.sigma, sub.cleaned_overlap, 'o-', label=label)
    axes[0].set_xlabel('Image noise (sigma)'); axes[0].set_ylabel('Exact grid recovery rate')
    axes[0].set_title(f'Noise robustness vs. H and K (M={M}, MNIST)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Image noise (sigma)'); axes[1].set_ylabel('Cleaned engram overlap (fraction of K)')
    axes[1].set_title('Same, in overlap terms (not directly comparable across K)')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('heteroassoc_H_K_sweep.png', dpi=150)
    print('\nSaved heteroassoc_H_K_sweep.png')


if __name__ == '__main__':
    main()
