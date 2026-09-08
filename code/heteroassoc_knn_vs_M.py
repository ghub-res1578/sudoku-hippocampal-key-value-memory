"""
Does our pipeline eventually catch up to the raw-pixel nearest-neighbor baseline as M
grows? The M=200 comparison (Table 7/Section 4.3) found nearest-neighbor significantly
beats our pipeline -- but that was only tested at one M. Nearest-neighbor's own accuracy
is not immune to M (more candidates to confuse a noisy query with), so this checks whether
the GAP between the two methods shrinks, stays flat, or grows as M scales from 200 to 2000,
at a fixed, informative noise level (sigma=0.4, matching Table 7's most differentiated
tested point) and properly ridge-regularized weights (lambda=1.0, not the capacity-cliff
default) so the M-scaling result isn't confounded with the already-diagnosed regularization
artifact from Section 4.4.
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
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise, k_wta, iterate_sudoku_attractor

M_VALUES = [200, 500, 800, 1200, 1600, 2000]
SIGMA = 0.4
N_TRIALS = 200
RIDGE_LAMBDA = 1.0


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - halfwidth, center + halfwidth)


def main():
    rows = []
    for M_local in M_VALUES:
        t0 = time.time()
        rng0 = np.random.default_rng(SEED)
        pool = generate_unique(M_local, rng0)
        C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
        from hippocampus_drive_readout import make_projection, make_engrams
        W_SH = make_projection(rng0)
        H_keys = make_engrams(C_all, W_SH, 150)
        B_drive = drive_vectors(pool)
        digits_true_all = pool.reshape(M_local, N)
        conf = conflict_matrix()
        images, img_dim = load_dataset('mnist', M_local)
        W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)

        rng = np.random.default_rng(SEED + 999)
        ours_results, knn_results = [], []
        for _ in range(N_TRIALS):
            m = rng.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], SIGMA, rng)[0]

            h_initial = k_wta(W_SI @ x_noisy)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            ours_results.append(int(np.all(digits_final == digits_true_all[m])))

            dists = np.sum((images - x_noisy[None, :]) ** 2, axis=1)
            nn_idx = np.argmin(dists)
            knn_results.append(int(nn_idx == m))

        b = sum(1 for o, k in zip(ours_results, knn_results) if o == 1 and k == 0)
        c = sum(1 for o, k in zip(ours_results, knn_results) if o == 0 and k == 1)
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0

        ours_mean = np.mean(ours_results); knn_mean = np.mean(knn_results)
        ours_lo, ours_hi = wilson_ci(sum(ours_results), N_TRIALS)
        knn_lo, knn_hi = wilson_ci(sum(knn_results), N_TRIALS)
        gap = knn_mean - ours_mean

        rows.append({'M': M_local, 'ours_exact': ours_mean, 'ours_ci_lo': ours_lo, 'ours_ci_hi': ours_hi,
                     'knn_exact': knn_mean, 'knn_ci_lo': knn_lo, 'knn_ci_hi': knn_hi,
                     'gap_knn_minus_ours': gap, 'mcnemar_p': pval})
        print(f'M={M_local:4d}  ours={ours_mean:.3f} [{ours_lo:.3f},{ours_hi:.3f}]  '
              f'knn={knn_mean:.3f} [{knn_lo:.3f},{knn_hi:.3f}]  gap={gap:+.3f}  '
              f'McNemar p={pval:.4f}  [{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_knn_vs_M.csv', index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(df.M, df.ours_exact, yerr=[df.ours_exact - df.ours_ci_lo, df.ours_ci_hi - df.ours_exact],
                fmt='o-', color='tab:green', label='Our network', capsize=3)
    ax.errorbar(df.M, df.knn_exact, yerr=[df.knn_exact - df.knn_ci_lo, df.knn_ci_hi - df.knn_exact],
                fmt='s-', color='tab:blue', label='Raw-pixel k-NN baseline', capsize=3)
    ax.set_xlabel('M (number of stored images)'); ax.set_ylabel('Exact recovery rate')
    ax.set_title(f'Does the k-NN gap close as M grows? (sigma={SIGMA}, N={N_TRIALS}/point)')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('heteroassoc_knn_vs_M.png', dpi=150)
    print('\nSaved heteroassoc_knn_vs_M.png')


if __name__ == '__main__':
    main()
