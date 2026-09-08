"""
Does a bigger engram (H) push the M-scaling capacity cliff further out?

There are actually TWO separate capacity ceilings in the bidirectional heteroassociation:
  1. Image -> engram (W_SI): dual-form pseudo-inverse solves an M x M system built from
     the IMAGE Gram matrix X X^T (X is M x IMG_DIM=784). Its rank is bounded by
     min(M, 784) -- entirely a function of the image side. H does not appear in this
     matrix at all, so this ceiling should NOT move when H changes.
  2. Engram -> image (W_IS): the same dual-form trick inverts an M x M system built from
     the ENGRAM Gram matrix H_keys H_keys.T. Its rank is bounded by min(M, H) -- this
     ceiling is directly set by H, so a bigger H should push it out.

This script measures, for M = 500/800/1200/1600/2000 and H = 1000 vs. 2000 (K scaled to
keep density=0.15 in both): (a) exact recovery and zero-noise overlap at a fixed ridge
(1.0, a reasonable value from the earlier ridge-scaling sweep), and (b) the two Gram
matrices' rank/conditioning directly, to see mechanistically which ceiling is actually
binding at each M.
"""
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, TAU, INPUT_DIM,
    relation, generate_unique, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk

M_VALUES = [500, 800, 1200, 1600, 2000]
H_CONFIGS = [(1000, 150), (2000, 300)]
RIDGE_LAMBDA = 1.0
SIGMA_TEST = 0.1
N_TRIALS = 50


def gram_diagnostics(X):
    Xm = X.astype(np.float64)
    Gram = Xm @ Xm.T
    eigvals = np.clip(np.linalg.eigvalsh(Gram), 0, None)
    eff_rank = int(np.sum(eigvals > 1e-8 * eigvals.max()))
    smallest_nonzero = eigvals[eigvals > 1e-8 * eigvals.max()].min() if eff_rank > 0 else 0.0
    cond = eigvals.max() / max(smallest_nonzero, 1e-300)
    return eff_rank, smallest_nonzero, cond


def main():
    rows, gram_rows = [], []
    for M_local in M_VALUES:
        t0 = time.time()
        rng0 = np.random.default_rng(SEED)
        pool = generate_unique(M_local, rng0)
        C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
        B_drive = drive_vectors(pool)
        digits_true_all = pool.reshape(M_local, N)
        conf = conflict_matrix()
        images, img_dim = load_dataset('mnist', M_local)
        print(f'\n=== M={M_local} (shared pool/images built in {time.time()-t0:.1f}s) ===')

        img_rank, img_smallest, img_cond = gram_diagnostics(images)
        print(f'  IMAGE Gram: rank={img_rank}/{M_local}  smallest_nonzero_eig={img_smallest:.3e}  '
              f'cond={img_cond:.3e}  (H-independent -- should be identical for both H below)')

        for H_val, K_val in H_CONFIGS:
            rng = np.random.default_rng(SEED)
            W_SH = make_projection_hk(rng, H_val)
            H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)

            eng_rank, eng_smallest, eng_cond = gram_diagnostics(H_keys.astype(np.float64))

            W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)
            W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images, RIDGE_LAMBDA)

            h_zero_pred = np.array([k_wta_hk(W_SI @ images[m], H_val, K_val) for m in range(M_local)])
            zero_ov = np.mean([np.sum(h_zero_pred[m] * H_keys[m]) / K_val for m in range(M_local)])

            rng_test = np.random.default_rng(SEED + 999)
            exact_results = []
            for _ in range(N_TRIALS):
                m = rng_test.integers(M_local)
                x_noisy = add_noise(images[m:m + 1], SIGMA_TEST, rng_test)[0]
                h_initial = k_wta_hk(W_SI @ x_noisy, H_val, K_val)
                h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor_hk(
                    H_keys, B_drive, W_SH, conf, h_initial, H_val, K_val)
                exact_results.append(int(np.all(digits_final == digits_true_all[m])))
            exact_rate = np.mean(exact_results)

            gram_rows.append({'M': M_local, 'H': H_val, 'K': K_val,
                               'engram_gram_rank': eng_rank, 'engram_gram_smallest_eig': eng_smallest,
                               'engram_gram_cond': eng_cond})
            rows.append({'M': M_local, 'H': H_val, 'K': K_val,
                         'zero_noise_overlap': zero_ov, f'sigma_{SIGMA_TEST}_exact': exact_rate})
            print(f'  H={H_val},K={K_val}: ENGRAM Gram rank={eng_rank}/{M_local}  '
                  f'smallest_nonzero_eig={eng_smallest:.3e}  cond={eng_cond:.3e}  |  '
                  f'zero_noise_ov={zero_ov:.3f}  sigma={SIGMA_TEST}_exact={exact_rate:.2%}  '
                  f'[{time.time()-t0:.1f}s]')

    pd.DataFrame(rows).to_csv('heteroassoc_H_extends_capacity.csv', index=False)
    pd.DataFrame(gram_rows).to_csv('heteroassoc_H_extends_capacity_gram.csv', index=False)
    print('\nSaved heteroassoc_H_extends_capacity.csv and heteroassoc_H_extends_capacity_gram.csv')


if __name__ == '__main__':
    main()
