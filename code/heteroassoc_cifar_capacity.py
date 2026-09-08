"""
Same capacity diagnostic as heteroassoc_H_extends_capacity.py (image Gram-matrix
effective rank, zero-noise overlap, exact-recovery under a little noise, all vs. M),
but for CIFAR-100-grayscale (32x32=1024, natural-image textures) instead of MNIST
(28x28=784, clean digit strokes on black background) -- to see whether a dataset with
different pixel statistics and a different ambient dimension shifts the effective-rank
ceiling found for MNIST (~600, well below its 784 ambient dimension).
"""
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    N, SEED, TAU,
    relation, generate_unique, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence,
)
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk, iterate_sudoku_attractor_hk
from heteroassoc_H_extends_capacity import gram_diagnostics

M_VALUES = [500, 800, 1200, 1600, 2000]
H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
SIGMA_TEST = 0.1
N_TRIALS = 50


def main():
    rows = []
    for M_local in M_VALUES:
        t0 = time.time()
        rng0 = np.random.default_rng(SEED)
        pool = generate_unique(M_local, rng0)
        C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
        B_drive = drive_vectors(pool)
        digits_true_all = pool.reshape(M_local, N)
        conf = conflict_matrix()
        images, img_dim = load_dataset('cifar100_gray', M_local)

        img_rank, img_smallest, img_cond = gram_diagnostics(images)

        rng = np.random.default_rng(SEED)
        W_SH = make_projection_hk(rng, H_val)
        H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)

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

        rows.append({'M': M_local, 'img_dim': img_dim, 'image_gram_rank': img_rank,
                     'image_gram_smallest_eig': img_smallest, 'image_gram_cond': img_cond,
                     'zero_noise_overlap': zero_ov, f'sigma_{SIGMA_TEST}_exact': exact_rate})
        print(f'M={M_local}: image_dim={img_dim}  IMAGE Gram rank={img_rank}/{M_local}  '
              f'smallest_eig={img_smallest:.3e}  cond={img_cond:.3e}  |  '
              f'zero_noise_ov={zero_ov:.3f}  sigma={SIGMA_TEST}_exact={exact_rate:.2%}  '
              f'[{time.time()-t0:.1f}s]')

    pd.DataFrame(rows).to_csv('heteroassoc_cifar_capacity.csv', index=False)
    print('\nSaved heteroassoc_cifar_capacity.csv')


if __name__ == '__main__':
    main()
