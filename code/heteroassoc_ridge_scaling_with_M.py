"""
Does the ridge_lambda that fixes the capacity cliff at M=500 (lambda=0.1) keep working as
M grows further, or does the required lambda itself have to keep increasing with M?

Builds the (pool, engrams, images) ONCE per M (lambda-independent), then trains W_SI/W_IS
at several candidate lambdas for that same M, measuring both zero-noise fit (does more
regularization start costing clean-image accuracy?) and noise robustness at sigma=0.1
(does this amount of regularization still fully rescue recovery at this M?).
"""
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    H, N, K, SEED, TAU,
    relation, make_projection, make_engrams, drive_vectors, generate_unique,
    conflict_matrix,
)
from heteroassoc_vectorhash_generalized import (
    load_dataset, train_pseudo_inverse_dual, k_wta, add_noise, iterate_sudoku_attractor,
)

M_VALUES = [500, 800, 1200, 1600, 2000]
CANDIDATE_LAMBDAS = [0.1, 1.0, 10.0]
SIGMA_TEST = 0.1
N_TRIALS = 50


def build_base(M_local):
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)
    B_drive = drive_vectors(pool)
    conf = conflict_matrix()
    images, img_dim = load_dataset('mnist', M_local)
    return dict(pool=pool, W_SH=W_SH, H_keys=H_keys, B_drive=B_drive, conf=conf, images=images)


def main():
    rows = []
    for M_local in M_VALUES:
        t0 = time.time()
        base = build_base(M_local)
        images, H_keys, B_drive, W_SH, conf = (
            base['images'], base['H_keys'], base['B_drive'], base['W_SH'], base['conf'])
        digits_true_all = base['pool'].reshape(M_local, N)
        print(f'\n=== M={M_local} (base built in {time.time()-t0:.1f}s) ===')

        for lam in CANDIDATE_LAMBDAS:
            W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), lam)
            W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images, lam)

            h_zero_pred = np.array([k_wta(W_SI @ images[m]) for m in range(M_local)])
            zero_ov = np.mean([np.sum(h_zero_pred[m] * H_keys[m]) / K for m in range(M_local)])

            rng_test = np.random.default_rng(SEED + 999)
            trial_results = []
            for _ in range(N_TRIALS):
                m = rng_test.integers(M_local)
                x_noisy = add_noise(images[m:m + 1], SIGMA_TEST, rng_test)[0]
                h_initial = k_wta(W_SI @ x_noisy)
                h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                    H_keys, B_drive, W_SH, conf, h_initial)
                exact = int(np.all(digits_final == digits_true_all[m]))
                trial_results.append(exact)
            exact_rate = np.mean(trial_results)

            rows.append({'M': M_local, 'ridge_lambda': lam, 'zero_noise_overlap': zero_ov,
                         f'sigma_{SIGMA_TEST}_exact': exact_rate})
            print(f'  lambda={lam:<6g}  zero_noise_ov={zero_ov:.3f}  '
                  f'sigma={SIGMA_TEST}_exact={exact_rate:.2%}  [{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_ridge_scaling_with_M.csv', index=False)
    print('\nSaved heteroassoc_ridge_scaling_with_M.csv')


if __name__ == '__main__':
    main()
