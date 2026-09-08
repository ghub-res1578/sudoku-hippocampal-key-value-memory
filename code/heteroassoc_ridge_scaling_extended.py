"""
Extends heteroassoc_ridge_scaling_with_M.py's M range (which stopped at 2000) further, to
check whether a single fixed ridge value keeps working indefinitely as M grows, or whether
the required lambda keeps climbing without bound. Reuses that script's build_base helper.
"""
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from heteroassoc_vectorhash_generalized import train_pseudo_inverse_dual, k_wta, add_noise, iterate_sudoku_attractor
from heteroassoc_ridge_scaling_with_M import build_base, N, SEED

M_VALUES = [2500, 3000, 4000]
CANDIDATE_LAMBDAS = [1.0, 10.0, 50.0, 100.0]
SIGMA_TEST = 0.1
N_TRIALS = 50


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
            zero_ov = np.mean([np.sum(h_zero_pred[m] * H_keys[m]) / 150 for m in range(M_local)])

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
    df.to_csv('heteroassoc_ridge_scaling_extended.csv', index=False)
    print('\nSaved heteroassoc_ridge_scaling_extended.csv')


if __name__ == '__main__':
    main()
