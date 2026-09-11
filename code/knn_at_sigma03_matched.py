"""
k-NN baseline at sigma=0.3, M in {100,200,400,800} -- the EXACT condition used by
hopfield_modern_sdm_dam_comparison.py's M-sweep (matching Table 10's sigma too), so the
k-NN number can sit directly alongside classical/covariance/pseudo-inverse/modern Hopfield,
SDM, DAM, and our own pipeline in one table. Uses the identical pool-construction path
(generate_unique with rng0=SEED, load_dataset('mnist', M)) as build_our_pipeline() in that
script, so the stored image set is identical for each M.
"""
import time
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import SEED, generate_unique, relation
from heteroassoc_vectorhash_generalized import load_dataset, add_noise

M_SWEEP = [100, 200, 400, 800]
SIGMA = 0.3
N_TRIALS = 200


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
    t0 = time.time()
    for M_local in M_SWEEP:
        rng0 = np.random.default_rng(SEED)
        _ = generate_unique(M_local, rng0)   # advance rng identically to build_our_pipeline
        images, img_dim = load_dataset('mnist', M_local)

        rng = np.random.default_rng(SEED + 999)
        results = []
        for _ in range(N_TRIALS):
            m = rng.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], SIGMA, rng)[0]
            dists = np.sum((images - x_noisy[None, :]) ** 2, axis=1)
            nn_idx = np.argmin(dists)
            results.append(int(nn_idx == m))

        lo, hi = wilson_ci(sum(results), N_TRIALS)
        rows.append({'M': M_local, 'sigma': SIGMA, 'knn_exact': np.mean(results),
                     'knn_ci_lo': lo, 'knn_ci_hi': hi})
        print(f'M={M_local:4d}  sigma={SIGMA}  knn_exact={np.mean(results):.3f} '
              f'[{lo:.3f},{hi:.3f}]  [{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('knn_at_sigma03_matched.csv', index=False)
    print('\nSaved knn_at_sigma03_matched.csv')
    return df


if __name__ == '__main__':
    main()
