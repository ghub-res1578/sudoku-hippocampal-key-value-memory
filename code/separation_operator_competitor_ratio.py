"""
Reproduces the competitor-ratio numbers cited in Section 3.4's mechanistic explanation
for why the rectified-polynomial (dense-associative-memory-style) separation operator
collapses earlier than softmax, despite comparable zero-noise selectivity: even at zero
noise, the best-competing WRONG memory among the M=200 stored grids already reaches a
similarity within a small factor of the true match's -- far tighter than the "naive"
factor predicted from the mean chance-overlap floor alone, an extreme-value effect of
having M-1 competitors -- and this ratio shrinks further toward 1 as query noise grows.

Reuses the identical M=200 engram store and bit-flip noise convention as
separation_operator_ablation.py (same SEED, same build_store()), so the numbers here
are directly comparable to that script's noise sweep.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import H, K, M
from separation_operator_ablation import build_store

NOISE_LEVELS = [0.0, 0.20, 0.30]
SEED_TRIALS = 24680


def main():
    print('Rebuilding the M=200 engram store (same SEED as separation_operator_ablation.py)...')
    W_SH, H_keys, B_values, digits_true = build_store()
    H_keys_f = H_keys.astype(np.float32)

    naive_factor = H / K
    print(f'\nNaive factor (H/K, i.e. true overlap K vs. mean chance-competitor overlap K^2/H): '
          f'{naive_factor:.2f}')

    rng = np.random.default_rng(SEED_TRIALS)
    rows = []
    for noise in NOISE_LEVELS:
        nflip = int(round(noise * H))
        ratios = []
        for t in range(M):
            hq = H_keys[t].copy()
            if nflip:
                hq[rng.choice(H, size=nflip, replace=False)] ^= 1
            sim = H_keys_f @ hq.astype(np.float32)
            true_sim = sim[t]
            best_wrong = np.max(np.delete(sim, t))
            ratios.append(true_sim / max(best_wrong, 1e-9))

        mean_ratio = float(np.mean(ratios))
        rows.append({'noise_percent': noise * 100, 'mean_competitor_ratio': mean_ratio})
        print(f'  noise={noise:5.0%}  mean competitor ratio (true/best-wrong) = {mean_ratio:.2f}  '
              f'(n={M} stored items, exhaustive)')

    df = pd.DataFrame(rows)
    df.to_csv('separation_operator_competitor_ratio.csv', index=False)
    print(f'\nFor comparison, the naive (mean-competitor, not best-of-{M-1}) factor is '
          f'{naive_factor:.2f} -- the empirical zero-noise ratio above is far tighter, the '
          f'extreme-value effect described in Section 3.4.')
    print('\nSaved separation_operator_competitor_ratio.csv')


if __name__ == '__main__':
    main()
