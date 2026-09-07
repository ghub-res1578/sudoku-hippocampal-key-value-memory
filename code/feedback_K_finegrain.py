"""
Higher-resolution, higher-trial-count re-check of the K=150 vs 300 vs 400 rescue
effect in the 28-42% noise transition band. feedback_vs_K_sweep.py used a coarse
5% noise grid and only 50 trials/point -- enough to see the pattern but not enough
to pin down the crossover cleanly. This reuses that script's tau-corrected engram/
readout/iterate_feedback machinery directly, just with a finer grid and 6x the trials.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hippocampus_drive_readout import H, N, SEED_TRIALS, relation, make_projection, drive_vectors, conflict_matrix
from feedback_vs_K_sweep import engrams_for_k, iterate_feedback, PROJ_SEED

M = 2000
K_VALUES = [150, 300, 400]
NOISE_LEVELS = [round(x, 2) for x in np.arange(0.28, 0.421, 0.01)]  # 28%-42%, 1% steps
N_TRIALS = 300
MAX_ITERS = 20

OUT_PREFIX = 'feedback_K_finegrain'
POOL_FILE = 'feedback_iterate_M2000_pool.npy'


def run():
    pool = np.load(POOL_FILE)[:M]
    print(f'Loaded {len(pool)} Sudokus from {POOL_FILE}')
    rng_proj = np.random.default_rng(PROJ_SEED)   # same encoder as feedback_vs_K_sweep.py
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    a = C_all @ W_SH.T
    a = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)
    B_values = drive_vectors(pool)
    digits_true = pool.reshape(M, N)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 6)

    rows = []
    for k in K_VALUES:
        H_keys = engrams_for_k(a, k)
        n_trials = min(N_TRIALS, M)
        for noise in NOISE_LEVELS:
            targets = trng.choice(M, size=n_trials, replace=False)
            nflip = int(round(noise * H))
            for t in targets:
                hq = H_keys[t].copy()
                if nflip:
                    hq[trng.choice(H, size=nflip, replace=False)] ^= 1
                res = iterate_feedback(H_keys, B_values, W_SH, conf, hq, digits_true[t], k, max_iters=MAX_ITERS)
                rows.append({'K': k, 'noise_percent': noise * 100, **res})
            sub = [r for r in rows if r['K'] == k and r['noise_percent'] == noise * 100]
            init_vals = [r['initial_acc'] for r in sub]; final_vals = [r['final_acc'] for r in sub]
            print(f'  K={k:4d}  noise={noise:5.1%}  '
                  f'initial={np.mean(init_vals):7.4%}+-{np.std(init_vals) / np.sqrt(len(sub)):.4%}  '
                  f'final={np.mean(final_vals):7.4%}+-{np.std(final_vals) / np.sqrt(len(sub)):.4%}  n={len(sub)}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def plot(df):
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = {150: 'tab:blue', 300: 'tab:orange', 400: 'tab:green'}
    for k in K_VALUES:
        sub = df[df.K == k].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'),
            initial_se=('initial_acc', lambda x: x.std() / np.sqrt(len(x))),
            final_acc=('final_acc', 'mean'),
            final_se=('final_acc', lambda x: x.std() / np.sqrt(len(x))),
        ).reset_index()
        ax.errorbar(sub.noise_percent, sub.initial_acc, yerr=sub.initial_se, fmt='o--',
                     color=colors[k], alpha=0.5, capsize=2, label=f'K={k} initial')
        ax.errorbar(sub.noise_percent, sub.final_acc, yerr=sub.final_se, fmt='o-',
                     color=colors[k], capsize=2, label=f'K={k} converged')
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'M={M}: fine-grained transition band, K=150/300/400, n={N_TRIALS} trials/point (±SE)')
    ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')


if __name__ == '__main__':
    df = run()
    plot(df)
