"""
Extends feedback_iterate_M2000.py's initial-vs-converged accuracy comparison to much
larger M (well past H=1000 hippocampal units), to see whether the noise-collapse
point and the iteration benefit keep holding steady, or whether engram collisions
between stored Sudokus eventually start mattering as M grows very large.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hippocampus_drive_readout import (
    H, K, N, SEED, SEED_TRIALS,
    generate_unique, relation, make_projection, make_engrams, drive_vectors,
    readout_attention, decode_digits, conflict_matrix, coincidence, clean_engram,
)

MEMORY_LOADS = [2000, 3000, 5000, 7000, 10000]
M_MAX = max(MEMORY_LOADS)
NOISE_LEVELS = [0.0, 0.20, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.70]
N_TRIALS = 50
MAX_ITERS = 20
COLLAPSE_THRESHOLD = 0.50

OUT_PREFIX = 'feedback_iterate_highM'
POOL_FILE = f'{OUT_PREFIX}_pool.npy'
PROJ_SEED = SEED + 2   # independent stream from feedback_iterate_M2000.py's encoder


def build_pool():
    try:
        pool = np.load(POOL_FILE)
        print(f'Loaded cached pool of {len(pool)} Sudokus from {POOL_FILE}')
    except FileNotFoundError:
        rng_pool = np.random.default_rng(SEED)
        print(f'Generating pool of {M_MAX} unique solved Sudokus '
              f'(~{M_MAX * 0.065:.0f}s expected)...')
        pool = generate_unique(M_MAX, rng_pool)
        np.save(POOL_FILE, pool)
    return pool


def iterate_feedback(H_keys, B_values, W_SH, conf, hq, true_digits, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    digits = decode_digits(readout_attention(H_keys, B_values, h_current))
    initial_acc = np.mean(digits == true_digits)
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_values, h_current)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH)
        key = h_next.tobytes()
        if key in visited:
            acc = np.mean(digits == true_digits)
            return {'iterations': it, 'converged': True,
                    'initial_acc': initial_acc, 'final_acc': acc}
        visited[key] = it
        h_current = h_next
    acc = np.mean(digits == true_digits)
    return {'iterations': max_iters, 'converged': False,
            'initial_acc': initial_acc, 'final_acc': acc}


def run():
    pool = build_pool()
    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 4)

    rows = []
    for M in MEMORY_LOADS:
        memories = pool[:M]
        C_all = np.asarray([relation(g).ravel() for g in memories], dtype=np.float32)
        H_keys = make_engrams(C_all, W_SH)
        B_values = drive_vectors(memories)
        digits_true = memories.reshape(M, N)

        n_trials = min(N_TRIALS, M)
        for noise in NOISE_LEVELS:
            targets = trng.choice(M, size=n_trials, replace=False)
            nflip = int(round(noise * H))
            for t in targets:
                hq = H_keys[t].copy()
                if nflip:
                    hq[trng.choice(H, size=nflip, replace=False)] ^= 1
                res = iterate_feedback(H_keys, B_values, W_SH, conf, hq, digits_true[t])
                rows.append({'M': M, 'noise_percent': noise * 100, **res})
            sub = [r for r in rows if r['M'] == M and r['noise_percent'] == noise * 100]
            print(f'  M={M:6d}  noise={noise:5.0%}  mean_iter={np.mean([r["iterations"] for r in sub]):5.2f}  '
                  f'initial={np.mean([r["initial_acc"] for r in sub]):7.4%}  '
                  f'final={np.mean([r["final_acc"] for r in sub]):7.4%}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def collapse_thresholds(df, threshold=COLLAPSE_THRESHOLD):
    rows = []
    for M in sorted(df.M.unique()):
        sub = df[df.M == M].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        for col in ['initial_acc', 'final_acc']:
            below = sub[sub[col] < threshold]
            thresh = below.noise_percent.min() if len(below) else np.nan
            rows.append({'M': M, 'condition': col, 'collapse_noise_percent': thresh})
    return pd.DataFrame(rows)


def plot_per_M_curves(df):
    Ms = sorted(df.M.unique())
    fig, axes = plt.subplots(1, len(Ms), figsize=(4.2 * len(Ms), 4), sharey=True)
    for ax, M in zip(axes, Ms):
        sub = df[df.M == M].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        ax.plot(sub.noise_percent, sub.initial_acc, 'o-', color='tab:orange',
                 label='initial', markersize=4)
        ax.plot(sub.noise_percent, sub.final_acc, 'o-', color='tab:green',
                 label='converged', markersize=4)
        ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'M={M}')
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel('Digit-cell retrieval accuracy')
    fig.supxlabel('Query noise (% of H bits flipped)')
    fig.suptitle('Initial vs converged accuracy at higher M')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_per_M_curves.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_per_M_curves.png')


def plot_collapse_threshold(cap_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'initial_acc': 'tab:orange', 'final_acc': 'tab:green'}
    for cond in ['initial_acc', 'final_acc']:
        sub = cap_df[cap_df.condition == cond].sort_values('M')
        ax.plot(sub.M, sub.collapse_noise_percent, 'o-', color=colors[cond], label=cond)
    ax.set_xscale('log')
    ax.set_xlabel('Number of stored memories (M)')
    ax.set_ylabel(f'Noise level (%) where accuracy first drops below {COLLAPSE_THRESHOLD:.0%}')
    ax.set_title('Collapse point at much larger M (beyond H=1000)')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_collapse_threshold.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_collapse_threshold.png')


if __name__ == '__main__':
    df = run()
    cap_df = collapse_thresholds(df)
    cap_df.to_csv(f'{OUT_PREFIX}_collapse_thresholds.csv', index=False)
    print('\nCollapse thresholds:')
    print(cap_df.pivot(index='M', columns='condition', values='collapse_noise_percent').to_string())
    plot_per_M_curves(df)
    plot_collapse_threshold(cap_df)
