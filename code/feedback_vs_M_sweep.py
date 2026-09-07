"""
Sweep both M (number of stored memories) and query noise, comparing the attention
readout with vs without the conflict-mask feedback/cleanup pass. Question: as more
memories are stored, does the noise level at which retrieval collapses shift (more
stored engrams = more competitors a corrupted query could be mistaken for), and does
the feedback loop's benefit shift with it?

Reuses the encoder/engram/readout/cleanup primitives from hippocampus_drive_readout.py
so the mechanics are identical to the single-M sweep already run there.
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

MEMORY_LOADS = [50, 100, 200, 500, 1000, 1500, 2000]
M_MAX = max(MEMORY_LOADS)
NOISE_LEVELS = [0.0, 0.20, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.80, 1.0]
N_TRIALS = 50
COLLAPSE_THRESHOLD = 0.50

OUT_PREFIX = 'feedback_vs_M_sweep'


def run():
    rng = np.random.default_rng(SEED)
    print(f'Generating pool of {M_MAX} unique solved Sudokus...')
    pool = generate_unique(M_MAX, rng)
    print('  done.')

    W_SH = make_projection(rng)   # fixed encoder, shared across all M
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 2)

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
            acc_no_fb, acc_fb = [], []
            for t in targets:
                hq = H_keys[t].copy()
                if nflip:
                    hq[trng.choice(H, size=nflip, replace=False)] ^= 1

                b1 = readout_attention(H_keys, B_values, hq)
                digits1 = decode_digits(b1)
                acc_no_fb.append(np.mean(digits1 == digits_true[t]))

                C_clean = coincidence(digits1, conf)
                h_clean = clean_engram(C_clean, W_SH)
                b2 = readout_attention(H_keys, B_values, h_clean)
                digits2 = decode_digits(b2)
                acc_fb.append(np.mean(digits2 == digits_true[t]))

            rows.append({
                'M': M, 'noise_percent': noise * 100,
                'no_feedback': np.mean(acc_no_fb),
                'with_feedback': np.mean(acc_fb),
            })
        print(f'  M={M:5d}  done  '
              f'(clean acc no_fb={rows[-len(NOISE_LEVELS)]["no_feedback"]:.3f})')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    print(f'\nSaved {OUT_PREFIX}.csv')
    return df


def collapse_thresholds(df, threshold=COLLAPSE_THRESHOLD):
    """For each M, the lowest tested noise level at which accuracy first drops
    below `threshold`, for both conditions."""
    rows = []
    for M in sorted(df.M.unique()):
        sub = df[df.M == M].sort_values('noise_percent')
        for col in ['no_feedback', 'with_feedback']:
            below = sub[sub[col] < threshold]
            thresh = below.noise_percent.min() if len(below) else np.nan
            rows.append({'M': M, 'condition': col, 'collapse_noise_percent': thresh})
    return pd.DataFrame(rows)


def plot_heatmaps(df):
    Ms = sorted(df.M.unique()); noises = sorted(df.noise_percent.unique())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    titles = ['no_feedback', 'with_feedback']
    for ax, col in zip(axes[:2], titles):
        grid = np.full((len(Ms), len(noises)), np.nan)
        for i, M in enumerate(Ms):
            for j, noise in enumerate(noises):
                row = df[(df.M == M) & (df.noise_percent == noise)]
                if len(row):
                    grid[i, j] = row[col].values[0]
        im = ax.imshow(grid, aspect='auto', vmin=0, vmax=1, cmap='viridis', origin='lower')
        ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:.0f}' for n in noises], rotation=45)
        ax.set_yticks(range(len(Ms))); ax.set_yticklabels(Ms)
        ax.set_xlabel('Query noise (%)'); ax.set_title(col)
    fig.colorbar(im, ax=axes[1], label='Digit-cell retrieval accuracy', shrink=0.8)

    # Delta heatmap
    ax = axes[2]
    grid = np.full((len(Ms), len(noises)), np.nan)
    for i, M in enumerate(Ms):
        for j, noise in enumerate(noises):
            row = df[(df.M == M) & (df.noise_percent == noise)]
            if len(row):
                grid[i, j] = row.with_feedback.values[0] - row.no_feedback.values[0]
    lim = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
    im2 = ax.imshow(grid, aspect='auto', vmin=-lim, vmax=lim, cmap='RdBu_r', origin='lower')
    ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:.0f}' for n in noises], rotation=45)
    ax.set_yticks(range(len(Ms))); ax.set_yticklabels(Ms)
    ax.set_xlabel('Query noise (%)'); ax.set_title('with_feedback - no_feedback')
    fig.colorbar(im2, ax=ax, label='Accuracy delta', shrink=0.8)

    axes[0].set_ylabel('Number of stored memories (M)')
    fig.suptitle('Feedback effect across M and query noise')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_heatmap.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_heatmap.png')


def plot_per_M_curves(df):
    """One noise-accuracy curve per M, no_feedback vs with_feedback, side by side."""
    Ms = sorted(df.M.unique())
    ncols = 4; nrows = int(np.ceil(len(Ms) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows),
                              sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, M in zip(axes, Ms):
        sub = df[df.M == M].sort_values('noise_percent')
        ax.plot(sub.noise_percent, sub.no_feedback, 'o-', color='tab:orange',
                 label='without feedback', markersize=4)
        ax.plot(sub.noise_percent, sub.with_feedback, 'o-', color='tab:green',
                 label='with feedback', markersize=4)
        ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'M={M}')
        ax.grid(alpha=0.3)
    for ax in axes[len(Ms):]:
        ax.axis('off')
    axes[0].legend(fontsize=8, loc='center left')
    fig.supxlabel('Query noise (% of H bits flipped)')
    fig.supylabel('Digit-cell retrieval accuracy')
    fig.suptitle('Noise-robustness curve at each M: with vs without feedback')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_per_M_curves.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_per_M_curves.png')


def plot_collapse_threshold(cap_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'no_feedback': 'tab:orange', 'with_feedback': 'tab:green'}
    for cond in ['no_feedback', 'with_feedback']:
        sub = cap_df[cap_df.condition == cond].sort_values('M')
        ax.plot(sub.M, sub.collapse_noise_percent, 'o-', color=colors[cond], label=cond)
    ax.set_xscale('log')
    ax.set_xlabel('Number of stored memories (M)')
    ax.set_ylabel(f'Noise level (%) where accuracy first drops below {COLLAPSE_THRESHOLD:.0%}')
    ax.set_title('Does the noise-collapse point shift as M grows?')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_collapse_threshold.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_collapse_threshold.png')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--replot':
        df = pd.read_csv(f'{OUT_PREFIX}.csv')
    else:
        df = run()
    cap_df = collapse_thresholds(df)
    cap_df.to_csv(f'{OUT_PREFIX}_collapse_thresholds.csv', index=False)
    print('\nCollapse thresholds (noise %% where accuracy first < %.0f%%):' % (COLLAPSE_THRESHOLD * 100))
    print(cap_df.pivot(index='M', columns='condition', values='collapse_noise_percent').to_string())
    plot_heatmaps(df)
    plot_collapse_threshold(cap_df)
    plot_per_M_curves(df)
