"""
Hold M=2000 and K=150 fixed, sweep the Sudoku-to-hippocampus connection density
(the fraction of nonzero entries in W_SH, currently 2%), and compare accuracy
before (single readout pass) vs after (attractor dynamics iterated to convergence)
at each density.

A diagnostic check (self/competitor engram-similarity scale at clean queries) found
density does NOT need the TAU recalibration that K did: self-overlap is always K=150
by construction, and mean competitor overlap stays ~30-34 across density 0.2%-100%
with TAU=2 giving ~100% confidence at every density tested. So any density-dependent
effect found here should reflect the feedback loop's re-encoding step (projecting a
partially-corrected coincidence matrix back through the SAME W_SH under noise), not
a baseline-separability or softmax-calibration artifact.

Densities are drawn from a SEEDED rng re-initialized fresh per density value, so
lower-density masks are literal subsets of higher-density masks (nested connectivity,
like the nested M-subsets used elsewhere in this project) -- a cleaner comparison
than independently-redrawn masks per density.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse

from hippocampus_drive_readout import (
    H, K, N, INPUT_DIM, SEED, SEED_TRIALS,
    relation, drive_vectors, readout_attention, decode_digits, conflict_matrix, coincidence,
)

M = 2000
DENSITY_VALUES = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
N_TRIALS = 50
MAX_ITERS = 20
COLLAPSE_THRESHOLD = 0.50

OUT_PREFIX = 'feedback_vs_density_sweep'
POOL_FILE = 'feedback_iterate_M2000_pool.npy'
PROJ_SEED = SEED + 20


def make_projection(rng, density):
    mask = rng.random((H, INPUT_DIM)) < density
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H, INPUT_DIM), dtype=np.float32)


def make_engrams(C, W, k=K):
    a = C @ W.T; a = a.toarray() if sparse.issparse(a) else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C.shape[0], H), np.uint8)
    Hs[np.arange(C.shape[0])[:, None], idx] = 1
    return Hs


def clean_engram(C_masked, W, k=K):
    a = np.asarray(W @ C_masked.astype(np.float32).ravel()).ravel()
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


def iterate_feedback(H_keys, B_values, W, conf, hq, true_digits, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    digits = decode_digits(readout_attention(H_keys, B_values, h_current))
    initial_acc = np.mean(digits == true_digits)
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_values, h_current)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W)
        key = h_next.tobytes()
        if key in visited:
            acc = np.mean(digits == true_digits)
            return {'iterations': it, 'converged': True, 'initial_acc': initial_acc, 'final_acc': acc}
        visited[key] = it
        h_current = h_next
    acc = np.mean(digits == true_digits)
    return {'iterations': max_iters, 'converged': False, 'initial_acc': initial_acc, 'final_acc': acc}


def run():
    pool = np.load(POOL_FILE)[:M]
    print(f'Loaded {len(pool)} Sudokus from {POOL_FILE}')
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    B_values = drive_vectors(pool)
    digits_true = pool.reshape(M, N)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 7)

    rows = []
    for density in DENSITY_VALUES:
        rng_proj = np.random.default_rng(PROJ_SEED)   # fresh each time -> nested masks across density
        W = make_projection(rng_proj, density)
        avg_conn = W.getnnz(axis=1).mean()
        H_keys = make_engrams(C_all, W)
        n_trials = min(N_TRIALS, M)
        for noise in NOISE_LEVELS:
            targets = trng.choice(M, size=n_trials, replace=False)
            nflip = int(round(noise * H))
            for t in targets:
                hq = H_keys[t].copy()
                if nflip:
                    hq[trng.choice(H, size=nflip, replace=False)] ^= 1
                res = iterate_feedback(H_keys, B_values, W, conf, hq, digits_true[t])
                rows.append({'density': density, 'noise_percent': noise * 100, **res})
            sub = [r for r in rows if r['density'] == density and r['noise_percent'] == noise * 100]
            print(f'  density={density:6.3f} (avg_conn/unit={avg_conn:6.1f})  noise={noise:5.0%}  '
                  f'mean_iter={np.mean([r["iterations"] for r in sub]):5.2f}  '
                  f'initial={np.mean([r["initial_acc"] for r in sub]):7.4%}  '
                  f'final={np.mean([r["final_acc"] for r in sub]):7.4%}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def collapse_thresholds(df, threshold=COLLAPSE_THRESHOLD):
    rows = []
    for density in sorted(df.density.unique()):
        sub = df[df.density == density].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        for col in ['initial_acc', 'final_acc']:
            below = sub[sub[col] < threshold]
            thresh = below.noise_percent.min() if len(below) else np.nan
            rows.append({'density': density, 'condition': col, 'collapse_noise_percent': thresh})
    return pd.DataFrame(rows)


def plot_per_density_curves(df):
    densities = sorted(df.density.unique())
    ncols = 5; nrows = int(np.ceil(len(densities) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, density in zip(axes, densities):
        sub = df[df.density == density].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        ax.plot(sub.noise_percent, sub.initial_acc, 'o-', color='tab:orange', label='initial', markersize=4)
        ax.plot(sub.noise_percent, sub.final_acc, 'o-', color='tab:green', label='converged', markersize=4)
        ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'density={density:.1%}')
        ax.grid(alpha=0.3)
    for ax in axes[len(densities):]:
        ax.axis('off')
    axes[0].legend(fontsize=8)
    fig.supxlabel('Query noise (% of H bits flipped)')
    fig.supylabel('Digit-cell retrieval accuracy')
    fig.suptitle(f'M={M}, K={K}: initial vs converged accuracy across connection density')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_per_density_curves.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_per_density_curves.png')


def plot_collapse_vs_density(cap_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'initial_acc': 'tab:orange', 'final_acc': 'tab:green'}
    for cond in ['initial_acc', 'final_acc']:
        sub = cap_df[cap_df.condition == cond].sort_values('density')
        ax.plot(sub.density, sub.collapse_noise_percent, 'o-', color=colors[cond], label=cond)
    ax.set_xscale('log')
    ax.set_xlabel('Sudoku-to-hippocampus connection density (log scale)')
    ax.set_ylabel(f'Noise level (%) where accuracy first drops below {COLLAPSE_THRESHOLD:.0%}')
    ax.set_title(f'M={M}, K={K}: noise-robustness ceiling vs connection density')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_collapse_vs_density.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_collapse_vs_density.png')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--replot':
        df = pd.read_csv(f'{OUT_PREFIX}.csv')
    else:
        df = run()
    cap_df = collapse_thresholds(df)
    cap_df.to_csv(f'{OUT_PREFIX}_collapse_thresholds.csv', index=False)
    print('\nCollapse thresholds vs density:')
    print(cap_df.pivot(index='density', columns='condition', values='collapse_noise_percent').to_string())
    plot_per_density_curves(df)
    plot_collapse_vs_density(cap_df)
