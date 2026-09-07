"""
Hold M=2000 fixed, sweep the engram sparsity K (active units out of H=1000), and
compare accuracy before (single readout pass) vs after (attractor dynamics iterated
to convergence) at each K.

Sparser engrams (small K) should give better baseline pattern separation (lower
chance overlap between unrelated stored keys, ~K^2/H) but less absolute redundancy
to survive corruption. Denser engrams (large K) are the reverse. This sweep checks
where the conflict-mask feedback loop's rescue zone actually sits as a function of K,
holding M and the noise model (fraction of H bits flipped) fixed.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hippocampus_drive_readout import (
    H, N, SEED, SEED_TRIALS, TAU as TAU_BASELINE, K as K_BASELINE,
    relation, make_projection, drive_vectors,
    decode_digits, conflict_matrix, coincidence, clean_engram,
)
from scipy.special import softmax

M = 2000
K_VALUES = [10, 25, 50, 75, 100, 150, 200, 300, 400, 500]

# TAU=2.0 was implicitly calibrated for the K=150 baseline: raw key-similarity values
# scale with K (self-overlap = K, competitor overlap ~ K^2/H), so a FIXED tau under-
# sharpens softmax at small K -- e.g. at K=10, tau=2 gives only 62% confidence on an
# unambiguous, noise-free self-match (margin 8 over the best competitor, the true
# maximum possible). Scaling tau with K keeps softmax sharpness comparable across the
# sweep, isolating the sparsity effect instead of conflating it with softmax miscalibration.
def tau_for_k(k):
    return TAU_BASELINE * (k / K_BASELINE)


def readout_attention_k(H_keys, B_values, h_query, k):
    sim = H_keys.astype(np.float32) @ h_query.astype(np.float32)
    p = softmax(sim / tau_for_k(k))
    return p @ B_values
NOISE_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]
N_TRIALS = 50
MAX_ITERS = 20
COLLAPSE_THRESHOLD = 0.50

OUT_PREFIX = 'feedback_vs_K_sweep'
POOL_FILE = 'feedback_iterate_M2000_pool.npy'   # reuse the existing cached M=2000 pool
PROJ_SEED = SEED + 3   # independent encoder stream for this experiment


def engrams_for_k(a, k):
    """a: (M, H) precomputed activations C_all @ W_SH.T -- top-k WTA, computed once
    and reused across every K value (avoids recomputing the sparse matmul per K)."""
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((a.shape[0], H), np.uint8)
    Hs[np.arange(a.shape[0])[:, None], idx] = 1
    return Hs


def iterate_feedback(H_keys, B_values, W_SH, conf, hq, true_digits, k, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    digits = decode_digits(readout_attention_k(H_keys, B_values, h_current, k))
    initial_acc = np.mean(digits == true_digits)
    for it in range(1, max_iters + 1):
        b = readout_attention_k(H_keys, B_values, h_current, k)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH, k=k)
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
    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    a = C_all @ W_SH.T
    a = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)   # (M, H), computed once
    B_values = drive_vectors(pool)
    digits_true = pool.reshape(M, N)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 5)

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
                res = iterate_feedback(H_keys, B_values, W_SH, conf, hq, digits_true[t], k)
                rows.append({'K': k, 'noise_percent': noise * 100, **res})
            sub = [r for r in rows if r['K'] == k and r['noise_percent'] == noise * 100]
            print(f'  K={k:4d}  noise={noise:5.0%}  mean_iter={np.mean([r["iterations"] for r in sub]):5.2f}  '
                  f'initial={np.mean([r["initial_acc"] for r in sub]):7.4%}  '
                  f'final={np.mean([r["final_acc"] for r in sub]):7.4%}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def collapse_thresholds(df, threshold=COLLAPSE_THRESHOLD):
    rows = []
    for k in sorted(df.K.unique()):
        sub = df[df.K == k].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        for col in ['initial_acc', 'final_acc']:
            below = sub[sub[col] < threshold]
            thresh = below.noise_percent.min() if len(below) else np.nan
            rows.append({'K': k, 'condition': col, 'collapse_noise_percent': thresh})
    return pd.DataFrame(rows)


def plot_per_K_curves(df):
    Ks = sorted(df.K.unique())
    ncols = 5; nrows = int(np.ceil(len(Ks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.5 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, k in zip(axes, Ks):
        sub = df[df.K == k].groupby('noise_percent').agg(
            initial_acc=('initial_acc', 'mean'), final_acc=('final_acc', 'mean')).reset_index()
        ax.plot(sub.noise_percent, sub.initial_acc, 'o-', color='tab:orange', label='initial', markersize=4)
        ax.plot(sub.noise_percent, sub.final_acc, 'o-', color='tab:green', label='converged', markersize=4)
        ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1)
        ax.set_title(f'K={k}')
        ax.grid(alpha=0.3)
    for ax in axes[len(Ks):]:
        ax.axis('off')
    axes[0].legend(fontsize=8)
    fig.supxlabel('Query noise (% of H bits flipped)')
    fig.supylabel('Digit-cell retrieval accuracy')
    fig.suptitle(f'M={M}: initial vs converged accuracy across engram sparsity K')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_per_K_curves.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_per_K_curves.png')


def plot_collapse_vs_K(cap_df):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'initial_acc': 'tab:orange', 'final_acc': 'tab:green'}
    for cond in ['initial_acc', 'final_acc']:
        sub = cap_df[cap_df.condition == cond].sort_values('K')
        ax.plot(sub.K, sub.collapse_noise_percent, 'o-', color=colors[cond], label=cond)
    ax.set_xlabel('Engram sparsity K (active units out of H=1000)')
    ax.set_ylabel(f'Noise level (%) where accuracy first drops below {COLLAPSE_THRESHOLD:.0%}')
    ax.set_title(f'M={M}: noise-robustness ceiling vs engram sparsity')
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_collapse_vs_K.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_collapse_vs_K.png')


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--replot':
        df = pd.read_csv(f'{OUT_PREFIX}.csv')
    else:
        df = run()
    cap_df = collapse_thresholds(df)
    cap_df.to_csv(f'{OUT_PREFIX}_collapse_thresholds.csv', index=False)
    print('\nCollapse thresholds vs K:')
    print(cap_df.pivot(index='K', columns='condition', values='collapse_noise_percent').to_string())
    plot_per_K_curves(df)
    plot_collapse_vs_K(cap_df)
