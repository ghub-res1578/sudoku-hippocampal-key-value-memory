"""
Sweep M (stored Sudokus) and K (engram sparsity, with TAU rescaled proportionally
per the K-sweep finding: TAU = 2.0 * K/150) for the partial-grid completion task.

Extends sudoku_partial_grid_MK_sweep.py with per-trial iteration tracking (not
saved there), to answer: across the full (M, K, clue_fraction) grid, does the
single-baseline finding hold everywhere -- that mean iterations PEAKS where
accuracy is LOWEST (the ambiguous, under-constrained regime), and DROPS to just a
few iterations once accuracy is high? Reports, per (M, K): (a) the clue fraction
where mean iterations is maximized and the accuracy there, and (b) the clue
fraction where accuracy first reaches >=90% and the mean iterations there.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax

from hippocampus_drive_readout import (
    H, N, SEED, SEED_TRIALS, TAU as TAU_BASELINE, K as K_BASELINE,
    relation, make_projection, drive_vectors, decode_digits, conflict_matrix, coincidence,
)

M_VALUES = [200, 500, 1000, 2000, 5000]
K_VALUES = [50, 100, 150, 200, 300, 400]
CLUE_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0]
N_TRIALS = 50               # bumped up from 30 for more reliable iteration statistics
MAX_ITERS = 20
RECOVERY_THRESHOLD = 0.90

POOL_FILE = 'feedback_iterate_highM_pool.npy'   # cached, up to 10000
PROJ_SEED = SEED + 50
OUT_PREFIX = 'sudoku_partial_grid_MK_iterations_sweep'


def make_engrams(C, W_SH, k):
    a = C @ W_SH.T
    a = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C.shape[0], H), np.uint8)
    Hs[np.arange(C.shape[0])[:, None], idx] = 1
    return Hs


def clean_engram(C_masked, W_SH, k):
    a = np.asarray(W_SH @ C_masked.astype(np.float32).ravel()).ravel()
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


def readout_attention_k(H_keys, B_values, h_query, k):
    tau = TAU_BASELINE * (k / K_BASELINE)
    sim = H_keys.astype(np.float32) @ h_query.astype(np.float32)
    p = softmax(sim / tau)
    return p @ B_values


def partial_relation(g, known_mask):
    x = g.ravel(); C = (x[:, None] == x[None, :])
    known2d = known_mask[:, None] & known_mask[None, :]
    C = C & known2d
    np.fill_diagonal(C, False)
    return C


def iterate_attractor(H_keys, B_drive, W_SH, conf, hq, k, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    for it in range(1, max_iters + 1):
        b = readout_attention_k(H_keys, B_drive, h_current, k)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH, k)
        key = h_next.tobytes()
        if key in visited:
            return h_next, digits, it, True
        visited[key] = it
        h_current = h_next
    return h_current, digits, max_iters, False


def run():
    pool_full = np.load(POOL_FILE)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 30)

    rows = []
    for M in M_VALUES:
        pool = pool_full[:M]
        rng_proj = np.random.default_rng(PROJ_SEED)
        W_SH = make_projection(rng_proj)
        C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
        digits_true = pool.reshape(M, N)
        B_drive = drive_vectors(pool)

        for k in K_VALUES:
            H_keys = make_engrams(C_all, W_SH, k)
            n_trials = min(N_TRIALS, M)
            for clue_frac in CLUE_FRACTIONS:
                n_known = max(1, int(round(clue_frac * N)))
                targets = trng.choice(M, size=n_trials, replace=False)
                exact_finals = []; iters_list = []
                for t in targets:
                    known_idx = trng.choice(N, size=n_known, replace=False)
                    known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
                    C_partial = partial_relation(pool[t], known_mask)
                    a = np.asarray(W_SH @ C_partial.astype(np.float32).ravel()).ravel()
                    idx = np.argpartition(a, -k)[-k:]
                    h_q = np.zeros(H, np.uint8); h_q[idx] = 1

                    h_final, digits_final, iters, conv = iterate_attractor(H_keys, B_drive, W_SH, conf, h_q, k)
                    exact_finals.append(int(np.all(digits_final == digits_true[t])))
                    iters_list.append(iters)
                acc = np.mean(exact_finals)
                mean_iters = np.mean(iters_list)
                rows.append({'M': M, 'K': k, 'clue_fraction': clue_frac,
                             'exact_acc': acc, 'mean_iterations': mean_iters})
            sub = [r for r in rows if r['M'] == M and r['K'] == k]
            print(f'M={M:5d} K={k:3d}  ' + '  '.join(
                f'{r["clue_fraction"]:.2f}:acc={r["exact_acc"]:.2f}/it={r["mean_iterations"]:.1f}' for r in sub))

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def analysis_table(df, threshold=RECOVERY_THRESHOLD):
    rows = []
    for M in M_VALUES:
        for k in K_VALUES:
            sub = df[(df.M == M) & (df.K == k)].sort_values('clue_fraction')

            # (a) where accuracy first reaches the recovery threshold, and iterations there
            above = sub[sub.exact_acc >= threshold]
            clue_at_90 = above.clue_fraction.min() if len(above) else np.nan
            iters_at_90 = (above.sort_values('clue_fraction').iloc[0].mean_iterations
                           if len(above) else np.nan)

            # (b) where mean iterations PEAKS (the hardest / most ambiguous regime), and accuracy there
            peak_row = sub.loc[sub.mean_iterations.idxmax()]

            rows.append({
                'M': M, 'K': k,
                'clue_fraction_for_90pct': clue_at_90, 'iterations_at_90pct': iters_at_90,
                'clue_fraction_at_peak_iters': peak_row.clue_fraction,
                'peak_iterations': peak_row.mean_iterations, 'accuracy_at_peak_iters': peak_row.exact_acc,
            })
    return pd.DataFrame(rows)


def plot(df, analysis_df):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    pivot1 = analysis_df.pivot(index='K', columns='M', values='clue_fraction_for_90pct')
    ax = axes[0]
    im = ax.imshow(pivot1.values, aspect='auto', cmap='viridis_r', origin='lower')
    ax.set_xticks(range(len(pivot1.columns))); ax.set_xticklabels(pivot1.columns)
    ax.set_yticks(range(len(pivot1.index))); ax.set_yticklabels(pivot1.index)
    ax.set_xlabel('M'); ax.set_ylabel('K')
    ax.set_title(f'Clue fraction for >={RECOVERY_THRESHOLD:.0%} accuracy')
    for i in range(len(pivot1.index)):
        for j in range(len(pivot1.columns)):
            v = pivot1.values[i, j]
            ax.text(j, i, f'{v:.2f}' if not np.isnan(v) else 'n/a', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    pivot2 = analysis_df.pivot(index='K', columns='M', values='iterations_at_90pct')
    ax = axes[1]
    im = ax.imshow(pivot2.values, aspect='auto', cmap='plasma', origin='lower')
    ax.set_xticks(range(len(pivot2.columns))); ax.set_xticklabels(pivot2.columns)
    ax.set_yticks(range(len(pivot2.index))); ax.set_yticklabels(pivot2.index)
    ax.set_xlabel('M'); ax.set_ylabel('K')
    ax.set_title('Mean iterations AT the 90%-accuracy point')
    for i in range(len(pivot2.index)):
        for j in range(len(pivot2.columns)):
            v = pivot2.values[i, j]
            ax.text(j, i, f'{v:.1f}' if not np.isnan(v) else 'n/a', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    pivot3 = analysis_df.pivot(index='K', columns='M', values='peak_iterations')
    ax = axes[2]
    im = ax.imshow(pivot3.values, aspect='auto', cmap='plasma', origin='lower')
    ax.set_xticks(range(len(pivot3.columns))); ax.set_xticklabels(pivot3.columns)
    ax.set_yticks(range(len(pivot3.index))); ax.set_yticklabels(pivot3.index)
    ax.set_xlabel('M'); ax.set_ylabel('K')
    ax.set_title('PEAK mean iterations (the struggle point)')
    for i in range(len(pivot3.index)):
        for j in range(len(pivot3.columns)):
            v = pivot3.values[i, j]
            ax.text(j, i, f'{v:.1f}' if not np.isnan(v) else 'n/a', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_heatmaps.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_heatmaps.png')


if __name__ == '__main__':
    df = run()
    analysis_df = analysis_table(df)
    analysis_df.to_csv(f'{OUT_PREFIX}_analysis.csv', index=False)
    print('\nPer (M,K): clue fraction & iterations at 90% accuracy, vs. clue fraction & '
          'accuracy at PEAK iterations (the struggle point):')
    print(analysis_df.to_string(index=False))
    plot(df, analysis_df)
