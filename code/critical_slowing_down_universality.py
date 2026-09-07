"""
Universality check for the critical-slowing-down finding (critical_slowing_down_test.py
found p_c~0.37, exponent nu~0.96, R^2=0.974 at M=500, K=400).

A critical exponent is only meaningful as "universal" if it recurs across different
microscopic parameters. Tests M=200 and M=2000 (holding K=400 fixed) and K=150 and
K=200 (holding M=500 fixed), using a coarser clue-fraction grid and fewer trials than
the flagship run (this is a confirmatory sweep, not the precision result), plus
reports each config's plateau timescale to see how that scales with K, M.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy import stats

from hippocampus_drive_readout import (
    H, N, SEED, SEED_TRIALS, TAU as TAU_BASELINE, K as K_BASELINE,
    relation, make_projection, drive_vectors, decode_digits, conflict_matrix, coincidence,
)

CONFIGS = [(200, 400), (2000, 400), (500, 150), (500, 200)]
CLUE_FRACTIONS = np.round(np.arange(0.05, 0.56, 0.05), 3)
N_TRIALS = 100
MAX_ITERS = 300

POOL_FILE = 'feedback_iterate_highM_pool.npy'
PROJ_SEED = SEED + 50
OUT_PREFIX = 'critical_slowing_down_universality'


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


def fit_power_law(sub_points, p_c_range, from_below):
    best = None
    for p_c in p_c_range:
        if from_below:
            dist = p_c - sub_points.clue_fraction
        else:
            dist = sub_points.clue_fraction - p_c
        valid = dist > 0
        if valid.sum() < 4:
            continue
        logx = np.log10(dist[valid]); logy = np.log10(sub_points.mean_iter[valid])
        slope, intercept, r, p, se = stats.linregress(logx, logy)
        if best is None or r**2 > best[3]:
            best = (p_c, slope, se, r**2, intercept)
    return best


def run_one(M, K, pool_full, conf, trng):
    pool = pool_full[:M]
    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    digits_true = pool.reshape(M, N)
    B_drive = drive_vectors(pool)
    H_keys = make_engrams(C_all, W_SH, K)

    rows = []
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        n_trials = min(N_TRIALS, M)
        targets = trng.choice(M, size=n_trials, replace=False)
        for t in targets:
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
            C_partial = partial_relation(pool[t], known_mask)
            a = np.asarray(W_SH @ C_partial.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx] = 1

            h_final, digits_final, iters, conv = iterate_attractor(H_keys, B_drive, W_SH, conf, h_q, K)
            rows.append({'M': M, 'K': K, 'clue_fraction': clue_frac, 'iterations': iters,
                         'exact_acc': int(np.all(digits_final == digits_true[t]))})
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        print(f'  M={M} K={K}  clue_frac={clue_frac:.2f}  mean_iter={np.mean([r["iterations"] for r in sub]):6.2f}  '
              f'acc={np.mean([r["exact_acc"] for r in sub]):.2%}')
    return pd.DataFrame(rows)


def run():
    pool_full = np.load(POOL_FILE)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 70)

    all_dfs = []
    fit_rows = []
    for M, K in CONFIGS:
        print(f'--- M={M}, K={K} ---')
        df = run_one(M, K, pool_full, conf, trng)
        all_dfs.append(df)

        summary = df.groupby('clue_fraction').agg(mean_iter=('iterations', 'mean')).reset_index()
        low = summary[summary.clue_fraction <= summary.clue_fraction.median()]
        plateau_mean = low.mean_iter.mean(); plateau_std = low.mean_iter.std()

        decay_candidates = summary[summary.mean_iter < plateau_mean - (plateau_std if plateau_std > 0 else 1)]
        fit = None
        if len(decay_candidates) >= 4:
            fit = fit_power_law(decay_candidates, np.arange(
                decay_candidates.clue_fraction.max() + 0.01, decay_candidates.clue_fraction.max() + 0.30, 0.01),
                from_below=True)

        if fit:
            p_c, slope, se, r2, intercept = fit
            print(f'  FIT: p_c={p_c:.2f}, exponent={-slope:.3f} +- {se:.3f}, R^2={r2:.4f}, '
                  f'plateau={plateau_mean:.1f}+-{plateau_std:.1f}')
            fit_rows.append({'M': M, 'K': K, 'p_c': p_c, 'exponent': slope, 'exponent_se': se,
                             'r2': r2, 'plateau_mean': plateau_mean, 'plateau_std': plateau_std})
        else:
            print(f'  FIT: insufficient decay data to fit. plateau={plateau_mean:.1f}+-{plateau_std:.1f}')
            fit_rows.append({'M': M, 'K': K, 'p_c': np.nan, 'exponent': np.nan, 'exponent_se': np.nan,
                             'r2': np.nan, 'plateau_mean': plateau_mean, 'plateau_std': plateau_std})

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(f'{OUT_PREFIX}_fits.csv', index=False)
    print('\n=== Universality summary ===')
    print(fit_df.to_string(index=False))
    return full_df, fit_df


def plot(full_df, fit_df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(CONFIGS)))

    ax = axes[0]
    for (M, K), c in zip(CONFIGS, colors):
        sub = full_df[(full_df.M == M) & (full_df.K == K)].groupby('clue_fraction').agg(
            mean_iter=('iterations', 'mean')).reset_index()
        ax.plot(sub.clue_fraction, sub.mean_iter, 'o-', color=c, label=f'M={M}, K={K}')
    ax.set_xlabel('Clue fraction'); ax.set_ylabel('Mean iterations')
    ax.set_title('Relaxation time across configurations')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    valid = fit_df.dropna(subset=['exponent'])
    ax.errorbar(range(len(valid)), valid.exponent, yerr=valid.exponent_se, fmt='o', capsize=4, color='tab:red')
    ax.axhline(0.963, color='gray', linestyle=':', label='flagship (M=500,K=400): 0.96')
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([f'M={r.M}\nK={r.K}' for _, r in valid.iterrows()], fontsize=8)
    ax.set_ylabel('Fitted exponent nu')
    ax.set_title('Exponent universality check')
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')


if __name__ == '__main__':
    full_df, fit_df = run()
    plot(full_df, fit_df)
