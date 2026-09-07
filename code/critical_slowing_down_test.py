"""
Tests whether the attractor loop's iteration count shows CRITICAL SLOWING DOWN near
the recovery phase transition -- the same phenomenon studied in statistical physics
of constraint satisfaction (k-SAT solver iteration counts diverging at the
SAT-UNSAT threshold; Mezard/Zecchina-style "physics of computation" literature).

If mean iterations scales approximately as a power law in |clue_fraction - critical|
on both sides of the transition, that's a genuine, quantitative, citable dynamical-
systems signature -- not just a relabeled accuracy curve.

Uses the sharpest transition found in the M-K-iterations sweep: M=500, K=400
(peak_iterations=19.4 at clue_fraction~0.15), with a much finer clue-fraction grid
and more trials for reliable statistics right around the transition.
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

M = 500
K = 400
CLUE_FRACTIONS = np.round(np.arange(0.05, 0.36, 0.02), 3)
N_TRIALS = 200
MAX_ITERS = 300   # smoke test showed 93% hit the cap at MAX_ITERS=40 near low clue
                   # fractions -- need a much higher ceiling to see genuine convergence
                   # vs a real "never settles below threshold" regime

POOL_FILE = 'feedback_iterate_highM_pool.npy'
PROJ_SEED = SEED + 50
OUT_PREFIX = 'critical_slowing_down_test'


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
    pool = np.load(POOL_FILE)[:M]
    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    digits_true = pool.reshape(M, N)
    B_drive = drive_vectors(pool)
    H_keys = make_engrams(C_all, W_SH, K)
    conf = conflict_matrix()

    trng = np.random.default_rng(SEED_TRIALS + 60)
    rows = []
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        targets = trng.choice(M, size=min(N_TRIALS, M), replace=False)
        for t in targets:
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
            C_partial = partial_relation(pool[t], known_mask)
            a = np.asarray(W_SH @ C_partial.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx] = 1

            h_final, digits_final, iters, conv = iterate_attractor(H_keys, B_drive, W_SH, conf, h_q, K)
            rows.append({'clue_fraction': clue_frac, 'target': int(t), 'iterations': iters,
                         'converged': conv, 'exact_acc': int(np.all(digits_final == digits_true[t]))})
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        print(f'clue_frac={clue_frac:.2f}  mean_iter={np.mean([r["iterations"] for r in sub]):6.2f}  '
              f'max_iter={np.max([r["iterations"] for r in sub]):3d}  '
              f'acc={np.mean([r["exact_acc"] for r in sub]):.2%}  '
              f'pct_hit_cap={np.mean([r["iterations"] == MAX_ITERS for r in sub]):.1%}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def analyze_and_plot(df):
    summary = df.groupby('clue_fraction').agg(
        mean_iter=('iterations', 'mean'), std_iter=('iterations', 'std'),
        acc=('exact_acc', 'mean'),
    ).reset_index()

    peak_idx = summary.mean_iter.idxmax()
    critical_frac = summary.clue_fraction.iloc[peak_idx]
    print(f'\nEmpirical critical point (peak mean_iter): clue_fraction={critical_frac:.2f}, '
          f'mean_iter={summary.mean_iter.iloc[peak_idx]:.2f}')

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    ax = axes[0]
    ax.errorbar(summary.clue_fraction, summary.mean_iter, yerr=summary.std_iter,
                fmt='o-', color='tab:purple', capsize=3)
    ax.axvline(critical_frac, color='gray', linestyle=':', label=f'peak at {critical_frac:.2f}')
    ax.set_xlabel('Clue fraction'); ax.set_ylabel('Mean iterations to converge')
    ax.set_title(f'M={M}, K={K}: iteration count vs clue fraction')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(summary.clue_fraction, summary.acc, 'o-', color='tab:green')
    ax.axvline(critical_frac, color='gray', linestyle=':')
    ax.set_xlabel('Clue fraction'); ax.set_ylabel('Exact recovery accuracy')
    ax.set_title('Accuracy vs clue fraction (same x-axis)')
    ax.grid(alpha=0.3)

    # log-log: iterations vs |distance from critical point|, both sides separately
    ax = axes[2]
    below = summary[summary.clue_fraction < critical_frac].copy()
    above = summary[summary.clue_fraction > critical_frac].copy()
    below['dist'] = critical_frac - below.clue_fraction
    above['dist'] = above.clue_fraction - critical_frac
    fits = {}
    for label, sub, color in [('below critical', below, 'tab:blue'), ('above critical', above, 'tab:red')]:
        valid = sub[(sub.dist > 0) & (sub.mean_iter > 0)]
        if len(valid) >= 3:
            logx = np.log10(valid.dist); logy = np.log10(valid.mean_iter)
            slope, intercept, r, p, se = stats.linregress(logx, logy)
            fits[label] = (slope, r**2)
            ax.plot(valid.dist, valid.mean_iter, 'o', color=color,
                    label=f'{label}: slope={slope:.2f}, R2={r**2:.2f}')
            xs = np.linspace(valid.dist.min(), valid.dist.max(), 20)
            ax.plot(xs, 10**intercept * xs**slope, '--', color=color, alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('|clue_fraction - critical| (log scale)'); ax.set_ylabel('Mean iterations (log scale)')
    ax.set_title('Power-law check: iterations vs distance from transition')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')
    for label, (slope, r2) in fits.items():
        print(f'  {label}: power-law exponent = {slope:.3f}, R^2 = {r2:.3f}')
    return fits


if __name__ == '__main__':
    df = run()
    analyze_and_plot(df)
