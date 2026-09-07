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

"""
EXTENDED version: the original test (0.05-0.35) found the decay hadn't bottomed out,
so p_c=0.37 was extrapolated rather than bracketed by data. This pushes the upper end
to 0.55 to directly observe the far side of the transition, and replaces the naive
"critical point = peak iteration" heuristic with a proper free-parameter search for
p_c (best log-log R^2 on the decaying branch), matching the corrected post-hoc
analysis done for the original run.
"""

M = 500
K = 400
CLUE_FRACTIONS = np.round(np.arange(0.05, 0.56, 0.02), 3)
N_TRIALS = 200
MAX_ITERS = 300   # smoke test showed 93% hit the cap at MAX_ITERS=40 near low clue
                   # fractions -- need a much higher ceiling to see genuine convergence
                   # vs a real "never settles below threshold" regime

POOL_FILE = 'feedback_iterate_highM_pool.npy'
PROJ_SEED = SEED + 50
OUT_PREFIX = 'critical_slowing_down_extended'


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


def fit_power_law(sub_points, p_c_range, from_below):
    """Search p_c over p_c_range for the best log-log power-law fit of mean_iter
    vs |clue_fraction - p_c|. Returns (best_p_c, slope, se, r2) or None."""
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


def analyze_and_plot(df):
    summary = df.groupby('clue_fraction').agg(
        mean_iter=('iterations', 'mean'), std_iter=('iterations', 'std'),
        acc=('exact_acc', 'mean'),
    ).reset_index()

    # Identify the plateau: the longest run of low-clue-fraction points whose
    # mean_iter stays within 1 std of the group's own mean (a real flat region,
    # not just "before the interior peak").
    low = summary[summary.clue_fraction <= summary.clue_fraction.median()]
    plateau_mean = low.mean_iter.mean(); plateau_std = low.mean_iter.std()
    print(f'Plateau estimate (clue_fraction <= median): mean={plateau_mean:.1f}, std={plateau_std:.1f}')

    # Fit p_c as a free parameter on the DECAYING branch (approach from below,
    # i.e. as clue_fraction increases toward p_c) -- this is what matters for the
    # "does the decay bottom out" question the original run couldn't answer.
    decay_candidates = summary[summary.mean_iter < plateau_mean - plateau_std]
    fit_below = fit_power_law(decay_candidates, np.arange(
        decay_candidates.clue_fraction.max() + 0.01, decay_candidates.clue_fraction.max() + 0.30, 0.01),
        from_below=True)

    # Also check the far side, if the sweep now reaches low-iteration territory
    # past the transition (mean_iter approaching its floor from above).
    post_transition = summary[summary.clue_fraction > (fit_below[0] if fit_below else summary.clue_fraction.median())]
    fit_above = None
    if len(post_transition) >= 4:
        fit_above = fit_power_law(post_transition, np.arange(
            max(0.01, post_transition.clue_fraction.min() - 0.30), post_transition.clue_fraction.min() - 0.005, 0.01),
            from_below=False)

    if fit_below:
        p_c, slope, se, r2, intercept = fit_below
        print(f'Fit approaching p_c from below: p_c={p_c:.2f}, exponent={slope:.3f} +- {se:.3f}, R^2={r2:.4f}')
    if fit_above:
        p_c2, slope2, se2, r22, intercept2 = fit_above
        print(f'Fit approaching p_c from above: p_c={p_c2:.2f}, exponent={slope2:.3f} +- {se2:.3f}, R^2={r22:.4f}')

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    p_c_display = fit_below[0] if fit_below else summary.clue_fraction.median()
    ax = axes[0]
    ax.errorbar(summary.clue_fraction, summary.mean_iter, yerr=summary.std_iter,
                fmt='o-', color='tab:purple', capsize=3)
    ax.axvline(p_c_display, color='gray', linestyle=':', label=f'fitted p_c={p_c_display:.2f}')
    ax.set_xlabel('Clue fraction'); ax.set_ylabel('Mean iterations to converge')
    ax.set_title(f'M={M}, K={K}: iteration count vs clue fraction')
    ax.grid(alpha=0.3); ax.legend()

    ax = axes[1]
    ax.plot(summary.clue_fraction, summary.acc, 'o-', color='tab:green')
    ax.axvline(p_c_display, color='gray', linestyle=':')
    ax.set_xlabel('Clue fraction'); ax.set_ylabel('Exact recovery accuracy')
    ax.set_title('Accuracy vs clue fraction (same x-axis)')
    ax.grid(alpha=0.3)

    ax = axes[2]
    if fit_below:
        p_c, slope, se, r2, intercept = fit_below
        dist = p_c - decay_candidates.clue_fraction
        valid = dist > 0
        ax.plot(dist[valid], decay_candidates.mean_iter[valid], 'o', color='tab:red',
                label=f'below p_c: exp={-slope:.2f}+-{se:.2f}, R2={r2:.2f}')
        xs = np.linspace(dist[valid].min(), dist[valid].max(), 20)
        ax.plot(xs, 10**intercept * xs**slope, '--', color='tab:red', alpha=0.5)
    if fit_above:
        p_c2, slope2, se2, r22, intercept2 = fit_above
        dist2 = post_transition.clue_fraction - p_c2
        valid2 = dist2 > 0
        ax.plot(dist2[valid2], post_transition.mean_iter[valid2], 's', color='tab:blue',
                label=f'above p_c: exp={-slope2:.2f}+-{se2:.2f}, R2={r22:.2f}')
        xs2 = np.linspace(dist2[valid2].min(), dist2[valid2].max(), 20)
        ax.plot(xs2, 10**intercept2 * xs2**slope2, '--', color='tab:blue', alpha=0.5)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('|clue_fraction - p_c| (log scale)'); ax.set_ylabel('Mean iterations (log scale)')
    ax.set_title('Power-law check, both sides of the transition')
    ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')
    return fit_below, fit_above, plateau_mean, plateau_std


if __name__ == '__main__':
    df = run()
    analyze_and_plot(df)
