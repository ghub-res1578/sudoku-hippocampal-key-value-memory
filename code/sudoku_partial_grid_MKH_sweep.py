"""
Full 3D sweep over M (stored Sudokus), K (engram sparsity, as a density K/H so it's
comparable across different H), and H (hippocampal population size), for the
partial-grid completion task. Reports the clue fraction needed for >=90% exact
recovery at each (M, K, H) triple, visualized as a 3D scatter (color = clue
fraction needed, lower = better).

hippocampus_drive_readout.py hardcodes H=1000 inside make_projection/make_engrams/
clean_engram, so this sweep needs self-contained versions parameterized by H
(same pattern used in feedback_vs_density_sweep.py and mnist_crossmodal_hash_H2000_K400.py
earlier in this project). TAU is rescaled per K (TAU = 2.0 * K/150), the same rule
validated throughout -- self-overlap is always K regardless of H, which is what TAU
needs to track.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy import sparse
from scipy.special import softmax

from hippocampus_drive_readout import (
    N, SEED, SEED_TRIALS, RIDGE_LAMBDA,
    relation, drive_vectors, decode_digits, conflict_matrix, coincidence,
)

INPUT_DIM = N * N
DENSITY = 0.02          # W_SH connection density (shown earlier not to matter)
TAU_BASELINE = 2.0
K_BASELINE = 150

H_VALUES = [500, 1000, 2000]
K_DENSITIES = [0.05, 0.10, 0.15, 0.20, 0.30]
M_VALUES = [200, 1000, 5000]
CLUE_FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.0]
N_TRIALS = 20
MAX_ITERS = 20
RECOVERY_THRESHOLD = 0.90

POOL_FILE = 'feedback_iterate_highM_pool.npy'
PROJ_SEED = SEED + 60
OUT_PREFIX = 'sudoku_partial_grid_MKH_sweep'


def make_projection(rng, H):
    mask = rng.random((H, INPUT_DIM)) < DENSITY
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H, INPUT_DIM), dtype=np.float32)


def make_engrams(C, W_SH, H, k):
    a = C @ W_SH.T
    a = a.toarray() if sparse.issparse(a) else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C.shape[0], H), np.uint8)
    Hs[np.arange(C.shape[0])[:, None], idx] = 1
    return Hs


def clean_engram(C_masked, W_SH, H, k):
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


def iterate_attractor(H_keys, B_drive, W_SH, H, k, conf, hq, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    for it in range(1, max_iters + 1):
        b = readout_attention_k(H_keys, B_drive, h_current, k)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH, H, k)
        key = h_next.tobytes()
        if key in visited:
            return h_next, digits, it, True
        visited[key] = it
        h_current = h_next
    return h_current, digits, max_iters, False


def run():
    pool_full = np.load(POOL_FILE)
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 40)

    rows = []
    for H in H_VALUES:
        rng_proj = np.random.default_rng(PROJ_SEED)
        W_SH = make_projection(rng_proj, H)

        for M in M_VALUES:
            pool = pool_full[:M]
            C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
            digits_true = pool.reshape(M, N)
            B_drive = drive_vectors(pool)
            n_trials = min(N_TRIALS, M)

            for density in K_DENSITIES:
                k = max(1, int(round(density * H)))
                H_keys = make_engrams(C_all, W_SH, H, k)

                for clue_frac in CLUE_FRACTIONS:
                    n_known = max(1, int(round(clue_frac * N)))
                    targets = trng.choice(M, size=n_trials, replace=False)
                    exact_finals = []
                    for t in targets:
                        known_idx = trng.choice(N, size=n_known, replace=False)
                        known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
                        C_partial = partial_relation(pool[t], known_mask)
                        a = np.asarray(W_SH @ C_partial.astype(np.float32).ravel()).ravel()
                        idx = np.argpartition(a, -k)[-k:]
                        h_q = np.zeros(H, np.uint8); h_q[idx] = 1

                        h_final, digits_final, iters, conv = iterate_attractor(
                            H_keys, B_drive, W_SH, H, k, conf, h_q)
                        exact_finals.append(int(np.all(digits_final == digits_true[t])))
                    acc = np.mean(exact_finals)
                    rows.append({'H': H, 'M': M, 'K': k, 'density': density,
                                 'clue_fraction': clue_frac, 'exact_acc': acc})
                sub = [r for r in rows if r['H'] == H and r['M'] == M and r['density'] == density]
                print(f'H={H:5d} M={M:5d} K={k:4d} (density={density:.2f})  ' +
                      '  '.join(f'{r["clue_fraction"]:.2f}:{r["exact_acc"]:.2f}' for r in sub))

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def recovery_threshold_table(df, threshold=RECOVERY_THRESHOLD):
    rows = []
    for H in H_VALUES:
        for M in M_VALUES:
            for density in K_DENSITIES:
                sub = df[(df.H == H) & (df.M == M) & (df.density == density)].sort_values('clue_fraction')
                above = sub[sub.exact_acc >= threshold]
                req = above.clue_fraction.min() if len(above) else np.nan
                k_val = sub.K.iloc[0] if len(sub) else np.nan
                rows.append({'H': H, 'M': M, 'K': k_val, 'density': density,
                             'clue_fraction_for_90pct': req})
    return pd.DataFrame(rows)


def plot_3d(thresh_df):
    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection='3d')
    valid = thresh_df.dropna(subset=['clue_fraction_for_90pct'])
    sc = ax.scatter(valid.M, valid.density, valid.H, c=valid.clue_fraction_for_90pct,
                     s=250, cmap='viridis_r', vmin=valid.clue_fraction_for_90pct.min(),
                     vmax=valid.clue_fraction_for_90pct.max(), edgecolor='k', linewidth=0.5)
    ax.set_xlabel('M (stored Sudokus)')
    ax.set_ylabel('K density (K/H)')
    ax.set_zlabel('H (hippocampal neurons)')
    ax.set_title('Clue fraction needed for >=90% exact recovery\n(color: lower/yellow = more efficient recovery)')
    fig.colorbar(sc, ax=ax, shrink=0.6, label='clue fraction required')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_3d.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_3d.png')


if __name__ == '__main__':
    df = run()
    thresh_df = recovery_threshold_table(df)
    thresh_df.to_csv(f'{OUT_PREFIX}_recovery_thresholds.csv', index=False)
    print('\nClue fraction needed for >=90% exact recovery, by (H, M, density):')
    print(thresh_df.to_string(index=False))
    best = thresh_df.dropna().sort_values('clue_fraction_for_90pct').iloc[0]
    print(f"\nBest regime: H={best.H:.0f}, M={best.M:.0f}, K={best.K:.0f} (density={best.density:.2f}), "
          f"needs only {best.clue_fraction_for_90pct:.0%} of cells known.")
    plot_3d(thresh_df)
