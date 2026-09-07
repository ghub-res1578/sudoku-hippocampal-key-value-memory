"""
Memory-capacity comparison for the Sudoku hippocampal key-value model.

Two readout mechanisms are compared on IDENTICAL keys (same random
projection W, same k-WTA engrams) as the number of stored memories M
is swept:

  regression -- W_HB trained by ridge regression, H_keys @ W_HB.T ~= B_values
                (matches complete_sudoku_hippocampus_analysis.py)
  attention  -- softmax(H_keys @ h_query / TAU) @ B_values, no learned weights
                (matches sudoku_conflict_mask_softmax_K100.py)

Capacity is operationalized as the largest M in the sweep for which mean
digit-cell retrieval accuracy stays >= CAPACITY_THRESHOLD.

Noise convention matches sudoku_conflict_mask_softmax_K100.py: nflip is a
fraction of the full H-unit population (not of the K active units), applied
as a random XOR flip to the stored engram before retrieval.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy import sparse

SEED_MEMORY = 42
SEED_TRIALS = 12345

H = 1000                 # hippocampal neurons
K = 150                  # engram size (active units per key)
N = 81
INPUT_DIM = N * N        # 6561
DENSITY = 0.02           # W_SH sparsity
TAU = 2.0                # softmax temperature
RIDGE_LAMBDA = 1e-4
N_TRIALS = 100
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.0]
MEMORY_LOADS = [10, 20, 50, 100, 200, 300, 500, 700, 1000, 1300, 1600, 2000]
M_MAX = max(MEMORY_LOADS)
CAPACITY_THRESHOLD = 0.95

DRIVE_LEVELS = np.array([.58, .59, .60, .61, .62, .63, .64, .65, .67])

OUT_PREFIX = 'capacity_regression_vs_attention'


# ---------------- Sudoku generation (self-contained, matches K100 file) ----------------

def generate_sudoku(rng):
    g = np.zeros((9, 9), dtype=np.int8)

    def cand(pos):
        r, c = divmod(pos, 9)
        used = set(g[r, :]); used.update(g[:, c])
        br, bc = (r // 3) * 3, (c // 3) * 3
        used.update(g[br:br + 3, bc:bc + 3].ravel())
        return np.array([d for d in range(1, 10) if d not in used], dtype=np.int8)

    def solve():
        best = -1; bc = None; bl = 10
        for pos in range(81):
            if g.ravel()[pos] != 0:
                continue
            cs = cand(pos)
            if len(cs) == 0:
                return False
            if len(cs) < bl:
                best, bc, bl = pos, cs, len(cs)
        if best < 0:
            return True
        rng.shuffle(bc); r, c = divmod(best, 9)
        for d in bc:
            g[r, c] = d
            if solve():
                return True
            g[r, c] = 0
        return False

    solve(); return g.copy()


def generate_unique(M, rng):
    out = []; seen = set()
    while len(out) < M:
        g = generate_sudoku(rng); k = tuple(g.ravel())
        if k not in seen:
            seen.add(k); out.append(g)
    return np.asarray(out)


# ---------------- Relational structures ----------------

def relation(g):
    x = g.ravel(); C = (x[:, None] == x[None, :])
    np.fill_diagonal(C, False); return C


def conflict_matrix():
    c = np.zeros((81, 81), bool)
    for i in range(81):
        r, col = divmod(i, 9)
        for j in range(81):
            rr, cc = divmod(j, 9)
            c[i, j] = (r == rr or col == cc or (r // 3 == rr // 3 and col // 3 == cc // 3))
    np.fill_diagonal(c, False); return c


# ---------------- Fixed encoder (shared across all M and both readouts) ----------------

def projection(rng):
    mask = rng.random((H, INPUT_DIM)) < DENSITY
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H, INPUT_DIM), dtype=np.float32)


def engrams(C, W, k):
    a = C @ W.T; a = a.toarray() if sparse.issparse(a) else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C.shape[0], H), np.uint8)
    Hs[np.arange(C.shape[0])[:, None], idx] = 1
    return Hs


def drive_vectors(sud):
    return DRIVE_LEVELS[sud.reshape(len(sud), N) - 1].astype(np.float32)


# ---------------- Readouts ----------------

def train_ridge_readout(Hs, B, ridge_lambda):
    Hm = Hs.astype(np.float64); Bm = B.astype(np.float64)
    Mloc = Hm.shape[0]
    A = Hm @ Hm.T + ridge_lambda * np.eye(Mloc)
    alpha = np.linalg.solve(A, Bm)
    return (Hm.T @ alpha).T  # (81, H)


def attention_readout(hq, Hs, B, tau=TAU):
    sim = Hs @ hq.astype(np.float32)
    p = softmax(sim / tau)
    return p @ B


# ---------------- Decoding / metrics (match K100 conventions) ----------------

def decode_digits(b):
    return np.argmin(np.abs(b[:, None] - DRIVE_LEVELS[None, :]), axis=1) + 1


def coincidence(d, conflict=None):
    C = d[:, None] == d[None, :]
    if conflict is not None:
        C[conflict] = False
    np.fill_diagonal(C, False); return C


def metrics(C, T, off):
    t = T[off].astype(float); p = C[off].astype(float)
    tp = np.sum((t == 1) & (p == 1)); fn = np.sum((t == 1) & (p == 0))
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    acc = np.mean(t == p); corr = np.nan if np.std(p) == 0 else np.corrcoef(t, p)[0, 1]
    return rec, acc, corr


def feedback(C, W, Hs, target, k):
    a = np.asarray(W @ C.astype(np.float32).ravel()).ravel()
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    overlap = np.sum(h * Hs[target]) / k
    sim = Hs @ h; p = softmax(sim / TAU)
    return overlap, p[target], int(np.argmax(p) == target)


# ---------------- Main sweep ----------------

def run():
    t_start = time.time()
    rng = np.random.default_rng(SEED_MEMORY)

    print(f'Generating pool of {M_MAX} unique solved Sudokus...')
    t0 = time.time()
    pool = generate_unique(M_MAX, rng)
    print(f'  done in {time.time() - t0:.1f}s')

    W = projection(rng)
    conf = conflict_matrix(); off = ~np.eye(N, dtype=bool)
    trng = np.random.default_rng(SEED_TRIALS)

    rows = []
    for M in MEMORY_LOADS:
        t_m = time.time()
        memories = pool[:M]
        C_all = np.asarray([relation(g).ravel() for g in memories], dtype=np.float32)
        Hs = engrams(C_all, W, K)
        B_all = drive_vectors(memories)

        t0 = time.time()
        W_HB = train_ridge_readout(Hs, B_all, RIDGE_LAMBDA)
        ridge_time = time.time() - t0

        n_trials = min(N_TRIALS, M)
        for noise in NOISE_LEVELS:
            targets = trng.choice(M, size=n_trials, replace=False)
            acc = {'regression': [], 'attention': []}
            for target in targets:
                hq = Hs[target].copy()
                nflip = int(round(noise * H))
                if nflip:
                    hq[trng.choice(H, size=nflip, replace=False)] ^= 1

                b_reg = W_HB @ hq.astype(np.float32)
                b_att = attention_readout(hq, Hs, B_all)

                for mode, b in [('regression', b_reg), ('attention', b_att)]:
                    digits = decode_digits(b)
                    digit_acc = np.mean(digits == memories[target].ravel())
                    C_rec = coincidence(digits, conf)
                    rec, ca, cc = metrics(C_rec, C_all[target].reshape(81, 81).astype(bool), off)
                    fo, fp, ft = feedback(C_rec, W, Hs, target, K)
                    acc[mode].append([digit_acc, rec, ca, cc, fo, fp, ft])

            for mode, vals in acc.items():
                m = np.nanmean(vals, axis=0)
                rows.append([M, noise * 100, mode, ridge_time, *m])

        print(f'  M={M:5d}  ridge_solve={ridge_time:6.2f}s  total={time.time() - t_m:6.2f}s')

    df = pd.DataFrame(rows, columns=[
        'M', 'noise_percent', 'readout', 'ridge_solve_seconds',
        'digit_cell_accuracy', 'coincidence_recall', 'coincidence_accuracy',
        'coincidence_correlation', 'feedback_overlap_fraction',
        'feedback_target_probability', 'feedback_top1_accuracy',
    ])
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    print(f'\nTotal wall time: {time.time() - t_start:.1f}s')
    print(df.to_string(index=False))
    return df


def estimate_capacity(df, metric='digit_cell_accuracy', threshold=CAPACITY_THRESHOLD):
    rows = []
    for noise in sorted(df.noise_percent.unique()):
        for mode in ['regression', 'attention']:
            sub = df[(df.noise_percent == noise) & (df.readout == mode)].sort_values('M')
            below = sub[sub[metric] < threshold]
            if len(below) == 0:
                cap = f'>= {sub.M.max()} (not reached)'
            else:
                first_fail_M = below.M.min()
                ok = sub[sub.M < first_fail_M]
                cap = ok.M.max() if len(ok) else f'< {first_fail_M}'
            rows.append({'noise_percent': noise, 'readout': mode, 'estimated_capacity_M': cap})
    return pd.DataFrame(rows)


COLORS = {'regression': 'tab:blue', 'attention': 'tab:orange'}


def plot_capacity_curve(df, noise=0.0):
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ['regression', 'attention']:
        sub = df[(df.noise_percent == noise) & (df.readout == mode)].sort_values('M')
        ax.plot(sub.M, sub.digit_cell_accuracy, 'o-', color=COLORS[mode], label=mode)
    ax.axhline(CAPACITY_THRESHOLD, color='gray', linestyle='--', linewidth=1,
               label=f'{int(CAPACITY_THRESHOLD * 100)}% threshold')
    ax.axvline(H, color='black', linestyle=':', linewidth=1, label=f'H={H}')
    ax.set_xscale('log')
    ax.set_xlabel('Number of stored memories (M)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'Storage capacity (clean query, noise={noise:.0f}%)')
    ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_capacity.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_capacity.png')


def plot_robustness_curve(df, M_ref=1000):
    if M_ref not in df.M.unique():
        M_ref = df.M.unique()[np.argmin(np.abs(df.M.unique() - M_ref))]
    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in ['regression', 'attention']:
        sub = df[(df.M == M_ref) & (df.readout == mode)].sort_values('noise_percent')
        ax.plot(sub.noise_percent, sub.digit_cell_accuracy, 'o-', color=COLORS[mode], label=mode)
    ax.axhline(CAPACITY_THRESHOLD, color='gray', linestyle='--', linewidth=1,
               label=f'{int(CAPACITY_THRESHOLD * 100)}% threshold')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'Noise robustness at M={M_ref} stored memories')
    ax.grid(alpha=0.3); ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}_robustness.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_robustness.png')


def plot_heatmaps(df):
    Ms = sorted(df.M.unique()); noises = sorted(df.noise_percent.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mode in zip(axes, ['regression', 'attention']):
        grid = np.full((len(Ms), len(noises)), np.nan)
        for i, M in enumerate(Ms):
            for j, noise in enumerate(noises):
                row = df[(df.M == M) & (df.noise_percent == noise) & (df.readout == mode)]
                if len(row):
                    grid[i, j] = row.digit_cell_accuracy.values[0]
        im = ax.imshow(grid, aspect='auto', vmin=0, vmax=1, cmap='viridis', origin='lower')
        ax.set_xticks(range(len(noises))); ax.set_xticklabels([f'{n:.0f}' for n in noises])
        ax.set_yticks(range(len(Ms))); ax.set_yticklabels(Ms)
        ax.set_xlabel('Query noise (%)'); ax.set_title(mode)
    axes[0].set_ylabel('Number of stored memories (M)')
    fig.colorbar(im, ax=axes, label='Digit-cell retrieval accuracy', shrink=0.8)
    fig.suptitle('Capacity x noise-robustness surface: ridge-regression vs softmax-attention readout')
    fig.savefig(f'{OUT_PREFIX}_heatmap.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}_heatmap.png')


if __name__ == '__main__':
    df = run()
    cap = estimate_capacity(df)
    cap.to_csv(f'{OUT_PREFIX}_capacity_estimates.csv', index=False)
    print('\nEstimated capacity (largest M with digit_cell_accuracy >= '
          f'{CAPACITY_THRESHOLD:.0%}):')
    print(cap.to_string(index=False))
    plot_capacity_curve(df, noise=0.0)
    plot_robustness_curve(df, M_ref=1000)
    plot_heatmaps(df)
