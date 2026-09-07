"""
Back to the original (non-modular, non-cross-modal) Sudoku-only key-value memory:
H=1000, K=150, fixed random W_SH, softmax attention readout, Sudoku conflict-mask
attractor loop -- all unchanged from hippocampus_drive_readout.py.

New corruption model: instead of noisy images (a different modality/encoder) or
direct engram bit-flips (an abstract corruption with no natural interpretation),
mask out a fraction of SUDOKU CELLS -- as if given a partially-filled puzzle with
only some clues known -- and build a PARTIAL coincidence matrix from just the known
cells (unknown-cell relations are conservatively treated as "no relation known", not
guessed). Project that partial coincidence matrix through the SAME W_SH used for
full grids, k-WTA, and see whether attention retrieval + the conflict-mask attractor
loop can fill in the missing cells.

This is the classic "half-filled Sudoku" pattern-completion test referenced in the
project's original complete_sudoku_hippocampus_analysis.py file, run here fresh
against the M=2000 pool and the exact machinery validated throughout this project.

For context: a real Sudoku puzzle needs a minimum of 17 clues (~21% of 81 cells) to
have a unique solution by pure logical deduction -- this sweep covers well below and
above that threshold to see how the associative mechanism compares.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hippocampus_drive_readout import (
    H, K, N, SEED, SEED_TRIALS,
    relation, make_projection, make_engrams, drive_vectors,
    decode_digits, conflict_matrix, coincidence, clean_engram, readout_attention,
)

M = 2000
CLUE_FRACTIONS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0]
N_TRIALS = 100
MAX_ITERS = 20

POOL_FILE = 'feedback_iterate_M2000_pool.npy'
PROJ_SEED = SEED + 50
OUT_PREFIX = 'sudoku_partial_grid_completion'


def partial_relation(g, known_mask):
    """C[i,j] = same digit AND both cells known; unknown-cell relations are 0
    (conservatively 'no relation known'), not guessed."""
    x = g.ravel()
    C = (x[:, None] == x[None, :])
    known2d = known_mask[:, None] & known_mask[None, :]
    C = C & known2d
    np.fill_diagonal(C, False)
    return C


def iterate_attractor(H_keys, B_drive, W_SH, conf, hq, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_drive, h_current)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH)
        key = h_next.tobytes()
        if key in visited:
            return h_next, digits, it, True
        visited[key] = it
        h_current = h_next
    return h_current, digits, max_iters, False


def run():
    pool = np.load(POOL_FILE)[:M]
    print(f'Loaded {len(pool)} Sudokus from {POOL_FILE}')
    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH)
    B_drive = drive_vectors(pool)
    digits_true = pool.reshape(M, N)
    conf = conflict_matrix()

    trng = np.random.default_rng(SEED_TRIALS + 20)
    rows = []
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        targets = trng.choice(M, size=N_TRIALS, replace=False)
        for t in targets:
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True

            C_partial = partial_relation(pool[t], known_mask)
            a = np.asarray(W_SH @ C_partial.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx] = 1

            sim0 = H_keys.astype(np.float32) @ h_q.astype(np.float32)
            top1_initial = int(np.argmax(sim0) == t)
            b0 = readout_attention(H_keys, B_drive, h_q)
            digits0 = decode_digits(b0)
            unknown_acc_initial = np.mean(digits0[~known_mask] == digits_true[t][~known_mask]) \
                if (~known_mask).any() else 1.0
            exact_initial = int(np.all(digits0 == digits_true[t]))

            h_final, digits_final, iters, converged = iterate_attractor(H_keys, B_drive, W_SH, conf, h_q)
            sim1 = H_keys.astype(np.float32) @ h_final.astype(np.float32)
            top1_final = int(np.argmax(sim1) == t)
            unknown_acc_final = np.mean(digits_final[~known_mask] == digits_true[t][~known_mask]) \
                if (~known_mask).any() else 1.0
            exact_final = int(np.all(digits_final == digits_true[t]))

            rows.append({
                'clue_fraction': clue_frac, 'n_known': n_known, 'target': int(t), 'iterations': iters,
                'top1_initial': top1_initial, 'top1_final': top1_final,
                'unknown_cell_acc_initial': unknown_acc_initial, 'unknown_cell_acc_final': unknown_acc_final,
                'exact_initial': exact_initial, 'exact_final': exact_final,
            })
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        print(f"  clue_frac={clue_frac:.2f} (n_known={n_known:2d})  mean_iter={np.mean([r['iterations'] for r in sub]):5.2f}  "
              f"top1: {np.mean([r['top1_initial'] for r in sub]):6.2%} -> {np.mean([r['top1_final'] for r in sub]):6.2%}  "
              f"unknown_cell_acc: {np.mean([r['unknown_cell_acc_initial'] for r in sub]):6.2%} -> "
              f"{np.mean([r['unknown_cell_acc_final'] for r in sub]):6.2%}  "
              f"exact_grid: {np.mean([r['exact_initial'] for r in sub]):6.2%} -> "
              f"{np.mean([r['exact_final'] for r in sub]):6.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def plot(df):
    summary = df.groupby('clue_fraction').agg(
        top1_initial=('top1_initial', 'mean'), top1_final=('top1_final', 'mean'),
        unknown_initial=('unknown_cell_acc_initial', 'mean'), unknown_final=('unknown_cell_acc_final', 'mean'),
        exact_initial=('exact_initial', 'mean'), exact_final=('exact_final', 'mean'),
    ).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(summary.clue_fraction, summary.top1_initial, 'o--', color='tab:orange', label='initial')
    ax.plot(summary.clue_fraction, summary.top1_final, 'o-', color='tab:green', label='converged')
    ax.axhline(1 / M, color='gray', linestyle=':', linewidth=1, label='chance (1/M)')
    ax.axvline(17 / 81, color='black', linestyle=':', alpha=0.5, label='17-clue uniqueness bound')
    ax.set_xlabel('Fraction of cells known (clues)'); ax.set_ylabel('Top-1 item identification')
    ax.set_title(f'Identity recovery from partial grid (M={M})'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(summary.clue_fraction, summary.unknown_initial, 'o--', color='tab:orange', label='initial')
    ax.plot(summary.clue_fraction, summary.unknown_final, 'o-', color='tab:green', label='converged')
    ax.axvline(17 / 81, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Fraction of cells known (clues)'); ax.set_ylabel('Accuracy on UNKNOWN cells')
    ax.set_title('Fill-in-the-blank accuracy'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(summary.clue_fraction, summary.exact_initial, 'o--', color='tab:orange', label='initial')
    ax.plot(summary.clue_fraction, summary.exact_final, 'o-', color='tab:green', label='converged')
    ax.axvline(17 / 81, color='black', linestyle=':', alpha=0.5)
    ax.set_xlabel('Fraction of cells known (clues)'); ax.set_ylabel('Exact full-grid match rate')
    ax.set_title('Complete puzzle solved exactly'); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle('Partial-Sudoku-grid pattern completion via the hippocampal key-value memory')
    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')


if __name__ == '__main__':
    df = run()
    plot(df)
