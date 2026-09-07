"""
At M=2000, iterate the conflict-mask feedback loop repeatedly instead of stopping
after one pass:

    h_0 = noisy query
    b_i      = softmax(H_keys @ h_i / TAU) @ B_values
    digits_i = decode(b_i)
    C_i      = coincidence(digits_i, conflict_mask)
    h_{i+1}  = k-WTA(W_SH @ C_i)

repeated until the engram stops changing (a fixed point: h_{i+1} == h_i) or enters a
short cycle (h_{i+1} matches an earlier h_j) -- either way, further iterations would
be identical, so that's "as close to max as this attractor reaches." Records the
number of passes taken to reach that point, per trial and per noise level.
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

M = 2000
NOISE_LEVELS = [0.0, 0.10, 0.20, 0.30, 0.35, 0.38, 0.40, 0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.70]
N_TRIALS = 50
MAX_ITERS = 20

OUT_PREFIX = 'feedback_iterate_M2000'
POOL_FILE = f'{OUT_PREFIX}_pool.npy'
PROJ_SEED = SEED + 1   # independent stream, so caching the pool never perturbs W_SH


def build_store():
    """Fresh, self-consistent M=2000 engram/drive store. The Sudoku pool (the slow
    ~2 minute part) is cached to disk on an independent RNG stream from the encoder,
    so a cache hit always reproduces the same W_SH regardless of whether the pool
    was just generated or loaded."""
    try:
        pool = np.load(POOL_FILE)
        print(f'Loaded cached pool of {len(pool)} Sudokus from {POOL_FILE}')
    except FileNotFoundError:
        rng_pool = np.random.default_rng(SEED)
        print(f'Generating pool of {M} unique solved Sudokus...')
        pool = generate_unique(M, rng_pool)
        np.save(POOL_FILE, pool)

    rng_proj = np.random.default_rng(PROJ_SEED)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH)
    B_values = drive_vectors(pool)
    digits_true = pool.reshape(M, N)
    return H_keys, B_values, digits_true, W_SH


def iterate_feedback(H_keys, B_values, W_SH, conf, hq, true_digits, max_iters=MAX_ITERS):
    h_current = hq.copy()
    visited = {h_current.tobytes(): 0}
    digits = decode_digits(readout_attention(H_keys, B_values, h_current))
    initial_acc = np.mean(digits == true_digits)   # accuracy before any feedback pass
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_values, h_current)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH)
        key = h_next.tobytes()
        if key in visited:
            cycle_len = it - visited[key]
            acc = np.mean(digits == true_digits)
            return {'iterations': it, 'converged': True, 'cycle_length': cycle_len,
                    'initial_acc': initial_acc, 'final_acc': acc}
        visited[key] = it
        h_current = h_next
    acc = np.mean(digits == true_digits)
    return {'iterations': max_iters, 'converged': False, 'cycle_length': None,
            'initial_acc': initial_acc, 'final_acc': acc}


def run():
    H_keys, B_values, digits_true, W_SH = build_store()
    conf = conflict_matrix()
    trng = np.random.default_rng(SEED_TRIALS + 3)

    rows = []
    for noise in NOISE_LEVELS:
        targets = trng.choice(M, size=N_TRIALS, replace=False)
        nflip = int(round(noise * H))
        for t in targets:
            hq = H_keys[t].copy()
            if nflip:
                hq[trng.choice(H, size=nflip, replace=False)] ^= 1
            res = iterate_feedback(H_keys, B_values, W_SH, conf, hq, digits_true[t])
            rows.append({'noise_percent': noise * 100, 'target': int(t), **res})
        sub = [r for r in rows if r['noise_percent'] == noise * 100]
        mean_it = np.mean([r['iterations'] for r in sub])
        mean_init = np.mean([r['initial_acc'] for r in sub])
        mean_acc = np.mean([r['final_acc'] for r in sub])
        frac_conv = np.mean([r['converged'] for r in sub])
        print(f'  noise={noise:5.0%}  mean_iterations={mean_it:5.2f}  '
              f'initial_acc={mean_init:7.4%}  final_acc={mean_acc:7.4%}  '
              f'delta={mean_acc - mean_init:+7.4%}  converged_within_cap={frac_conv:6.2%}')

    df = pd.DataFrame(rows)
    df.to_csv(f'{OUT_PREFIX}.csv', index=False)
    return df


def plot(df):
    summary = df.groupby('noise_percent').agg(
        mean_iterations=('iterations', 'mean'),
        initial_acc=('initial_acc', 'mean'),
        final_acc=('final_acc', 'mean'),
        converged_frac=('converged', 'mean'),
    ).reset_index()
    summary['delta'] = summary.final_acc - summary.initial_acc
    summary.to_csv(f'{OUT_PREFIX}_summary.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(summary.noise_percent, summary.mean_iterations, 'o-', color='tab:purple')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Mean iterations to reach a fixed point / cycle')
    ax.set_title(f'M={M}: iterations needed to converge')
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(summary.noise_percent, summary.initial_acc, 'o-', color='tab:orange',
             label='initial (before any feedback)')
    ax.plot(summary.noise_percent, summary.final_acc, 'o-', color='tab:green',
             label='at convergence (iterated)')
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'M={M}: initial vs converged accuracy')
    ax.grid(alpha=0.3); ax.legend()

    fig.tight_layout()
    fig.savefig(f'{OUT_PREFIX}.png', dpi=150)
    print(f'Saved plot to {OUT_PREFIX}.png')
    print('\nSummary (initial accuracy vs accuracy at convergence):')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    df = run()
    plot(df)
