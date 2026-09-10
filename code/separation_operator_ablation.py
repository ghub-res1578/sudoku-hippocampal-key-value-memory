"""
Separation-operator ablation, testing the theoretical framework from Gershman, Fiete & Irie
(2025, Neuron, "Key-value memory in the brain"): correlation-matrix memory (Kohonen 1972),
sparse distributed memory (Kanerva 1988), dense associative memory / modern Hopfield networks
(Krotov & Hopfield 2016; Ramsauer et al. 2021), and transformer self-attention are all the SAME
dual-form key-value memory

    b_hat = sum_n sigma(S(K,q))_n * v_n

differing only in the "separation operator" sigma applied to key-query similarity. Our
attention readout has so far only used sigma = softmax. This script swaps in alternative
separation operators -- identity (no separation), a sharper softmax, a rectified-polynomial
power (modern-Hopfield-style), a hard top-j threshold (sparse-distributed-memory-style), and
argmax/winner-take-all (the "ideal noiseless" operator the paper calls out explicitly as
maximally discriminative but least robust) -- and re-runs the SAME query-noise sweep used
elsewhere in this project (0-100% of H bits flipped), with every operator evaluated on
IDENTICAL corrupted queries at each noise level (fully paired design).

No conflict-mask cleanup is applied here -- the goal is to isolate the readout operator's own
selectivity/robustness trade-off, not to re-test the cleanup mechanism (already characterized
in hippocampus_drive_readout.py / hippocampus_feedback_noise_zoomed.py).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    SEED, M, H, K, N, generate_unique, relation, make_projection, make_engrams,
    drive_vectors, decode_digits, NOISE_SWEEP,
)

SEED_TRIALS = 24680
N_TRIALS = 100

OPERATORS = [
    ('identity (no separation)',            dict(kind='identity')),
    ('softmax tau=2.0 (baseline)',          dict(kind='softmax', tau=2.0)),
    ('softmax tau=0.5 (sharper)',           dict(kind='softmax', tau=0.5)),
    ('polynomial n=8 (dense assoc. mem.)',  dict(kind='polynomial', n=8)),
    ('threshold top-5 (sparse dist. mem.)', dict(kind='threshold', j=5)),
    ('argmax (winner-take-all)',            dict(kind='threshold', j=1)),
]


def build_store():
    rng = np.random.default_rng(SEED)
    sud = generate_unique(M, rng)
    C_all = np.asarray([relation(g).ravel() for g in sud], dtype=np.float32)
    W_SH = make_projection(rng)
    H_keys = make_engrams(C_all, W_SH)
    B_values = drive_vectors(sud)
    digits_true = sud.reshape(M, N)
    return W_SH, H_keys, B_values, digits_true


def separate(sim, kind, **kw):
    """Applies separation operator sigma to a (M,) similarity vector, returning
    normalized attention weights p (M,) with sum(p) = 1."""
    if kind == 'identity':
        s = sim - sim.min()          # keep weights non-negative for a meaningful average
        return s / (s.sum() + 1e-12)
    if kind == 'softmax':
        return softmax(sim / kw['tau'])
    if kind == 'polynomial':
        # normalize by K (max possible overlap) first -- sim^n on the raw 0..K scale barely
        # separates anything for modest n, since (unlike softmax's exponential) a power of a
        # near-1 ratio stays near 1; the fair dense-associative-memory analog sharpens the
        # *ratio* of normalized similarities.
        sim_n = np.clip(sim, 0, None) / K
        r = sim_n ** kw['n']
        return r / (r.sum() + 1e-12)
    if kind == 'threshold':
        j = kw['j']
        idx = np.argpartition(sim, -j)[-j:]
        p = np.zeros_like(sim)
        p[idx] = 1.0 / j
        return p
    raise ValueError(kind)


def run_ablation(H_keys, B_values, digits_true, noise_levels=NOISE_SWEEP, n_trials=N_TRIALS,
                  seed=SEED_TRIALS):
    trng = np.random.default_rng(seed)
    rows = []
    for noise in noise_levels:
        targets = trng.choice(M, size=min(n_trials, M), replace=False)
        nflip = int(round(noise * H))
        queries = []
        for t in targets:
            hq = H_keys[t].copy()
            if nflip:
                hq[trng.choice(H, size=nflip, replace=False)] ^= 1
            queries.append(hq)

        level_accs = {label: [] for label, _ in OPERATORS}
        for t, hq in zip(targets, queries):
            sim = H_keys.astype(np.float32) @ hq.astype(np.float32)
            for label, kw in OPERATORS:
                p = separate(sim, **kw)
                b_hat = p @ B_values
                digits_hat = decode_digits(b_hat)
                level_accs[label].append(np.mean(digits_hat == digits_true[t]))

        for label, accs in level_accs.items():
            rows.append({'operator': label, 'noise_percent': noise * 100,
                         'digit_cell_acc': float(np.mean(accs))})
        summary = '  '.join(f'{lbl.split(" ")[0]}={np.mean(level_accs[lbl]):.3f}'
                             for lbl, _ in OPERATORS)
        print(f'noise={noise:5.0%}  {summary}')
    return pd.DataFrame(rows)


def plot_ablation(df, out='separation_operator_ablation.png'):
    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=150)
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(OPERATORS)))
    for (label, _), color in zip(OPERATORS, colors):
        sub = df[df.operator == label]
        ax.plot(sub.noise_percent, sub.digit_cell_acc, 'o-', color=color, label=label,
                linewidth=1.8, markersize=4)
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'Separation-operator ablation (M={M}, no cleanup pass, n={N_TRIALS}/point)')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')


def main():
    print('Rebuilding the M=200 engram store (same SEED as hippocampus_drive_readout.py)...')
    W_SH, H_keys, B_values, digits_true = build_store()
    print(f'\nRunning separation-operator ablation: {len(OPERATORS)} operators x '
          f'{len(NOISE_SWEEP)} noise levels x {N_TRIALS} paired trials...')
    df = run_ablation(H_keys, B_values, digits_true)
    df.to_csv('separation_operator_ablation.csv', index=False)
    print('\nSaved separation_operator_ablation.csv')
    plot_ablation(df)


if __name__ == '__main__':
    main()
