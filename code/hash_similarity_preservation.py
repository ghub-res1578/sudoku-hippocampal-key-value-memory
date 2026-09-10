"""
Locality-sensitive-hashing / similarity-preservation check for the engram encoder
(W_SH -> k-WTA), motivated by the fly olfactory mushroom-body hashing circuit (Dasgupta,
Stevens & Navlakha, 2017, Science): sparse random projection followed by winner-take-all
sparsification is known to approximately preserve relative similarity between inputs while
decorrelating/sparsifying the code. We test this directly for our encoder rather than assuming
it by analogy: does engram (H_keys) similarity between a corrupted query and its true target
track the RAW INPUT (coincidence-matrix) similarity between the corrupted and true C matrices,
across a wide range of corruption levels?

Every noise-robustness figure elsewhere in this project already shows DOWNSTREAM decode
accuracy degrading gracefully with input corruption; this script instead measures the
encoder's own geometry -- input similarity vs. engram similarity -- directly, one step
upstream of any decoding, which is the property the hashing-circuit analogy actually claims.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import sys
sys.path.insert(0, '.')
from separation_operator_ablation import build_store
from hippocampus_drive_readout import M, H, K, N

SEED_TRIALS = 13579
N_TARGETS = 30
CORRUPT_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]


def corrupt_C(C, frac, rng):
    """Flip a `frac` fraction of the N x N boolean relation matrix's off-diagonal entries,
    symmetrically (C[i,j] and C[j,i] together), leaving the diagonal at False."""
    Cc = C.copy()
    iu = np.triu_indices(N, k=1)
    n_pairs = len(iu[0])
    nflip = int(round(frac * n_pairs))
    if nflip:
        idx = rng.choice(n_pairs, size=nflip, replace=False)
        rows, cols = iu[0][idx], iu[1][idx]
        Cc[rows, cols] = ~Cc[rows, cols]
        Cc[cols, rows] = Cc[rows, cols]
    return Cc


def jaccard(a, b):
    inter = np.sum(a & b)
    union = np.sum(a | b)
    return inter / union if union else 1.0


def main():
    print('Rebuilding the M=200 engram store (same SEED as hippocampus_drive_readout.py)...')
    W_SH, H_keys, B_values, digits_true = build_store()

    trng = np.random.default_rng(SEED_TRIALS)
    targets = trng.choice(M, size=N_TARGETS, replace=False)

    print(f'Sweeping {len(CORRUPT_LEVELS)} corruption levels x {N_TARGETS} target grids...')
    rows = []
    for t in targets:
        digits_t = digits_true[t]
        C_true = (digits_t[:, None] == digits_t[None, :])
        np.fill_diagonal(C_true, False)
        h_true = H_keys[t].astype(bool)
        for frac in CORRUPT_LEVELS:
            C_noisy = corrupt_C(C_true, frac, trng)
            a = np.asarray(W_SH @ C_noisy.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_noisy = np.zeros(H, bool); h_noisy[idx] = True

            input_sim = jaccard(C_true, C_noisy)
            engram_sim = np.sum(h_noisy & h_true) / K
            rows.append({'target': int(t), 'corrupt_frac': frac,
                         'input_jaccard': input_sim, 'engram_overlap': engram_sim})

    df = pd.DataFrame(rows)
    df.to_csv('hash_similarity_preservation.csv', index=False)

    rho, p = spearmanr(df.input_jaccard, df.engram_overlap)
    print(f'\nSpearman rho(input Jaccard similarity, engram overlap) = {rho:.4f} '
          f'(p={p:.2e}), n={len(df)}')

    summary = df.groupby('corrupt_frac').agg(
        input_jaccard=('input_jaccard', 'mean'),
        engram_overlap=('engram_overlap', 'mean'),
        engram_overlap_se=('engram_overlap', lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()
    print(summary.to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    ax = axes[0]
    ax.scatter(df.input_jaccard, df.engram_overlap, s=10, alpha=0.35, color='tab:blue')
    ax.set_xlabel('Input similarity (Jaccard of same-digit pairs, corrupted vs. true C)')
    ax.set_ylabel('Engram overlap (shared active units / K)')
    ax.set_title(f'Similarity preservation (n={len(df)} trials, Spearman rho={rho:.3f})')
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.errorbar(summary.corrupt_frac * 100, summary.engram_overlap,
                yerr=summary.engram_overlap_se, fmt='o-', color='tab:green', capsize=3)
    ax.set_xlabel('C-matrix corruption (%)')
    ax.set_ylabel('Mean engram overlap with true target')
    ax.set_title('Engram overlap vs. input corruption')
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig('hash_similarity_preservation.png', dpi=150, bbox_inches='tight')
    print('\nSaved hash_similarity_preservation.csv and .png')


if __name__ == '__main__':
    main()
