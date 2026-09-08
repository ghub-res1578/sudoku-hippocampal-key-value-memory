"""
Vector-HaSH-style four-stage pipeline:

  1. External cue (noisy image) -> activates an engram, via BIDIRECTIONAL Hebbian/pseudo-
     inverse heteroassociation -- W_SI (sensory->hippocampus) and W_IS (hippocampus->sensory)
     trained on M DISTINCT (image_m, engram_m) pairs, one memory per pair (NOT many exemplars
     forced onto one shared class target, which is what the old W_IH regression did and why
     it generalized poorly). This is the only plastic, content-specific part -- its job is
     only to get "close enough", matching Vector-HaSH's actual division of labor.
  2. The engram selects a set of DRIVES via the existing, already-validated attention
     readout (readout_attention over the M stored (engram, drive) pairs) -- unchanged.
  3. If the drives are clean/strong enough, decoding digits -> building the coincidence
     matrix -> re-encoding via W_SH "activates" a legitimate Sudoku-consistent engram --
     this is our existing conflict-mask cleanup step, playing the role Vector-HaSH's fixed
     grid-hippocampus recurrence plays (the ACTUAL error-correcting scaffold dynamics,
     content-independent in their case, Sudoku-legality-specific in ours -- this remains the
     one structural difference from a true Vector-HaSH scaffold, flagged honestly below).
  4. The cleaned engram is decoded back to a reconstructed image via W_IS -- the hetero-
     associative link run in reverse, exactly Vector-HaSH's hippocampus-to-sensory step.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    H, N, K, SEED, TAU, RIDGE_LAMBDA,
    relation, make_projection, make_engrams, drive_vectors, generate_unique,
    readout_attention, decode_digits, conflict_matrix, coincidence, clean_engram,
)

IMG_DIM = 784
M = 200
MAX_ITERS = 20
NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
N_TRIALS_PER_SIGMA = 60


def load_mnist_flat(n_images):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), _ = mnist.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    rng = np.random.default_rng(SEED + 555)
    idx = rng.choice(len(x_train), size=n_images, replace=False)
    return x_train[idx].reshape(n_images, -1), y_train[idx]


def train_pseudo_inverse_dual(X, T, ridge_lambda=RIDGE_LAMBDA):
    """W (out_dim x in_dim) minimizing ||X @ W.T - T||^2 + ridge||W||^2, DUAL form
    (inverts an M x M matrix) -- efficient when M < in_dim, as here (M=200 << 784, 1000)."""
    Xm = X.astype(np.float64); Tm = T.astype(np.float64)
    Mloc = Xm.shape[0]
    Amat = Xm @ Xm.T + ridge_lambda * np.eye(Mloc)
    alpha = np.linalg.solve(Amat, Tm)
    return (Xm.T @ alpha).T.astype(np.float32)  # (out_dim, in_dim)


def k_wta(a, k=K):
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


def add_noise(images, sigma, rng):
    return np.clip(images + rng.normal(0, sigma, images.shape), 0, 1).astype(np.float32)


def iterate_sudoku_attractor(H_keys, B_drive, W_SH, conf, h_query, max_iters=MAX_ITERS):
    h_current = h_query.copy()
    visited = {h_current.tobytes(): 0}
    for it in range(1, max_iters + 1):
        b = readout_attention(H_keys, B_drive, h_current, tau=TAU)
        digits = decode_digits(b)
        C_clean = coincidence(digits, conf)
        h_next = clean_engram(C_clean, W_SH, K)
        key = h_next.tobytes()
        if key in visited:
            return h_next, digits, it, True
        visited[key] = it
        h_current = h_next
    return h_current, digits, max_iters, False


def main():
    print(f'Building M={M} stored Sudoku grids + M hetero-associated MNIST images...')
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)  # (M, H) -- true target engrams
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M, N)
    conf = conflict_matrix()

    images, image_labels = load_mnist_flat(M)  # M distinct images, one per stored grid

    print('Training bidirectional Hebbian/pseudo-inverse heteroassociation '
          '(W_SI: image->engram, W_IS: engram->image), one memory per pair...')
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32))       # (H, 784)
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images)      # (784, H)

    # sanity check: clean-query round trip (no noise at all)
    h_clean_pred = np.array([k_wta(W_SI @ images[m]) for m in range(M)])
    clean_overlap = np.mean([np.sum(h_clean_pred[m] * H_keys[m]) / K for m in range(M)])
    recon_clean = W_IS @ H_keys.astype(np.float32).T  # reconstruct from the TRUE engram
    recon_corr = np.mean([np.corrcoef(recon_clean[:, m], images[m])[0, 1] for m in range(M)])
    print(f'  sanity check: clean image->engram overlap={clean_overlap:.3f}, '
          f'clean engram->image reconstruction correlation={recon_corr:.3f}')

    rng_test = np.random.default_rng(SEED + 999)
    rows = []
    for sigma in NOISE_SIGMAS:
        for _ in range(N_TRIALS_PER_SIGMA):
            m = rng_test.integers(M)
            x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]

            # Stage 1: cue activates an engram (Hebbian/pseudo-inverse heteroassociation)
            h_initial = k_wta(W_SI @ x_noisy)
            initial_overlap = np.sum(h_initial * H_keys[m]) / K

            # Stage 2+3: engram selects drives -> if clean/strong, sudoku activates via
            # coincidence matrix (existing, unchanged mechanism; iterated to convergence)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / K
            cell_acc = np.mean(digits_final == digits_true_all[m])
            exact = int(np.all(digits_final == digits_true_all[m]))

            # Stage 4: cleaned engram decoded back to a reconstructed image
            img_recon = W_IS @ h_cleaned.astype(np.float32)
            recon_correlation = np.corrcoef(img_recon, images[m])[0, 1]
            recon_mse = np.mean((img_recon - images[m]) ** 2)

            # also reconstruct from the INITIAL (pre-attractor) engram, for comparison
            img_recon_initial = W_IS @ h_initial.astype(np.float32)
            recon_correlation_initial = np.corrcoef(img_recon_initial, images[m])[0, 1]

            rows.append({
                'sigma': sigma, 'target': m,
                'initial_overlap': initial_overlap, 'cleaned_overlap': cleaned_overlap,
                'iters': iters, 'cell_acc': cell_acc, 'exact': exact,
                'recon_corr_initial': recon_correlation_initial,
                'recon_corr_cleaned': recon_correlation, 'recon_mse_cleaned': recon_mse,
            })
        sub = [r for r in rows if r['sigma'] == sigma]
        print(f'sigma={sigma:.2f}  initial_ov={np.mean([r["initial_overlap"] for r in sub]):.3f}  '
              f'cleaned_ov={np.mean([r["cleaned_overlap"] for r in sub]):.3f}  '
              f'exact={np.mean([r["exact"] for r in sub]):.2%}  '
              f'recon_corr(initial->cleaned)={np.mean([r["recon_corr_initial"] for r in sub]):.3f}'
              f'->{np.mean([r["recon_corr_cleaned"] for r in sub]):.3f}')

    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_vectorhash_pipeline.csv', index=False)

    summary = df.groupby('sigma').agg(
        initial_overlap=('initial_overlap', 'mean'), cleaned_overlap=('cleaned_overlap', 'mean'),
        exact=('exact', 'mean'),
        recon_corr_initial=('recon_corr_initial', 'mean'), recon_corr_cleaned=('recon_corr_cleaned', 'mean'),
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(summary.sigma, summary.initial_overlap, 'o--', color='gray', label='initial (stage 1 only)')
    axes[0].plot(summary.sigma, summary.cleaned_overlap, 'o-', color='tab:purple', label='cleaned (after stage 3)')
    axes[0].axhline(K / H, color='gray', linestyle=':', label=f'chance (K/H={K/H:.2f})')
    axes[0].set_xlabel('Image noise (sigma)'); axes[0].set_ylabel('Engram overlap with true target')
    axes[0].set_title('Engram quality: before vs. after Sudoku cleanup'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].plot(summary.sigma, summary.recon_corr_initial, 'o--', color='gray', label='reconstruction from initial engram')
    axes[1].plot(summary.sigma, summary.recon_corr_cleaned, 'o-', color='tab:green', label='reconstruction from cleaned engram')
    axes[1].set_xlabel('Image noise (sigma)'); axes[1].set_ylabel('Reconstructed-vs-true image correlation')
    axes[1].set_title('Image reconstruction quality: before vs. after Sudoku cleanup'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('heteroassoc_vectorhash_pipeline.png', dpi=150)
    print('\nSaved heteroassoc_vectorhash_pipeline.png')


if __name__ == '__main__':
    main()
