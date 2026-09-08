"""
Visual demonstration of the Vector-HaSH-style pipeline: for a few example images at a few
representative noise levels, show noisy input -> reconstruction from the INITIAL (pre-
cleanup) engram -> reconstruction from the CLEANED (post-Sudoku-attractor) engram -> true
original, as actual images.
"""
import numpy as np
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
from heteroassoc_vectorhash_pipeline import (
    train_pseudo_inverse_dual, k_wta, add_noise, iterate_sudoku_attractor, load_mnist_flat, M
)

SIGMAS_TO_SHOW = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N_EXAMPLES = 4


def main():
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)
    B_drive = drive_vectors(pool)
    conf = conflict_matrix()

    images, image_labels = load_mnist_flat(M)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32))
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images)

    rng_test = np.random.default_rng(SEED + 2468)
    example_targets = rng_test.choice(M, size=N_EXAMPLES, replace=False)

    fig, axes = plt.subplots(N_EXAMPLES, len(SIGMAS_TO_SHOW) + 1,
                              figsize=(1.7 * (len(SIGMAS_TO_SHOW) + 1), 1.9 * N_EXAMPLES))
    for row, m in enumerate(example_targets):
        axes[row, 0].imshow(images[m].reshape(28, 28), cmap='gray')
        axes[row, 0].set_title('true\noriginal' if row == 0 else '', fontsize=9)
        axes[row, 0].set_ylabel(f'digit {image_labels[m]}', fontsize=9)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])

        for col, sigma in enumerate(SIGMAS_TO_SHOW):
            rng_noise = np.random.default_rng(SEED + 999 + m * 13 + int(sigma * 100))
            x_noisy = add_noise(images[m:m + 1], sigma, rng_noise)[0]
            h_initial = k_wta(W_SI @ x_noisy)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            img_recon = np.clip(W_IS @ h_cleaned.astype(np.float32), 0, 1)
            overlap = np.sum(h_cleaned * H_keys[m]) / K

            axes[row, col + 1].imshow(img_recon.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            if row == 0:
                axes[row, col + 1].set_title(f'sigma={sigma}', fontsize=9)
            axes[row, col + 1].set_xlabel(f'ov={overlap:.2f}', fontsize=8)
            axes[row, col + 1].set_xticks([]); axes[row, col + 1].set_yticks([])

    fig.suptitle('Reconstructed image (from the CLEANED, post-Sudoku-attractor engram) '
                  'vs. increasing input noise', fontsize=12)
    fig.tight_layout()
    fig.savefig('heteroassoc_vectorhash_visual_cleaned.png', dpi=150)
    print('Saved heteroassoc_vectorhash_visual_cleaned.png')

    # ---- second figure: noisy input vs initial-engram recon vs cleaned-engram recon vs truth ----
    fig2, axes2 = plt.subplots(N_EXAMPLES, 4, figsize=(9, 2.2 * N_EXAMPLES))
    sigma_demo = 0.4  # the "sweet spot" rescue regime from the summary curve
    for row, m in enumerate(example_targets):
        rng_noise = np.random.default_rng(SEED + 999 + m * 13 + int(sigma_demo * 100))
        x_noisy = add_noise(images[m:m + 1], sigma_demo, rng_noise)[0]
        h_initial = k_wta(W_SI @ x_noisy)
        recon_initial = np.clip(W_IS @ h_initial.astype(np.float32), 0, 1)
        h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
            H_keys, B_drive, W_SH, conf, h_initial)
        recon_cleaned = np.clip(W_IS @ h_cleaned.astype(np.float32), 0, 1)

        for col, (img, title) in enumerate([
            (x_noisy, 'noisy input' if row == 0 else ''),
            (recon_initial, 'recon (initial\nengram)' if row == 0 else ''),
            (recon_cleaned, 'recon (cleaned\nengram)' if row == 0 else ''),
            (images[m], 'true original' if row == 0 else ''),
        ]):
            axes2[row, col].imshow(img.reshape(28, 28), cmap='gray', vmin=0, vmax=1)
            axes2[row, col].set_title(title, fontsize=9)
            axes2[row, col].set_xticks([]); axes2[row, col].set_yticks([])

    fig2.suptitle(f'Full pipeline at sigma={sigma_demo} (the rescue "sweet spot")', fontsize=12)
    fig2.tight_layout()
    fig2.savefig('heteroassoc_vectorhash_visual_pipeline.png', dpi=150)
    print('Saved heteroassoc_vectorhash_visual_pipeline.png')


if __name__ == '__main__':
    main()
