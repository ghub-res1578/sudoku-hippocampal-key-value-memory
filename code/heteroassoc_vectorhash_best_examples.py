"""
Finds and renders the image examples where the Sudoku-attractor cleanup step visibly
rescues a badly-corrupted reconstruction: for each dataset (MNIST, Fashion-MNIST,
CIFAR-100-grayscale), searches across noise levels/trials for cases where the
INITIAL (pre-attractor) reconstruction is a poor match to the truth but the CLEANED
(post-attractor) reconstruction is both a correct grid recovery (exact=1) and a
high-correlation match -- i.e. the largest "rescue gap" -- and plots
noisy input / recon from initial engram / recon from cleaned engram / true original.
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
from heteroassoc_vectorhash_generalized import (
    load_dataset, train_pseudo_inverse_dual, k_wta, add_noise, iterate_sudoku_attractor,
)

M = 200
CANDIDATE_SIGMAS = [0.3, 0.4, 0.5]
N_TRIALS_PER_SIGMA = 30
N_EXAMPLES_PER_DATASET = 3
DATASETS = [
    ('mnist', 28, 'MNIST'),
    ('fashion_mnist', 28, 'Fashion-MNIST'),
    ('cifar100_gray', 32, 'CIFAR-100 (gray)'),
]


def build_store(dataset_name, M_local):
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)
    B_drive = drive_vectors(pool)
    conf = conflict_matrix()
    images, img_dim = load_dataset(dataset_name, M_local)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32))
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images)
    return dict(pool=pool, W_SH=W_SH, H_keys=H_keys, B_drive=B_drive, conf=conf,
                images=images, img_dim=img_dim, W_SI=W_SI, W_IS=W_IS)


def find_best_examples(store, n_examples):
    H_keys, B_drive, W_SH, conf = store['H_keys'], store['B_drive'], store['W_SH'], store['conf']
    images, W_SI, W_IS = store['images'], store['W_SI'], store['W_IS']
    M_local = images.shape[0]

    rng_test = np.random.default_rng(SEED + 4242)
    candidates = []
    for sigma in CANDIDATE_SIGMAS:
        for _ in range(N_TRIALS_PER_SIGMA):
            m = rng_test.integers(M_local)
            noise_seed = rng_test.integers(1 << 30)
            rng_noise = np.random.default_rng(noise_seed)
            x_noisy = add_noise(images[m:m + 1], sigma, rng_noise)[0]

            h_initial = k_wta(W_SI @ x_noisy)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            true_digits = store['pool'][m].reshape(N)
            exact = int(np.all(digits_final == true_digits))

            recon_initial = np.clip(W_IS @ h_initial.astype(np.float32), 0, 1)
            recon_cleaned = np.clip(W_IS @ h_cleaned.astype(np.float32), 0, 1)
            corr_initial = np.corrcoef(recon_initial, images[m])[0, 1]
            corr_cleaned = np.corrcoef(recon_cleaned, images[m])[0, 1]

            if exact and corr_cleaned > 0.85 and corr_initial < 0.65:
                gap = corr_cleaned - corr_initial
                candidates.append((gap, m, sigma, x_noisy, recon_initial, recon_cleaned,
                                    corr_initial, corr_cleaned))

    candidates.sort(key=lambda c: -c[0])
    # de-duplicate by target so we show variety, not the same digit repeated
    seen_targets = set()
    picked = []
    for c in candidates:
        if c[1] in seen_targets:
            continue
        seen_targets.add(c[1])
        picked.append(c)
        if len(picked) == n_examples:
            break
    return picked


def main():
    fig, axes = plt.subplots(len(DATASETS) * N_EXAMPLES_PER_DATASET, 4,
                              figsize=(9, 2.3 * len(DATASETS) * N_EXAMPLES_PER_DATASET))

    row = 0
    for dataset_name, side, label in DATASETS:
        print(f'Building store + searching best rescue examples for {label}...')
        store = build_store(dataset_name, M)
        best = find_best_examples(store, N_EXAMPLES_PER_DATASET)
        print(f'  found {len(best)} qualifying examples')

        for i, (gap, m, sigma, x_noisy, recon_initial, recon_cleaned, ci, cc) in enumerate(best):
            true_img = store['images'][m]
            col_data = [
                (x_noisy, f'noisy input\n(sigma={sigma})'),
                (recon_initial, f'recon: initial\nengram (corr={ci:.2f})'),
                (recon_cleaned, f'recon: cleaned\nengram (corr={cc:.2f})'),
                (true_img, 'true original'),
            ]
            for col, (img, title) in enumerate(col_data):
                ax = axes[row, col]
                ax.imshow(img.reshape(side, side), cmap='gray', vmin=0, vmax=1)
                ax.set_title(title, fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
            axes[row, 0].set_ylabel(label, fontsize=10, rotation=90)
            row += 1

    fig.suptitle('Largest rescue examples: attractor cleanup turns a poor initial\n'
                  'reconstruction into a correct one (all cases shown are exact grid recoveries)',
                  fontsize=12)
    fig.tight_layout()
    fig.savefig('heteroassoc_vectorhash_best_examples.png', dpi=150)
    print('\nSaved heteroassoc_vectorhash_best_examples.png')


if __name__ == '__main__':
    main()
