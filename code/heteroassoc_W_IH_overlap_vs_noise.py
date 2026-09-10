"""
Goes back to the ORIGINAL cross-modal implementation (mnist_class_hash.py): W_IH, a
RIDGE-REGRESSION-TRAINED weight matrix (H x 784) -- not a fixed random projection -- trained
on many (500) MNIST exemplars per class, all regressed onto their class's single target
Sudoku engram. That script only measured downstream top-1 class accuracy; this measures the
actual ENGRAM OVERLAP (fraction of K shared active units) between a noisy test image's
engram (h_q = k_wta(W_IH @ x_noisy)) and its true target engram, directly, as a function of
noise -- the same metric used for the fixed-random-projection (W_IMG) comparison, so the two
encoders can be compared apples-to-apples on identical grounds, no attention/hetero-
association step involved for either one.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    H, K, N, SEED, SEED_TRIALS, RIDGE_LAMBDA, DENSITY,
    relation, make_projection, make_engrams, generate_unique,
)

N_CLASSES = 9
N_PER_CLASS_TRAIN = 500
N_PER_CLASS_TEST = 100
IMG_DIM = 28 * 28
NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0]
N_TRIALS = 100
M_SUDOKU_POOL = 200
PROJ_SEED = SEED + 40
REF_SEED = SEED + 41


def load_mnist_by_class(n_train, n_test):
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train_by_class, test_by_class = {}, {}
    for d in range(1, N_CLASSES + 1):
        idx_tr = np.flatnonzero(y_train == d)[:n_train]
        idx_te = np.flatnonzero(y_test == d)[:n_test]
        train_by_class[d] = x_train[idx_tr].reshape(len(idx_tr), -1).astype(np.float32) / 255.0
        test_by_class[d] = x_test[idx_te].reshape(len(idx_te), -1).astype(np.float32) / 255.0
    return train_by_class, test_by_class


def train_pseudo_inverse_primal(X, T, ridge_lambda=RIDGE_LAMBDA):
    Xm = X.astype(np.float64); Tm = T.astype(np.float64)
    D = Xm.shape[1]
    A = Xm.T @ Xm + ridge_lambda * np.eye(D)
    W = Tm.T @ Xm @ np.linalg.inv(A)
    return W.astype(np.float32)


def k_wta(a, k=K):
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


def make_image_projection(rng):
    mask = rng.random((H, IMG_DIM)) < DENSITY
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H, IMG_DIM), dtype=np.float32)


def main():
    rng_ref = np.random.default_rng(REF_SEED)
    rng_proj = np.random.default_rng(PROJ_SEED)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_ref = make_engrams(C_ref, W_SH, K)  # (9, H) target engram per class

    print(f'Loading MNIST: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    AllImages, AllTargets = [], []
    for d in range(1, N_CLASSES + 1):
        imgs = train_by_class[d]
        AllImages.append(imgs)
        AllTargets.append(np.tile(H_keys_ref[d - 1], (len(imgs), 1)))
    AllImages = np.concatenate(AllImages, axis=0)
    AllTargets = np.concatenate(AllTargets, axis=0)

    print(f'Training W_IH ({H} x {IMG_DIM}) via ridge regression on {AllImages.shape[0]} exemplars...')
    W_IH = train_pseudo_inverse_primal(AllImages, AllTargets.astype(np.float32))

    print(f'Building W_IMG (fixed random projection, same shape, for direct comparison)...')
    rng_img = np.random.default_rng(SEED + 777)
    W_IMG = make_image_projection(rng_img)
    # W_IMG has no reason to overlap a SPECIFIC target on its own (it's untrained/unrelated) --
    # so for a fair, direct, apples-to-apples "does the weight matrix preserve its OWN clean
    # mapping under noise" comparison, measure self-consistency: does the noisy image's W_IMG
    # engram still overlap the SAME image's own clean W_IMG engram?
    clean_img_engrams = {}
    for d in range(1, N_CLASSES + 1):
        a = np.asarray(W_IMG @ test_by_class[d].T).T
        idx = np.argpartition(a, -K, axis=1)[:, -K:]
        Hs = np.zeros((a.shape[0], H), np.uint8)
        Hs[np.arange(a.shape[0])[:, None], idx] = 1
        clean_img_engrams[d] = Hs

    trng = np.random.default_rng(SEED_TRIALS + 9)
    rows = []
    for sigma in NOISE_SIGMAS:
        for _ in range(N_TRIALS):
            d = trng.integers(1, N_CLASSES + 1)
            test_idx = trng.integers(len(test_by_class[d]))
            img = test_by_class[d][test_idx]
            x_noisy = np.clip(img + trng.normal(0, sigma, IMG_DIM), 0, 1).astype(np.float32)

            # W_IH: regression-trained, overlap vs its TRAINED target engram
            h_ih = k_wta(W_IH @ x_noisy)
            overlap_ih = np.sum(h_ih * H_keys_ref[d - 1]) / K

            # W_IMG: fixed random projection, overlap vs its OWN clean self
            a_img = np.asarray(W_IMG @ x_noisy).ravel()
            h_img = k_wta(a_img)
            overlap_img_self = np.sum(h_img * clean_img_engrams[d][test_idx]) / K

            rows.append({'sigma': sigma, 'true_class': d,
                         'overlap_W_IH_vs_target': overlap_ih,
                         'overlap_W_IMG_self_consistency': overlap_img_self})
        sub = [r for r in rows if r['sigma'] == sigma]
        ih_vals = [r['overlap_W_IH_vs_target'] for r in sub]
        img_vals = [r['overlap_W_IMG_self_consistency'] for r in sub]
        print(f'  sigma={sigma:4.2f}  W_IH overlap={np.mean(ih_vals):.3f}+-{np.std(ih_vals):.3f}  '
              f'W_IMG self-overlap={np.mean(img_vals):.3f}+-{np.std(img_vals):.3f}')

    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_W_IH_overlap_vs_noise.csv', index=False)

    summary = df.groupby('sigma').agg(
        ih_mean=('overlap_W_IH_vs_target', 'mean'), ih_std=('overlap_W_IH_vs_target', 'std'),
        img_mean=('overlap_W_IMG_self_consistency', 'mean'), img_std=('overlap_W_IMG_self_consistency', 'std'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(summary.sigma, summary.ih_mean, yerr=summary.ih_std, fmt='o-', color='tab:purple',
                capsize=3, label='W_IH (ridge-regression trained) vs. its trained target')
    ax.errorbar(summary.sigma, summary.img_mean, yerr=summary.img_std, fmt='s-', color='tab:blue',
                capsize=3, label='W_IMG (fixed random projection) self-consistency')
    ax.axhline(K / H, color='gray', linestyle=':', label=f'chance level (K/H={K/H:.2f})')
    ax.set_xlabel('Image noise (Gaussian sigma)'); ax.set_ylabel('Engram overlap (fraction of K)')
    ax.set_title('Learned (regression) vs. fixed (random projection) image-to-engram mapping')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('heteroassoc_W_IH_overlap_vs_noise.png', dpi=150)
    print('\nSaved heteroassoc_W_IH_overlap_vs_noise.png')


if __name__ == '__main__':
    main()
