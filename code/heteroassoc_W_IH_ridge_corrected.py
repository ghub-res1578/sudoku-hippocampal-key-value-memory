"""
Minimal-diff replay of heteroassoc_W_IH_overlap_vs_noise.py -- identical methodology
(same primal-form solver, same reference-grid selection via pool[ref_idx], same seeds,
same image loading) -- with exactly ONE change: ridge_lambda=1.0 instead of the
original's default (1e-4, imported unchanged from hippocampus_drive_readout). Also adds
a direct classification-accuracy measurement (argmax over the 9 classes' overlap
scores), which the original script never computed -- it only reported raw overlap with
the true target, not a decoded class decision.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import (
    H, K, N, SEED, SEED_TRIALS, generate_unique, relation, make_projection, make_engrams,
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
RIDGE_CORRECTED = 1.0  # the ONLY thing changed vs. the original script


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


def train_pseudo_inverse_primal(X, T, ridge_lambda):
    Xm = X.astype(np.float64); Tm = T.astype(np.float64)
    D = Xm.shape[1]
    A = Xm.T @ Xm + ridge_lambda * np.eye(D)
    W = Tm.T @ Xm @ np.linalg.inv(A)
    return W.astype(np.float32)


def k_wta(a, k=K):
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


def main():
    rng_ref = np.random.default_rng(REF_SEED)
    rng_proj = np.random.default_rng(PROJ_SEED)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_ref = make_engrams(C_ref, W_SH, K)  # (9, H) target engram per class -- IDENTICAL to original

    print(f'Loading MNIST: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    AllImages, AllTargets = [], []
    for d in range(1, N_CLASSES + 1):
        imgs = train_by_class[d]
        AllImages.append(imgs)
        AllTargets.append(np.tile(H_keys_ref[d - 1], (len(imgs), 1)))
    AllImages = np.concatenate(AllImages, axis=0)
    AllTargets = np.concatenate(AllTargets, axis=0)

    print(f'Training W_IH ({H} x {IMG_DIM}) via ridge regression, '
          f'lambda={RIDGE_CORRECTED} (original used the tiny 1e-4 default)...')
    W_IH = train_pseudo_inverse_primal(AllImages, AllTargets.astype(np.float32), RIDGE_CORRECTED)

    trng = np.random.default_rng(SEED_TRIALS + 9)  # IDENTICAL trial seed to the original
    rows = []
    for sigma in NOISE_SIGMAS:
        overlaps, correct_ct = [], 0
        for _ in range(N_TRIALS):
            d = trng.integers(1, N_CLASSES + 1)
            test_idx = trng.integers(len(test_by_class[d]))
            img = test_by_class[d][test_idx]
            x_noisy = np.clip(img + trng.normal(0, sigma, IMG_DIM), 0, 1).astype(np.float32)

            h_ih = k_wta(W_IH @ x_noisy)
            scores = H_keys_ref @ h_ih.astype(np.float32) / K  # (9,) overlap with each class target
            overlap_true = scores[d - 1]
            d_hat = int(np.argmax(scores)) + 1

            overlaps.append(overlap_true)
            correct_ct += int(d_hat == d)

        rows.append({'sigma': sigma, 'overlap_W_IH_vs_target': np.mean(overlaps),
                     'classification_accuracy': correct_ct / N_TRIALS})
        print(f'  sigma={sigma:4.2f}  overlap={np.mean(overlaps):.3f}  '
              f'classification_accuracy={correct_ct/N_TRIALS:.2%}')

    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_W_IH_ridge_corrected.csv', index=False)
    print('\nSaved heteroassoc_W_IH_ridge_corrected.csv')

    print('\n=== COMPARISON TO ORIGINAL (ridge=1e-4, from heteroassoc_W_IH_overlap_vs_noise.csv) ===')
    print('Original: sigma=0.00 overlap=0.835  sigma=0.10 overlap=0.330  (reported cliff)')
    print(f'Corrected (ridge={RIDGE_CORRECTED}): sigma=0.00 overlap={rows[0]["overlap_W_IH_vs_target"]:.3f}  '
          f'sigma=0.10 overlap={rows[1]["overlap_W_IH_vs_target"]:.3f}')


if __name__ == '__main__':
    main()
