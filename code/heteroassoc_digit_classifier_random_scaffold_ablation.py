"""
Isolates how much of the digit classifier's accuracy comes from the Sudoku scaffold
(W_SH + k-WTA + real Sudoku-grid class targets) versus the ridge-regression map W_IH
itself. The classifier pipeline (heteroassoc_W_IH_ridge_corrected.py) has NO cleanup/
attractor pass at all -- unlike the main image-reconstruction pipeline (Section 4.1), it
is a single linear regression (image -> engram) followed by k-WTA and nearest-target
overlap, with no iterative Sudoku-legality correction. So the only way the "scaffold"
could matter here is through the specific geometry of the 9 Sudoku-derived target
engrams themselves.

This ablation replaces those 9 targets with 9 independently drawn RANDOM K-sparse binary
vectors (same H=1000, K=150, no Sudoku structure whatsoever) and retrains the identical
ridge regression (lambda=1.0) on the identical MNIST split. If accuracy is statistically
indistinguishable from the real-scaffold result (78.8%), the scaffold is contributing
nothing beyond being an arbitrary source of 9 fixed, well-separated sparse binary labels,
and the classification work is being done entirely by the ridge regression + k-WTA
nonlinearity -- not by anything Sudoku-specific.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import H, K, N, SEED, generate_unique, relation, make_projection, make_engrams
from heteroassoc_W_IH_ridge_corrected import (
    N_CLASSES, N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, IMG_DIM, M_SUDOKU_POOL,
    PROJ_SEED, REF_SEED, RIDGE_CORRECTED,
    load_mnist_by_class, train_pseudo_inverse_primal, k_wta,
)

RANDOM_TARGET_SEED = SEED + 999


def k_sparse_random_targets(n_classes, h, k, rng):
    targets = np.zeros((n_classes, h), dtype=np.float32)
    for c in range(n_classes):
        idx = rng.choice(h, size=k, replace=False)
        targets[c, idx] = 1
    return targets


def evaluate(H_keys_ref, W_IH, test_by_class):
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for d in range(1, N_CLASSES + 1):
        for img in test_by_class[d]:
            h_ih = k_wta(W_IH @ img)
            scores = H_keys_ref @ h_ih.astype(np.float32) / K
            d_hat = int(np.argmax(scores)) + 1
            confusion[d - 1, d_hat - 1] += 1
    return confusion


def main():
    print(f'Loading MNIST: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    # --- Real Sudoku scaffold (reference, identical to heteroassoc_W_IH_ridge_corrected.py) ---
    rng_ref = np.random.default_rng(REF_SEED)
    rng_proj = np.random.default_rng(PROJ_SEED)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_sudoku = make_engrams(C_ref, W_SH, K)

    # --- Random scaffold-free targets: same H, K, no Sudoku structure ---
    rng_rand = np.random.default_rng(RANDOM_TARGET_SEED)
    H_keys_random = k_sparse_random_targets(N_CLASSES, H, K, rng_rand)

    results = {}
    for name, H_keys_ref in [('Sudoku scaffold (real)', H_keys_sudoku),
                              ('Random targets (no scaffold)', H_keys_random)]:
        AllImages, AllTargets = [], []
        for d in range(1, N_CLASSES + 1):
            imgs = train_by_class[d]
            AllImages.append(imgs)
            AllTargets.append(np.tile(H_keys_ref[d - 1], (len(imgs), 1)))
        AllImages = np.concatenate(AllImages, axis=0)
        AllTargets = np.concatenate(AllTargets, axis=0)

        print(f'\nTraining W_IH for "{name}" (ridge={RIDGE_CORRECTED})...')
        W_IH = train_pseudo_inverse_primal(AllImages, AllTargets.astype(np.float32), RIDGE_CORRECTED)

        confusion = evaluate(H_keys_ref, W_IH, test_by_class)
        acc = np.trace(confusion) / confusion.sum()
        print(f'  {name}: accuracy = {acc:.2%} (exhaustive over {confusion.sum()} test images)')
        results[name] = {'accuracy': acc, 'confusion': confusion}

    # McNemar-style paired comparison isn't quite right here since the two conditions use
    # different targets/predictions on the SAME images -- report a simple two-proportion
    # z-test instead, on independent-enough grounds (same images, different classifiers).
    from scipy.stats import norm
    n = 900
    p1 = results['Sudoku scaffold (real)']['accuracy']
    p2 = results['Random targets (no scaffold)']['accuracy']
    p_pool = (p1 * n + p2 * n) / (2 * n)
    se = np.sqrt(2 * p_pool * (1 - p_pool) / n)
    z = (p1 - p2) / se if se > 0 else 0.0
    p_value = 2 * (1 - norm.cdf(abs(z)))

    print(f'\n=== COMPARISON ===')
    print(f'Sudoku scaffold accuracy:      {p1:.2%}')
    print(f'Random-target accuracy:        {p2:.2%}')
    print(f'Difference:                    {(p1-p2)*100:+.1f}pp')
    print(f'Two-proportion z-test: z={z:.2f}, p={p_value:.3f}')

    df = pd.DataFrame([
        {'condition': k, 'accuracy': v['accuracy']} for k, v in results.items()
    ])
    df.to_csv('heteroassoc_digit_classifier_random_scaffold_ablation.csv', index=False)
    print('\nSaved heteroassoc_digit_classifier_random_scaffold_ablation.csv')


if __name__ == '__main__':
    main()
