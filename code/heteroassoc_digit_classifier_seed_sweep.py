"""
Checks whether the "confidently-wrong errors funnel into a small number of attractor
classes" pattern observed for the single-W_IH, 9-class digit classifier
(heteroassoc_digit_classifier_best_worst.py, seed REF_SEED=SEED+41/PROJ_SEED=SEED+40,
where 5/9 worst-case errors predicted class 8 or 9) is a general property of this
architecture or an artifact of that one draw of the Sudoku-engram class targets and the
random projection W_SH.

Reruns the identical training/eval pipeline (same ridge=1.0, same 500-train/100-test
MNIST split, loaded once) across N_SEEDS independent draws of (REF_SEED, PROJ_SEED) --
seed index 0 reproduces the original run exactly -- and reports, per seed: overall
accuracy, the full 9x9 confusion matrix, and the share of all errors absorbed by the
single most-attractive predicted class.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import H, K, N, SEED, generate_unique, relation, make_projection, make_engrams
from heteroassoc_W_IH_ridge_corrected import (
    N_CLASSES, N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, IMG_DIM, M_SUDOKU_POOL,
    RIDGE_CORRECTED, load_mnist_by_class, train_pseudo_inverse_primal, k_wta,
)

N_SEEDS = 6


def run_one_seed(seed_idx, train_by_class, test_by_class):
    ref_seed = SEED + 41 + 1000 * seed_idx
    proj_seed = SEED + 40 + 1000 * seed_idx

    rng_ref = np.random.default_rng(ref_seed)
    rng_proj = np.random.default_rng(proj_seed)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_ref = make_engrams(C_ref, W_SH, K)

    AllImages, AllTargets = [], []
    for d in range(1, N_CLASSES + 1):
        imgs = train_by_class[d]
        AllImages.append(imgs)
        AllTargets.append(np.tile(H_keys_ref[d - 1], (len(imgs), 1)))
    AllImages = np.concatenate(AllImages, axis=0)
    AllTargets = np.concatenate(AllTargets, axis=0)
    W_IH = train_pseudo_inverse_primal(AllImages, AllTargets.astype(np.float32), RIDGE_CORRECTED)

    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)  # [true-1, pred-1]
    for d in range(1, N_CLASSES + 1):
        for img in test_by_class[d]:
            h_ih = k_wta(W_IH @ img)
            scores = H_keys_ref @ h_ih.astype(np.float32) / K
            d_hat = int(np.argmax(scores)) + 1
            confusion[d - 1, d_hat - 1] += 1

    n_total = confusion.sum()
    n_correct = np.trace(confusion)
    accuracy = n_correct / n_total

    off_diag = confusion.copy()
    np.fill_diagonal(off_diag, 0)
    errors_per_pred_class = off_diag.sum(axis=0)  # errors routed TO each predicted class
    n_errors = off_diag.sum()
    top_class = int(np.argmax(errors_per_pred_class)) + 1
    top_share = errors_per_pred_class[top_class - 1] / n_errors if n_errors else 0.0
    top2_idx = np.argsort(errors_per_pred_class)[::-1][:2]
    top2_share = errors_per_pred_class[top2_idx].sum() / n_errors if n_errors else 0.0
    top2_classes = tuple(int(i) + 1 for i in top2_idx)

    return {
        'seed_idx': seed_idx, 'ref_seed': ref_seed, 'proj_seed': proj_seed,
        'accuracy': accuracy, 'n_errors': int(n_errors),
        'top_attractor_class': top_class, 'top_attractor_share': top_share,
        'top2_attractor_classes': top2_classes, 'top2_attractor_share': top2_share,
        'confusion': confusion,
    }


def main():
    print(f'Loading MNIST once: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    results = []
    for s in range(N_SEEDS):
        print(f'\n--- Seed index {s} (ref_seed={SEED+41+1000*s}, proj_seed={SEED+40+1000*s}) ---')
        r = run_one_seed(s, train_by_class, test_by_class)
        results.append(r)
        print(f"  accuracy={r['accuracy']:.2%}  n_errors={r['n_errors']}  "
              f"top attractor class={r['top_attractor_class']} "
              f"({r['top_attractor_share']:.1%} of all errors)  "
              f"top-2 attractor classes={r['top2_attractor_classes']} "
              f"({r['top2_attractor_share']:.1%} of all errors)")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'confusion'} for r in results])
    df.to_csv('heteroassoc_digit_classifier_seed_sweep.csv', index=False)

    print('\n=== SUMMARY ACROSS SEEDS ===')
    print(df[['seed_idx', 'accuracy', 'n_errors', 'top_attractor_class',
               'top_attractor_share', 'top2_attractor_classes', 'top2_attractor_share']]
          .to_string(index=False))
    print(f"\nMean accuracy: {df['accuracy'].mean():.2%} (range {df['accuracy'].min():.2%}"
          f"-{df['accuracy'].max():.2%})")
    print(f"Mean top-1 attractor share: {df['top_attractor_share'].mean():.1%} "
          f"(chance if errors were uniform across the 8 wrong classes: {1/8:.1%})")
    print(f"Mean top-2 attractor share: {df['top2_attractor_share'].mean():.1%} "
          f"(chance: {2/8:.1%})")
    print(f"Distinct top-1 attractor classes seen across seeds: "
          f"{sorted(df['top_attractor_class'].unique().tolist())}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=150)
    for i, r in enumerate(results):
        ax = axes[i // 3, i % 3]
        cm = r['confusion'].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / row_sums
        im = ax.imshow(cm_norm, cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(range(1, N_CLASSES + 1))
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(range(1, N_CLASSES + 1))
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        ax.set_title(f"seed {i}: acc={r['accuracy']:.1%}\n"
                      f"top attractor: class {r['top_attractor_class']} "
                      f"({r['top_attractor_share']:.0%} of errors)", fontsize=9)
    fig.suptitle('Confusion matrices (row-normalized) across 6 independent seeds\n'
                  '(same architecture, ridge=1.0, different W_SH + class-target draws)',
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig('heteroassoc_digit_classifier_seed_sweep.png', dpi=150, bbox_inches='tight')
    print('\nSaved heteroassoc_digit_classifier_seed_sweep.csv and .png')


if __name__ == '__main__':
    main()
