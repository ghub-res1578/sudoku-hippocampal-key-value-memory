"""
Visualizes best and worst classification examples for the single-weight-matrix,
9-class digit classifier (heteroassoc_W_IH_ridge_corrected.py): trains the identical
W_IH (ridge=1.0) on the identical 500-per-class MNIST training set and 9 Sudoku-engram
class targets (same PROJ_SEED/REF_SEED, so the same pool, class-target grids, and
projection), then evaluates EVERY held-out test image (900 = 9 classes x 100) at zero
query noise, and shows the highest-margin correct classifications ("best") alongside
the highest-margin incorrect ones ("worst": confidently wrong), where margin is the
gap between the top-scoring class and the runner-up.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import H, K, N, SEED, generate_unique, relation, make_projection, make_engrams
from heteroassoc_W_IH_ridge_corrected import (
    N_CLASSES, N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, IMG_DIM, M_SUDOKU_POOL,
    PROJ_SEED, REF_SEED, RIDGE_CORRECTED,
    load_mnist_by_class, train_pseudo_inverse_primal, k_wta,
)

N_BEST = 6
N_WORST = 6
PER_CLASS = True  # if True, show one best + one worst example per digit class (9 cols) instead
                   # of a global top-6, since the global top-6 by margin is dominated by exact
                   # ties within a single class (many images collapsing to the identical top-K
                   # engram) and hides which classes actually confuse the classifier.


def main():
    rng_ref = np.random.default_rng(REF_SEED)
    rng_proj = np.random.default_rng(PROJ_SEED)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_ref = make_engrams(C_ref, W_SH, K)  # (9, H) target engram per class -- identical to ridge_corrected

    print(f'Loading MNIST: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    AllImages, AllTargets = [], []
    for d in range(1, N_CLASSES + 1):
        imgs = train_by_class[d]
        AllImages.append(imgs)
        AllTargets.append(np.tile(H_keys_ref[d - 1], (len(imgs), 1)))
    AllImages = np.concatenate(AllImages, axis=0)
    AllTargets = np.concatenate(AllTargets, axis=0)

    print(f'Training W_IH ({H} x {IMG_DIM}) via ridge regression, lambda={RIDGE_CORRECTED}...')
    W_IH = train_pseudo_inverse_primal(AllImages, AllTargets.astype(np.float32), RIDGE_CORRECTED)

    records = []
    for d in range(1, N_CLASSES + 1):
        for img in test_by_class[d]:
            h_ih = k_wta(W_IH @ img)
            scores = H_keys_ref @ h_ih.astype(np.float32) / K
            order = np.argsort(scores)[::-1]
            d_hat = int(order[0]) + 1
            margin = float(scores[order[0]] - scores[order[1]])
            records.append({
                'img': img, 'true': d, 'pred': d_hat, 'margin': margin,
                'correct': d_hat == d, 'score_true': float(scores[d - 1]),
                'score_pred': float(scores[order[0]]),
            })

    n_total = len(records)
    n_correct = sum(r['correct'] for r in records)
    print(f'\nOverall test accuracy (zero noise, exhaustive over {n_total} held-out images): '
          f'{n_correct/n_total:.2%}')

    correct_recs = sorted([r for r in records if r['correct']], key=lambda r: -r['margin'])
    wrong_recs = sorted([r for r in records if not r['correct']], key=lambda r: -r['margin'])

    if PER_CLASS:
        best, worst = [], []
        for d in range(1, N_CLASSES + 1):
            d_correct = [r for r in correct_recs if r['true'] == d]
            d_wrong = [r for r in wrong_recs if r['true'] == d]
            if d_correct:
                best.append(d_correct[0])
            if d_wrong:
                worst.append(d_wrong[0])
    else:
        best = correct_recs[:N_BEST]
        worst = wrong_recs[:N_WORST]

    print(f'Misclassified: {len(wrong_recs)}/{n_total} ({len(wrong_recs)/n_total:.1%}).')
    print('\nBest (highest-margin correct):')
    for r in best:
        print(f"  true={r['true']} pred={r['pred']} margin={r['margin']:.3f}")
    print('\nWorst (highest-margin incorrect, i.e. confidently wrong):')
    for r in worst:
        print(f"  true={r['true']} pred={r['pred']} margin={r['margin']:.3f} "
              f"score_true={r['score_true']:.3f} score_pred={r['score_pred']:.3f}")

    ncols = max(len(best), len(worst), N_BEST, N_WORST)
    fig, axes = plt.subplots(2, ncols, figsize=(2.2 * ncols, 5.2), dpi=150)
    for i, r in enumerate(best):
        ax = axes[0, i]
        ax.imshow(r['img'].reshape(28, 28), cmap='gray')
        ax.set_title(f"true {r['true']}  pred {r['pred']}\nmargin {r['margin']:.2f}",
                     fontsize=9, color='darkgreen')
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(best), ncols):
        axes[0, i].axis('off')
    for i, r in enumerate(worst):
        ax = axes[1, i]
        ax.imshow(r['img'].reshape(28, 28), cmap='gray')
        ax.set_title(f"true {r['true']}  pred {r['pred']}\nmargin {r['margin']:.2f}",
                     fontsize=9, color='crimson')
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(len(worst), ncols):
        axes[1, i].axis('off')

    fig.text(0.01, 0.72, 'BEST\n(correct)', va='center', ha='center', fontsize=11,
              weight='bold', color='darkgreen', rotation=90)
    fig.text(0.01, 0.27, 'WORST\n(confidently\nwrong)', va='center', ha='center', fontsize=11,
              weight='bold', color='crimson', rotation=90)
    per_class_note = ' (one per class)' if PER_CLASS else ''
    fig.suptitle(f'Digit classifier (single $W_{{IH}}$, 9 Sudoku-engram class targets, '
                 f'ridge={RIDGE_CORRECTED}): best vs. worst held-out classifications{per_class_note}\n'
                 f'zero noise, exhaustive over {n_total} held-out test images, '
                 f'overall accuracy {n_correct/n_total:.1%}', fontsize=10)
    fig.subplots_adjust(left=0.07, top=0.82, hspace=0.5)
    fig.savefig('heteroassoc_digit_classifier_best_worst.png', dpi=150, bbox_inches='tight')
    print('\nSaved heteroassoc_digit_classifier_best_worst.png')


if __name__ == '__main__':
    main()
