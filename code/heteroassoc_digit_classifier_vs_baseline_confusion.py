"""
Checks whether the single-W_IH digit classifier's confusion matrix (errors funneling
disproportionately into one "attractor" class, e.g. class 9, largely independent of the
true digit's visual similarity to it) resembles the confusion matrix shape of genuine,
well-established MNIST classifiers, or is instead specific to this architecture.

Trains two standard reference classifiers -- k-NN (k=5, raw pixels) and multinomial
logistic regression (raw pixels) -- on the IDENTICAL 500-train/100-test-per-class MNIST
split (digits 1-9) used throughout this project's heteroassociation experiments, computes
their confusion matrices, and compares:
  (a) whether either baseline shows a single dominant "attractor" predicted class the way
      ours does (top-1/top-2 share of all errors);
  (b) whether baselines' errors are pairwise-symmetric and concentrated in classically
      visually-similar digit pairs (4<->9, 3<->5, 3<->8, 5<->6, 7<->9), unlike a one-way
      many-to-one funnel;
  (c) the direct correlation between our off-diagonal error distribution and each
      baseline's.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import H, K, N, SEED, generate_unique, relation, make_projection, make_engrams
from heteroassoc_W_IH_ridge_corrected import (
    N_CLASSES, N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, IMG_DIM, M_SUDOKU_POOL,
    PROJ_SEED, REF_SEED, RIDGE_CORRECTED,
    load_mnist_by_class, train_pseudo_inverse_primal, k_wta,
)


def attractor_stats(confusion):
    off_diag = confusion.copy().astype(float)
    np.fill_diagonal(off_diag, 0)
    n_errors = off_diag.sum()
    errors_per_pred = off_diag.sum(axis=0)
    top_idx = np.argsort(errors_per_pred)[::-1][:2]
    top1_share = errors_per_pred[top_idx[0]] / n_errors
    top2_share = errors_per_pred[top_idx].sum() / n_errors
    return {
        'n_errors': int(n_errors),
        'top1_class': int(top_idx[0]) + 1, 'top1_share': top1_share,
        'top2_classes': (int(top_idx[0]) + 1, int(top_idx[1]) + 1), 'top2_share': top2_share,
    }


def symmetry_score(confusion):
    """Correlation between confusion[i,j] and confusion[j,i] for i != j -- high for
    classic visually-similar-pair confusions (e.g. 4->9 and 9->4 both elevated), low/zero
    for a one-way funnel (many rows -> one column, but not reciprocated)."""
    off_diag = confusion.copy().astype(float)
    np.fill_diagonal(off_diag, 0)
    a = off_diag[np.triu_indices(len(confusion), k=1)]
    b = off_diag.T[np.triu_indices(len(confusion), k=1)]
    if a.std() == 0 or b.std() == 0:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print(f'Loading MNIST: {N_PER_CLASS_TRAIN} train / {N_PER_CLASS_TEST} test exemplars per class...')
    train_by_class, test_by_class = load_mnist_by_class(N_PER_CLASS_TRAIN, N_PER_CLASS_TEST)

    X_train, y_train, X_test, y_test = [], [], [], []
    for d in range(1, N_CLASSES + 1):
        X_train.append(train_by_class[d]); y_train += [d] * len(train_by_class[d])
        X_test.append(test_by_class[d]); y_test += [d] * len(test_by_class[d])
    X_train = np.concatenate(X_train, axis=0); y_train = np.array(y_train)
    X_test = np.concatenate(X_test, axis=0); y_test = np.array(y_test)

    # --- Our architecture (seed 0, identical to heteroassoc_W_IH_ridge_corrected.py) ---
    rng_ref = np.random.default_rng(REF_SEED)
    rng_proj = np.random.default_rng(PROJ_SEED)
    pool = generate_unique(M_SUDOKU_POOL, rng_ref)
    ref_idx = rng_ref.choice(M_SUDOKU_POOL, size=N_CLASSES, replace=False)
    ref_sudokus = pool[ref_idx]
    W_SH = make_projection(rng_proj)
    C_ref = np.asarray([relation(g).ravel() for g in ref_sudokus], dtype=np.float32)
    H_keys_ref = make_engrams(C_ref, W_SH, K)

    AllTargets = np.concatenate(
        [np.tile(H_keys_ref[d - 1], (len(train_by_class[d]), 1)) for d in range(1, N_CLASSES + 1)], axis=0)
    W_IH = train_pseudo_inverse_primal(X_train, AllTargets.astype(np.float32), RIDGE_CORRECTED)

    y_pred_ours = []
    for img in X_test:
        h_ih = k_wta(W_IH @ img)
        scores = H_keys_ref @ h_ih.astype(np.float32) / K
        y_pred_ours.append(int(np.argmax(scores)) + 1)
    y_pred_ours = np.array(y_pred_ours)
    cm_ours = confusion_matrix(y_test, y_pred_ours, labels=list(range(1, N_CLASSES + 1)))

    # --- Baseline 1: k-NN (k=5), raw pixels ---
    print('Training k-NN (k=5) on raw pixels...')
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    cm_knn = confusion_matrix(y_test, y_pred_knn, labels=list(range(1, N_CLASSES + 1)))

    # --- Baseline 2: multinomial logistic regression, raw pixels ---
    print('Training multinomial logistic regression on raw pixels...')
    logreg = LogisticRegression(max_iter=2000, multi_class='multinomial')
    logreg.fit(X_train, y_train)
    y_pred_lr = logreg.predict(X_test)
    cm_lr = confusion_matrix(y_test, y_pred_lr, labels=list(range(1, N_CLASSES + 1)))

    classifiers = {'ours (single W_IH)': cm_ours, 'k-NN (k=5)': cm_knn, 'logistic regression': cm_lr}

    print('\n=== Per-classifier accuracy and attractor structure ===')
    rows = []
    for name, cm in classifiers.items():
        acc = np.trace(cm) / cm.sum()
        stats = attractor_stats(cm)
        sym = symmetry_score(cm)
        print(f"{name:22s} acc={acc:.2%}  top1={stats['top1_class']} ({stats['top1_share']:.1%})  "
              f"top2={stats['top2_classes']} ({stats['top2_share']:.1%})  symmetry_r={sym:.3f}")
        rows.append({'classifier': name, 'accuracy': acc, **stats, 'symmetry_r': sym})
    df = pd.DataFrame(rows)
    df.to_csv('heteroassoc_digit_classifier_vs_baseline_confusion.csv', index=False)

    # --- Direct correlation between off-diagonal error distributions ---
    def off_diag_flat(cm):
        m = cm.copy().astype(float)
        np.fill_diagonal(m, 0)
        m = m / m.sum()
        return m.flatten()

    flat_ours = off_diag_flat(cm_ours)
    flat_knn = off_diag_flat(cm_knn)
    flat_lr = off_diag_flat(cm_lr)
    r_ours_knn = np.corrcoef(flat_ours, flat_knn)[0, 1]
    r_ours_lr = np.corrcoef(flat_ours, flat_lr)[0, 1]
    r_knn_lr = np.corrcoef(flat_knn, flat_lr)[0, 1]
    print(f'\nCorrelation of off-diagonal error distributions (72 cells each):')
    print(f'  ours vs. k-NN:              r={r_ours_knn:.3f}')
    print(f'  ours vs. logistic regression: r={r_ours_lr:.3f}')
    print(f'  k-NN vs. logistic regression: r={r_knn_lr:.3f}  (reference: two independent, '
          f'genuine classifiers on the same data)')

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), dpi=150)
    for ax, (name, cm) in zip(axes, classifiers.items()):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        im = ax.imshow(cm_norm, cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(range(1, N_CLASSES + 1))
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(range(1, N_CLASSES + 1))
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f'{name}\nacc={acc:.1%}', fontsize=11)
    fig.suptitle('Confusion matrices (row-normalized): our architecture vs. two standard '
                  'MNIST baselines\n(identical 500-train/100-test-per-class split, digits 1-9)',
                  fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig('heteroassoc_digit_classifier_vs_baseline_confusion.png', dpi=150, bbox_inches='tight')
    print('\nSaved heteroassoc_digit_classifier_vs_baseline_confusion.csv and .png')


if __name__ == '__main__':
    main()
