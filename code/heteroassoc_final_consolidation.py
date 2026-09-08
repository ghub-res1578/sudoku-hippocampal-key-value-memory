"""
Consolidates the heteroassociation results against three open gaps, plus a benchmark
against Vector-HaSH's actual mechanism:

  (A) STATISTICAL RIGOR: re-runs the noise sweep (MNIST/Fashion-MNIST/CIFAR-100-gray,
      M=200) at N=200 trials/sigma (up from 40-60), reporting Wilson 95% CIs on the
      binary exact-recovery rate and bootstrap 95% CIs on the continuous cleaned-overlap.

  (B) BASELINE: for the SAME noisy queries (paired), adds a trivial raw-pixel
      nearest-neighbor classifier among the M stored clean images, and runs a paired
      McNemar exact test against our pipeline's exact-recovery outcome per noise level
      -- directly answering "is this just k-NN into M buckets?"

  (C) CAPACITY CLIFF DIAGNOSIS: a fine-grained M sweep (200/300/400/500/650/800) at
      fixed low noise, PLUS the actual linear-algebra explanation: the Gram-matrix
      condition number and solution weight-norm as a function of M (does conditioning
      degrade well before M approaches the ambient dimension?), PLUS a ridge-lambda
      sensitivity check at the cliff M (does more regularization trade zero-noise
      perfection for noise robustness, confirming the conditioning diagnosis causally?).

  (D) VECTOR-HASH BENCHMARK: the fine-grained M-sweep at fixed noise directly mirrors
      Vector-HaSH's own capacity-vs-recall diagnostic (their Fig. 3d/f: MI or exact
      recall vs. number of stored patterns, at fixed sensory noise ~2.5%). Their paper's
      central claim is that Vector-HaSH "avoids the memory cliff of prior memory models
      ... and instead exhibits a graceful trade-off between number of stored items and
      recall detail" (abstract) -- because the scaffold (grid<->hippocampus) is FIXED,
      content-independent, and provably gives exponentially many EQUALLY LARGE basins
      (their Fig. 2), so the heteroassociative pathway only needs to get "close enough."
      Our system has no such independent, content-independent scaffold: the Sudoku
      attractor's basins are legality-dependent, and the SAME pseudo-inverse-fitted
      pathway that Vector-HaSH treats as merely "close enough" is, in our case, the only
      thing standing between a noisy image and the engram. This script quantifies
      whether/where that difference produces the classic cliff Vector-HaSH is explicitly
      designed to avoid.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import binomtest
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

N_TRIALS_RIGOROUS = 200
RIGOR_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
RIGOR_CONFIGS = [('mnist', 'MNIST'), ('fashion_mnist', 'Fashion-MNIST'), ('cifar100_gray', 'CIFAR-100 (gray)')]

CAPACITY_M_VALUES = [200, 300, 400, 500, 650, 800]
CAPACITY_SIGMAS = [0.0, 0.1]
N_TRIALS_CAPACITY = 60

RIDGE_M = 500
RIDGE_LAMBDAS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RIDGE_SIGMA = 0.1
N_TRIALS_RIDGE = 60


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - halfwidth, center + halfwidth)


def bootstrap_ci(values, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    if len(values) == 0:
        return (np.nan, np.nan)
    boots = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (np.percentile(boots, 2.5), np.percentile(boots, 97.5))


def build_store(dataset_name, M_local, ridge_lambda=RIDGE_LAMBDA):
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)
    B_drive = drive_vectors(pool)
    conf = conflict_matrix()
    images, img_dim = load_dataset(dataset_name, M_local)
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), ridge_lambda)
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images, ridge_lambda)
    return dict(pool=pool, W_SH=W_SH, H_keys=H_keys, B_drive=B_drive, conf=conf,
                images=images, img_dim=img_dim, W_SI=W_SI, W_IS=W_IS)


# ---------------------------------------------------------------------------
# (A) + (B): rigorous noise sweep with CIs, paired against a raw-pixel k-NN baseline
# ---------------------------------------------------------------------------
def run_rigorous_with_baseline(dataset_name, label, M_local=200):
    print(f'\n{"=" * 78}\nRIGOROUS SWEEP + k-NN BASELINE: {label} (M={M_local}, N={N_TRIALS_RIGOROUS}/sigma)\n{"=" * 78}')
    t0 = time.time()
    store = build_store(dataset_name, M_local)
    H_keys, B_drive, W_SH, conf = store['H_keys'], store['B_drive'], store['W_SH'], store['conf']
    images, W_SI, W_IS = store['images'], store['W_SI'], store['W_IS']
    digits_true_all = store['pool'].reshape(M_local, N)
    print(f'  store built [{time.time()-t0:.1f}s]')

    rng_test = np.random.default_rng(SEED + 999)
    rows = []
    for sigma in RIGOR_SIGMAS:
        for _ in range(N_TRIALS_RIGOROUS):
            m = rng_test.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]

            # our pipeline
            h_initial = k_wta(W_SI @ x_noisy)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / K
            exact = int(np.all(digits_final == digits_true_all[m]))

            # raw-pixel nearest-neighbor baseline (paired: same x_noisy)
            dists = np.sum((images - x_noisy[None, :]) ** 2, axis=1)
            nn_idx = np.argmin(dists)
            knn_exact = int(nn_idx == m)

            rows.append({'dataset': label, 'sigma': sigma, 'target': m,
                         'cleaned_overlap': cleaned_overlap, 'exact': exact, 'knn_exact': knn_exact})
        sub = [r for r in rows if r['sigma'] == sigma]
        ours = [r['exact'] for r in sub]
        knn = [r['knn_exact'] for r in sub]
        b = sum(1 for r in sub if r['exact'] == 1 and r['knn_exact'] == 0)
        c = sum(1 for r in sub if r['exact'] == 0 and r['knn_exact'] == 1)
        pval = binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
        ov_lo, ov_hi = bootstrap_ci([r['cleaned_overlap'] for r in sub])
        ex_lo, ex_hi = wilson_ci(sum(ours), len(ours))
        knn_lo, knn_hi = wilson_ci(sum(knn), len(knn))
        print(f'  sigma={sigma:.2f}  ours_exact={np.mean(ours):.3f} [{ex_lo:.3f},{ex_hi:.3f}]  '
              f'knn_exact={np.mean(knn):.3f} [{knn_lo:.3f},{knn_hi:.3f}]  '
              f'cleaned_ov={np.mean([r["cleaned_overlap"] for r in sub]):.3f} [{ov_lo:.3f},{ov_hi:.3f}]  '
              f'McNemar(ours vs knn) p={pval:.4f}  [{time.time()-t0:.1f}s]')

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# (C) + (D): fine-grained capacity sweep + conditioning diagnostics + ridge sensitivity
# ---------------------------------------------------------------------------
def run_capacity_and_conditioning():
    print(f'\n{"=" * 78}\nCAPACITY CLIFF: fine-grained M sweep + conditioning diagnostics (MNIST)\n{"=" * 78}')
    cap_rows, cond_rows = [], []
    for M_local in CAPACITY_M_VALUES:
        t0 = time.time()
        store = build_store('mnist', M_local)
        images, H_keys, W_SI = store['images'], store['H_keys'], store['W_SI']

        Xm = images.astype(np.float64)
        Gram = Xm @ Xm.T
        eigvals = np.linalg.eigvalsh(Gram)
        eigvals_clipped = np.clip(eigvals, 1e-12, None)
        cond_number = eigvals_clipped[-1] / eigvals_clipped[0]
        w_si_norm = np.linalg.norm(W_SI)
        cond_rows.append({'M': M_local, 'gram_cond_number': cond_number, 'W_SI_frobenius_norm': w_si_norm,
                           'smallest_eigval': eigvals_clipped[0], 'largest_eigval': eigvals_clipped[-1]})
        print(f'  M={M_local}: Gram cond#={cond_number:.3e}  smallest_eig={eigvals_clipped[0]:.3e}  '
              f'||W_SI||_F={w_si_norm:.1f}  [{time.time()-t0:.1f}s]')

        B_drive, W_SH, conf = store['B_drive'], store['W_SH'], store['conf']
        W_IS = store['W_IS']
        digits_true_all = store['pool'].reshape(M_local, N)
        rng_test = np.random.default_rng(SEED + 999)
        for sigma in CAPACITY_SIGMAS:
            trial_rows = []
            for _ in range(N_TRIALS_CAPACITY):
                m = rng_test.integers(M_local)
                x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]
                h_initial = k_wta(W_SI @ x_noisy)
                h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                    H_keys, B_drive, W_SH, conf, h_initial)
                cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / K
                exact = int(np.all(digits_final == digits_true_all[m]))
                trial_rows.append((cleaned_overlap, exact))
            ov_mean = np.mean([t[0] for t in trial_rows])
            ex_mean = np.mean([t[1] for t in trial_rows])
            cap_rows.append({'M': M_local, 'sigma': sigma, 'cleaned_overlap': ov_mean, 'exact': ex_mean})
            print(f'    sigma={sigma:.2f}  cleaned_ov={ov_mean:.3f}  exact={ex_mean:.2%}  [{time.time()-t0:.1f}s]')

    return pd.DataFrame(cap_rows), pd.DataFrame(cond_rows)


def run_ridge_sensitivity():
    print(f'\n{"=" * 78}\nRIDGE-LAMBDA SENSITIVITY at M={RIDGE_M} (does more regularization trade\n'
          f'zero-noise perfection for noise robustness? -- causal test of the conditioning hypothesis)\n{"=" * 78}')
    rows = []
    for lam in RIDGE_LAMBDAS:
        t0 = time.time()
        store = build_store('mnist', RIDGE_M, ridge_lambda=lam)
        images, H_keys, W_SI, W_IS = store['images'], store['H_keys'], store['W_SI'], store['W_IS']
        B_drive, W_SH, conf = store['B_drive'], store['W_SH'], store['conf']
        digits_true_all = store['pool'].reshape(RIDGE_M, N)

        h_zero_pred = np.array([k_wta(W_SI @ images[m]) for m in range(RIDGE_M)])
        zero_noise_overlap = np.mean([np.sum(h_zero_pred[m] * H_keys[m]) / K for m in range(RIDGE_M)])

        rng_test = np.random.default_rng(SEED + 999)
        trial_rows = []
        for _ in range(N_TRIALS_RIDGE):
            m = rng_test.integers(RIDGE_M)
            x_noisy = add_noise(images[m:m + 1], RIDGE_SIGMA, rng_test)[0]
            h_initial = k_wta(W_SI @ x_noisy)
            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / K
            exact = int(np.all(digits_final == digits_true_all[m]))
            trial_rows.append((cleaned_overlap, exact))
        noisy_ov = np.mean([t[0] for t in trial_rows])
        noisy_exact = np.mean([t[1] for t in trial_rows])
        rows.append({'ridge_lambda': lam, 'zero_noise_overlap': zero_noise_overlap,
                     f'sigma_{RIDGE_SIGMA}_cleaned_overlap': noisy_ov, f'sigma_{RIDGE_SIGMA}_exact': noisy_exact})
        print(f'  lambda={lam:<8g}  zero_noise_ov={zero_noise_overlap:.3f}  '
              f'sigma={RIDGE_SIGMA}_ov={noisy_ov:.3f}  sigma={RIDGE_SIGMA}_exact={noisy_exact:.2%}  '
              f'[{time.time()-t0:.1f}s]')
    return pd.DataFrame(rows)


def main():
    rigor_dfs = []
    for dataset_name, label in RIGOR_CONFIGS:
        rigor_dfs.append(run_rigorous_with_baseline(dataset_name, label))
    rigor_df = pd.concat(rigor_dfs, ignore_index=True)
    rigor_df.to_csv('heteroassoc_final_rigor_baseline.csv', index=False)

    cap_df, cond_df = run_capacity_and_conditioning()
    cap_df.to_csv('heteroassoc_final_capacity_finegrained.csv', index=False)
    cond_df.to_csv('heteroassoc_final_conditioning.csv', index=False)

    ridge_df = run_ridge_sensitivity()
    ridge_df.to_csv('heteroassoc_final_ridge_sensitivity.csv', index=False)

    # ---------------- summary plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    for label in [c[1] for c in RIGOR_CONFIGS]:
        sub = rigor_df[rigor_df.dataset == label].groupby('sigma').agg(
            exact=('exact', 'mean'), knn_exact=('knn_exact', 'mean')).reset_index()
        axes[0, 0].plot(sub.sigma, sub.exact, 'o-', label=f'{label} (ours)')
        axes[0, 0].plot(sub.sigma, sub.knn_exact, 's--', alpha=0.5, label=f'{label} (k-NN baseline)')
    axes[0, 0].set_xlabel('Image noise (sigma)'); axes[0, 0].set_ylabel('Exact recovery rate')
    axes[0, 0].set_title('Our pipeline vs. raw-pixel k-NN baseline (paired)')
    axes[0, 0].legend(fontsize=6); axes[0, 0].grid(alpha=0.3)

    cap_pivot = cap_df.pivot(index='M', columns='sigma', values='exact')
    for sigma in CAPACITY_SIGMAS:
        axes[0, 1].plot(cap_pivot.index, cap_pivot[sigma], 'o-', label=f'sigma={sigma}')
    axes[0, 1].set_xlabel('M (number of stored image-grid pairs)'); axes[0, 1].set_ylabel('Exact recovery rate')
    axes[0, 1].set_title('Capacity cliff: exact recovery vs. M (fixed noise)')
    axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(cond_df.M, cond_df.gram_cond_number, 'o-', color='tab:red')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('M'); axes[1, 0].set_ylabel('Gram matrix condition number (log scale)')
    axes[1, 0].set_title('Conditioning of the pseudo-inverse solve vs. M')
    axes[1, 0].grid(alpha=0.3)

    ax2 = axes[1, 1]
    ax2.plot(ridge_df.ridge_lambda, ridge_df.zero_noise_overlap, 'o-', color='tab:blue', label='zero-noise overlap')
    ax2.plot(ridge_df.ridge_lambda, ridge_df[f'sigma_{RIDGE_SIGMA}_exact'], 's-', color='tab:orange',
             label=f'sigma={RIDGE_SIGMA} exact recovery')
    ax2.set_xscale('log')
    ax2.set_xlabel('Ridge lambda'); ax2.set_ylabel('Rate / overlap')
    ax2.set_title(f'Ridge regularization trade-off at M={RIDGE_M}')
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig('heteroassoc_final_consolidation.png', dpi=150)
    print('\nSaved heteroassoc_final_consolidation.png')


if __name__ == '__main__':
    main()
