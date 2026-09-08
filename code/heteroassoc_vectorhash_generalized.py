"""
Generalizes the Vector-HaSH-style pipeline (heteroassoc_vectorhash_pipeline.py) to:
  - larger M (test the pseudo-inverse heteroassociation's CAPACITY ceiling, mirroring the
    paper's own M~H collapse finding for ridge-regression readouts elsewhere in this project)
  - different image dimensions and datasets (MNIST 28x28=784; Fashion-MNIST 28x28=784, same
    dims/different content; CIFAR-100 32x32=1024 grayscale, different dims AND content)

Same four-stage pipeline throughout: image -> Hebbian/pseudo-inverse heteroassociation
(one memory per pair) -> engram -> attention readout -> drives -> coincidence-matrix cleanup
(Sudoku attractor) -> cleaned engram -> Hebbian/pseudo-inverse decode -> reconstructed image.
"""
import time
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

MAX_ITERS = 20
NOISE_SIGMAS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
N_TRIALS_PER_SIGMA = 40


def load_dataset(name, n_images):
    rng = np.random.default_rng(SEED + 555)
    if name == 'mnist':
        from tensorflow.keras.datasets import mnist
        (x_train, _), _ = mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        idx = rng.choice(len(x_train), size=n_images, replace=False)
        return x_train[idx].reshape(n_images, -1), 784
    elif name == 'fashion_mnist':
        from tensorflow.keras.datasets import fashion_mnist
        (x_train, _), _ = fashion_mnist.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        idx = rng.choice(len(x_train), size=n_images, replace=False)
        return x_train[idx].reshape(n_images, -1), 784
    elif name == 'cifar100_gray':
        from tensorflow.keras.datasets import cifar100
        (x_train, _), _ = cifar100.load_data()
        x_train = x_train.astype(np.float32) / 255.0
        gray = x_train.mean(axis=-1)  # (N, 32, 32) -- grayscale
        idx = rng.choice(len(gray), size=n_images, replace=False)
        return gray[idx].reshape(n_images, -1), 1024
    else:
        raise ValueError(name)


def train_pseudo_inverse_dual(X, T, ridge_lambda=RIDGE_LAMBDA):
    Xm = X.astype(np.float64); Tm = T.astype(np.float64)
    Mloc = Xm.shape[0]
    Amat = Xm @ Xm.T + ridge_lambda * np.eye(Mloc)
    alpha = np.linalg.solve(Amat, Tm)
    return (Xm.T @ alpha).T.astype(np.float32)


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


def run_config(dataset_name, M, label):
    print(f'\n{"=" * 78}\nCONFIG: {label}  (dataset={dataset_name}, M={M})\n{"=" * 78}')
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    pool = generate_unique(M, rng)
    W_SH = make_projection(rng)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    H_keys = make_engrams(C_all, W_SH, K)
    B_drive = drive_vectors(pool)
    digits_true_all = pool.reshape(M, N)
    conf = conflict_matrix()

    images, img_dim = load_dataset(dataset_name, M)
    print(f'  loaded {M} images, dim={img_dim}  [{time.time()-t0:.1f}s]')

    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32))
    W_IS = train_pseudo_inverse_dual(H_keys.astype(np.float32), images)
    print(f'  trained W_SI ({H}x{img_dim}) and W_IS ({img_dim}x{H})  [{time.time()-t0:.1f}s]')

    h_clean_pred = np.array([k_wta(W_SI @ images[m]) for m in range(M)])
    clean_overlap = np.mean([np.sum(h_clean_pred[m] * H_keys[m]) / K for m in range(M)])
    recon_clean = W_IS @ H_keys.astype(np.float32).T
    recon_corr = np.mean([np.corrcoef(recon_clean[:, m], images[m])[0, 1] for m in range(M)])
    print(f'  sanity (zero noise): image->engram overlap={clean_overlap:.3f}, '
          f'engram->image recon corr={recon_corr:.3f}')

    rng_test = np.random.default_rng(SEED + 999)
    rows = []
    for sigma in NOISE_SIGMAS:
        for _ in range(N_TRIALS_PER_SIGMA):
            m = rng_test.integers(M)
            x_noisy = add_noise(images[m:m + 1], sigma, rng_test)[0]

            h_initial = k_wta(W_SI @ x_noisy)
            initial_overlap = np.sum(h_initial * H_keys[m]) / K

            h_cleaned, digits_final, iters, converged = iterate_sudoku_attractor(
                H_keys, B_drive, W_SH, conf, h_initial)
            cleaned_overlap = np.sum(h_cleaned * H_keys[m]) / K
            exact = int(np.all(digits_final == digits_true_all[m]))

            img_recon_cleaned = W_IS @ h_cleaned.astype(np.float32)
            recon_correlation = np.corrcoef(img_recon_cleaned, images[m])[0, 1]
            img_recon_initial = W_IS @ h_initial.astype(np.float32)
            recon_correlation_initial = np.corrcoef(img_recon_initial, images[m])[0, 1]

            rows.append({'config': label, 'dataset': dataset_name, 'M': M, 'img_dim': img_dim,
                         'sigma': sigma, 'target': m,
                         'initial_overlap': initial_overlap, 'cleaned_overlap': cleaned_overlap,
                         'exact': exact, 'recon_corr_initial': recon_correlation_initial,
                         'recon_corr_cleaned': recon_correlation})
        sub = [r for r in rows if r['sigma'] == sigma]
        print(f'  sigma={sigma:.2f}  initial_ov={np.mean([r["initial_overlap"] for r in sub]):.3f}  '
              f'cleaned_ov={np.mean([r["cleaned_overlap"] for r in sub]):.3f}  '
              f'exact={np.mean([r["exact"] for r in sub]):.2%}  '
              f'recon_corr={np.mean([r["recon_corr_initial"] for r in sub]):.3f}->'
              f'{np.mean([r["recon_corr_cleaned"] for r in sub]):.3f}  '
              f'[{time.time()-t0:.1f}s]')
    return pd.DataFrame(rows), clean_overlap, recon_corr


def main():
    configs = [
        ('mnist', 200, 'MNIST M=200 (baseline)'),
        ('mnist', 500, 'MNIST M=500 (capacity test)'),
        ('mnist', 800, 'MNIST M=800 (capacity test)'),
        ('fashion_mnist', 200, 'Fashion-MNIST M=200 (diff. dataset, same dims)'),
        ('cifar100_gray', 200, 'CIFAR-100 grayscale M=200 (diff. dataset AND dims)'),
    ]

    all_dfs = []
    sanity_rows = []
    for dataset_name, M, label in configs:
        df, clean_ov, recon_corr = run_config(dataset_name, M, label)
        all_dfs.append(df)
        sanity_rows.append({'config': label, 'dataset': dataset_name, 'M': M,
                            'zero_noise_overlap': clean_ov, 'zero_noise_recon_corr': recon_corr})

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.to_csv('heteroassoc_vectorhash_generalized.csv', index=False)
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv('heteroassoc_vectorhash_generalized_sanity.csv', index=False)

    print('\n\n=== ZERO-NOISE SANITY (does the pseudo-inverse even fit at this M/dim?) ===')
    print(sanity_df.to_string(index=False))

    # ---------------- plots ----------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))
    for (dataset_name, M, label), color in zip(configs, colors):
        sub = full_df[full_df.config == label].groupby('sigma').agg(
            cleaned_overlap=('cleaned_overlap', 'mean'), exact=('exact', 'mean'),
        ).reset_index()
        axes[0].plot(sub.sigma, sub.cleaned_overlap, 'o-', color=color, label=label)
        axes[1].plot(sub.sigma, sub.exact, 'o-', color=color, label=label)
    axes[0].axhline(K / H, color='gray', linestyle=':', label='chance')
    axes[0].set_xlabel('Image noise (sigma)'); axes[0].set_ylabel('Cleaned engram overlap with true target')
    axes[0].set_title('Engram quality after Sudoku cleanup'); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Image noise (sigma)'); axes[1].set_ylabel('Exact grid recovery')
    axes[1].set_title('Exact recovery rate'); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('heteroassoc_vectorhash_generalized.png', dpi=150)
    print('\nSaved heteroassoc_vectorhash_generalized.png')


if __name__ == '__main__':
    main()
