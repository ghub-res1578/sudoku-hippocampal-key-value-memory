"""
Tests our architecture against the remaining four baselines from Vector-HaSH's own Fig. 3d
comparison (Chandra, Sharma, Chaudhuri & Fiete, Nature 2025) that we had not yet built:
classical and pseudo-inverse Hopfield are already covered extensively elsewhere in this
project (hopfield_vs_our_network_comparison.py, hopfield_modern_sdm_dam_comparison.py). The
four new ones, exactly as specified in that paper's Methods ("Parameter values", Figure 3
paragraph):

  (3) Bounded-synapse Hopfield: standard Hebbian outer-product learning applied SEQUENTIALLY
      (one pattern at a time, not batched), with each synapse clipped to a fixed range after
      every update -- the palimpsest/bounded-synapse memory model (Fusi & Abbott 2007-style).
      Clip bound = 5/N, chosen so a synapse saturates after ~5 coherently-reinforcing pattern
      contributions -- documented explicitly since Vector-HaSH's Methods does not state a
      specific bound.

  (4) Sparse-input Hopfield: the Tsodyks-Feigelman (1988) learning rule for low-activity
      patterns, W_ij = 1/N sum_mu (x_i^mu - p)(x_j^mu - p) / (p(1-p)), corrected for the
      population-mean bias that a low mean-activity level otherwise introduces. For random
      patterns, p=0.1 (10% active, "sparsity" = 100(1-p) = 90%, a standard sparse-Hopfield
      operating point). For real MNIST images, p is computed from the DATA'S OWN mean
      bipolarized activity level, not chosen arbitrarily -- real MNIST is naturally sparse
      (~14% foreground, already established in this project's classical-Hopfield discussion).

  (5) Sparse-synapse (diluted) Hopfield: classical outer-product Hopfield with a fixed random
      binary dilution mask retaining a kappa fraction of the N^2 synapses, kappa=0.5.

  (6) Tail-biting overparameterized autoencoder: an unconstrained, end-to-end gradient-trained
      dense autoencoder (784-275-38-275-784, reusing Vector-HaSH's own reported hidden-layer
      sizes for their matched-synapse-budget comparison), with an output-to-input identity
      ("tail-biting") connection for iterative reconstruction at test time -- the one baseline
      in this set that is trained by backprop rather than a closed-form learning rule.

TWO test conditions, per the explicit choice to check whether the random-vs-real-image
distinction (already shown to matter enormously for SDM) matters for these baselines too:
  A. RANDOM bipolar patterns, dimension D=784 (matching our own image dimension rather than
     Vector-HaSH's N=708, for consistency with the rest of this project) -- the condition
     Vector-HaSH's own Fig. 3d actually uses.
  B. REAL MNIST images (bipolarized where the method requires bipolar states), our project's
     established comparison convention throughout Sections 4.6-4.8.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

import sys
sys.path.insert(0, '.')
from heteroassoc_vectorhash_generalized import load_dataset, add_noise
from hopfield_vs_our_network_comparison import bipolarize
from hopfield_modern_sdm_dam_comparison import wilson_ci

D = 784
M_SWEEP = [100, 200, 400, 800]
SIGMA = 0.3
N_TRIALS = 200
SEED_TRIALS = 24680
MAX_ITERS = 15

BOUNDED_CLIP = 5.0 / D
SPARSE_P_RANDOM = 0.1
DILUTION_KAPPA = 0.5
AE_HIDDEN = [275, 38, 275]
AE_EPOCHS = 400
AE_LR = 1e-3


# ---------- (3) Bounded-synapse Hopfield ----------

def train_bounded_hopfield(patterns_bipolar, clip=BOUNDED_CLIP):
    N = patterns_bipolar.shape[1]
    W = np.zeros((N, N))
    for x in patterns_bipolar:
        W += np.outer(x, x) / N
        W = np.clip(W, -clip, clip)
    np.fill_diagonal(W, 0.0)
    return W


def recall_bounded_hopfield(W, cue, max_iters=MAX_ITERS):
    s = cue.copy()
    for _ in range(max_iters):
        net = W @ s
        s_new = np.where(net == 0, s, np.sign(net))
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


# ---------- (4) Sparse-input Hopfield (Tsodyks-Feigelman) ----------

def train_sparse_hopfield(patterns_pm1, p):
    """patterns_pm1: bipolar {-1,+1} patterns. p: fraction of +1 (active) units."""
    N = patterns_pm1.shape[1]
    centered = (patterns_pm1 + 1) / 2 - p  # map to {0,1}-centered-at-p, i.e. activity - p
    W = (centered.T @ centered) / (N * p * (1 - p))
    np.fill_diagonal(W, 0.0)  # zero self-connections, matching every other Hopfield variant
    return W


def recall_sparse_hopfield(W, cue, p, max_iters=MAX_ITERS):
    """k-winner-take-all update at the SAME sparsity level p as the stored patterns -- a naive
    net>0 threshold does not enforce output sparsity and lets far more than a p-fraction of
    units go active for low p, which is not what the Tsodyks-Feigelman rule assumes."""
    N = len(cue)
    k = max(1, int(round(p * N)))
    s01 = (cue + 1) / 2
    for _ in range(max_iters):
        net = W @ (s01 - p)
        top_idx = np.argpartition(net, -k)[-k:]
        s01_new = np.zeros(N)
        s01_new[top_idx] = 1.0
        if np.array_equal(s01_new, s01):
            break
        s01 = s01_new
    return s01 * 2 - 1


# ---------- (5) Sparse-synapse (diluted) Hopfield ----------

def train_diluted_hopfield(patterns_bipolar, rng, kappa=DILUTION_KAPPA):
    N = patterns_bipolar.shape[1]
    W = (patterns_bipolar.T @ patterns_bipolar) / N
    np.fill_diagonal(W, 0.0)
    mask = (rng.random((N, N)) < kappa).astype(np.float64)
    mask = np.triu(mask, 1)
    mask = mask + mask.T
    return W * mask


recall_diluted_hopfield = recall_bounded_hopfield  # identical sign(Ws) dynamics


# ---------- (6) Tail-biting overparameterized autoencoder ----------

def train_tail_biting_autoencoder(patterns_01, seed):
    tf.random.set_seed(seed)
    d = patterns_01.shape[1]
    inputs = keras.Input(shape=(d,))
    x = inputs
    for h in AE_HIDDEN:
        x = keras.layers.Dense(h, activation='tanh')(x)
    outputs = keras.layers.Dense(d, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(AE_LR), loss='mse')
    model.fit(patterns_01, patterns_01, epochs=AE_EPOCHS, batch_size=min(64, len(patterns_01)),
              verbose=0)
    final_loss = model.evaluate(patterns_01, patterns_01, verbose=0)
    return model, final_loss


def recall_tail_biting_autoencoder(model, cue_01, max_iters=MAX_ITERS, tol=1e-5):
    s = cue_01.copy()[None, :]
    for _ in range(max_iters):
        s_new = model.predict(s, verbose=0)
        if np.max(np.abs(s_new - s)) < tol:
            s = s_new
            break
        s = s_new
    return s[0]


# ---------- shared trial harness ----------

def run_condition(label, patterns_source_fn, is_bipolar_native, M_sweep, sigma, n_trials):
    rows = []
    for M in M_sweep:
        t0 = time.time()
        patterns_01, patterns_bp, sparsity_p, sparse_patterns_01, sparse_patterns_bp = \
            patterns_source_fn(M)

        W_bounded = train_bounded_hopfield(patterns_bp)
        W_sparse = train_sparse_hopfield(sparse_patterns_bp, sparsity_p)
        rng_dil = np.random.default_rng(SEED_TRIALS)
        W_diluted = train_diluted_hopfield(patterns_bp, rng_dil)
        ae_model, ae_loss = train_tail_biting_autoencoder(patterns_01, SEED_TRIALS)

        rng = np.random.default_rng(SEED_TRIALS)
        results = {'bounded': [], 'sparse_input': [], 'diluted': [], 'tail_biting_ae': []}
        for _ in range(n_trials):
            m = rng.integers(M)
            x01_noisy = np.clip(patterns_01[m] + rng.normal(0, sigma, D), 0, 1)
            cue_bp = np.where(x01_noisy > 0.5, 1.0, -1.0)

            out_b = recall_bounded_hopfield(W_bounded, cue_bp)
            results['bounded'].append(int(np.array_equal(out_b, patterns_bp[m])))

            x01_sparse_noisy = np.clip(sparse_patterns_01[m] + rng.normal(0, sigma, D), 0, 1)
            cue_bp_sparse = np.where(x01_sparse_noisy > 0.5, 1.0, -1.0)
            out_s = recall_sparse_hopfield(W_sparse, cue_bp_sparse, sparsity_p)
            results['sparse_input'].append(int(np.array_equal(out_s, sparse_patterns_bp[m])))

            out_d = recall_diluted_hopfield(W_diluted, cue_bp)
            results['diluted'].append(int(np.array_equal(out_d, patterns_bp[m])))

            out_ae = recall_tail_biting_autoencoder(ae_model, x01_noisy)
            nn_idx = int(np.argmin(np.sum((patterns_01 - out_ae[None, :]) ** 2, axis=1)))
            results['tail_biting_ae'].append(int(nn_idx == m))

        row = {'condition': label, 'M': M, 'sigma': sigma, 'ae_final_train_loss': ae_loss}
        for key in results:
            arr = results[key]
            lo, hi = wilson_ci(sum(arr), len(arr))
            row[f'{key}_exact'] = np.mean(arr)
            row[f'{key}_ci_lo'] = lo
            row[f'{key}_ci_hi'] = hi
        rows.append(row)
        print(f'[{label}] M={M:5d}  ' +
              '  '.join(f'{k}={row[f"{k}_exact"]:.3f}' for k in results) +
              f'  ae_train_loss={ae_loss:.4f}  [{time.time()-t0:.1f}s]')
    return rows


def random_patterns_source(M, seed=SEED_TRIALS + 1):
    """Dense (p=0.5) patterns for bounded/diluted/autoencoder, matching Vector-HaSH's other
    Fig. 3d baselines; a SEPARATE, genuinely sparse (p=SPARSE_P_RANDOM) pattern set for the
    sparse-input Hopfield baseline specifically, since Vector-HaSH's Methods describes that
    one baseline as using sparse inputs, distinct from the shared dense patterns."""
    rng = np.random.default_rng(seed)
    patterns_bp = rng.choice([-1.0, 1.0], size=(M, D))
    patterns_01 = (patterns_bp + 1) / 2

    rng_sparse = np.random.default_rng(seed + 1)
    k = max(1, int(round(SPARSE_P_RANDOM * D)))
    sparse_patterns_01 = np.zeros((M, D))
    for i in range(M):
        idx = rng_sparse.choice(D, size=k, replace=False)
        sparse_patterns_01[i, idx] = 1.0
    sparse_patterns_bp = sparse_patterns_01 * 2 - 1

    return patterns_01, patterns_bp, SPARSE_P_RANDOM, sparse_patterns_01, sparse_patterns_bp


def real_image_source(M):
    """Real images are used as-is for every baseline, including sparse-input Hopfield: the
    data's own natural sparsity (~14% foreground) IS the "sparse input" condition here, not a
    separately synthesized pattern set."""
    images, img_dim = load_dataset('mnist', M)
    patterns_bp = bipolarize(images)
    p_data = float(np.mean(patterns_bp > 0))
    return images, patterns_bp, p_data, images, patterns_bp


def main():
    print(f'{"=" * 100}\nCONDITION A: random bipolar patterns, D={D}\n{"=" * 100}')
    rows_random = run_condition('random', random_patterns_source, True, M_SWEEP, SIGMA, N_TRIALS)

    print(f'\n{"=" * 100}\nCONDITION B: real MNIST images\n{"=" * 100}')
    rows_real = run_condition('real_mnist', real_image_source, False, M_SWEEP, SIGMA, N_TRIALS)

    df = pd.DataFrame(rows_random + rows_real)
    df.to_csv('vectorhash_own_baselines_comparison.csv', index=False)
    print('\nSaved vectorhash_own_baselines_comparison.csv')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = {'bounded': 'tab:red', 'sparse_input': 'tab:orange', 'diluted': 'tab:purple',
              'tail_biting_ae': 'tab:green'}
    labels = {'bounded': 'Bounded-synapse Hopfield', 'sparse_input': 'Sparse-input Hopfield',
              'diluted': f'Diluted Hopfield (kappa={DILUTION_KAPPA})',
              'tail_biting_ae': 'Tail-biting autoencoder'}
    for ax, (cond_label, title) in zip(axes, [('random', 'Random patterns (D=784)'),
                                                ('real_mnist', 'Real MNIST images')]):
        sub = df[df.condition == cond_label]
        for key in colors:
            ax.errorbar(sub.M, sub[f'{key}_exact'],
                        yerr=[sub[f'{key}_exact'] - sub[f'{key}_ci_lo'],
                              sub[f'{key}_ci_hi'] - sub[f'{key}_exact']],
                        fmt='o-', color=colors[key], label=labels[key], capsize=3)
        ax.set_xlabel('M (number of stored patterns)')
        ax.set_ylabel('Exact recovery rate')
        ax.set_title(title)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle(f'Vector-HaSH\'s own Fig. 3d baselines (new four), sigma={SIGMA}, N={N_TRIALS}/point')
    fig.tight_layout()
    fig.savefig('vectorhash_own_baselines_comparison.png', dpi=150)
    print('Saved vectorhash_own_baselines_comparison.png')


if __name__ == '__main__':
    main()
