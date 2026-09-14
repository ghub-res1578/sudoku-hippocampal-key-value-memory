"""
Extends the single-point (M=800) beta*Delta fairness check across the full M=100-4000 grid
already used throughout this project's Modern Hopfield / single-shot-readout cliff tests.

Question being answered: is comparing our raw beta=1/tau=0.5 to Modern Hopfield's raw beta=60
fair? Raw beta values are not comparable on their own because they multiply similarity scores
living on completely different scales (our engram-overlap "sim" ranges 0-K=150; Modern
Hopfield's cosine "sim" ranges -1 to 1). What determines actual discriminative sharpness is the
PRODUCT beta*Delta, where Delta = true-match similarity minus the best-competitor similarity --
this is exactly the quantity Ramsauer et al.'s own separation theorem is stated in terms of. This
script measures Delta directly (from clean, zero-noise queries, so it isolates the geometry of
the stored representations themselves, not noise-driven distortion) for BOTH models at every M in
the standard sweep, and reports beta*Delta for each, so the M=800 finding (ours=51.74 far exceeds
Modern Hopfield's=11.49) can be checked for robustness across the whole capacity range rather than
trusted at a single M.
"""
import time
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from hippocampus_drive_readout import N, SEED, TAU, relation, generate_unique, drive_vectors
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual
from heteroassoc_H_K_sweep import make_projection_hk, make_engrams_hk, k_wta_hk
from hopfield_modern_sdm_dam_comparison import train_modern_hopfield, MODERN_BETA

H_val, K_val = 1000, 150
RIDGE_LAMBDA = 1.0
OUR_BETA = 1.0 / TAU
M_SWEEP = [100, 200, 400, 800, 1200, 1600, 2000, 2500, 3000, 4000]
N_SAMPLE = 300  # patterns sampled per M to estimate mean Delta (all M if M <= N_SAMPLE)


def modern_hopfield_delta(images, M_local, rng):
    X = train_modern_hopfield(images)  # L2-normalized, shape (M, D)
    gram = X @ X.T  # cosine similarities, (M, M)
    idx = rng.choice(M_local, size=min(N_SAMPLE, M_local), replace=False)
    deltas, true_sims = [], []
    for m in idx:
        row = gram[m].copy()
        true_sim = row[m]
        row[m] = -np.inf
        best_competitor = row.max()
        deltas.append(true_sim - best_competitor)
        true_sims.append(true_sim)
    return float(np.mean(true_sims)), float(np.mean(deltas))


def ours_delta(M_local, rng):
    rng0 = np.random.default_rng(SEED)
    pool = generate_unique(M_local, rng0)
    C_all = np.asarray([relation(g).ravel() for g in pool], dtype=np.float32)
    images, img_dim = load_dataset('mnist', M_local)
    rng_proj = np.random.default_rng(SEED)
    W_SH = make_projection_hk(rng_proj, H_val)
    H_keys = make_engrams_hk(C_all, W_SH, H_val, K_val)  # (M, H) binary, K active/row
    W_SI = train_pseudo_inverse_dual(images, H_keys.astype(np.float32), RIDGE_LAMBDA)

    idx = rng.choice(M_local, size=min(N_SAMPLE, M_local), replace=False)
    deltas, true_sims = [], []
    for m in idx:
        h_query = k_wta_hk(W_SI @ images[m], H_val, K_val)
        sims = H_keys.astype(np.float32) @ h_query.astype(np.float32)  # (M,), 0..K
        true_sim = sims[m]
        s = sims.copy()
        s[m] = -np.inf
        best_competitor = s.max()
        deltas.append(true_sim - best_competitor)
        true_sims.append(true_sim)
    return float(np.mean(true_sims)), float(np.mean(deltas))


def main():
    print(f'{"=" * 100}\nFull M-sweep: beta*Delta for Modern Hopfield (beta={MODERN_BETA}) '
          f'vs. ours (beta={OUR_BETA:g}), clean queries, N_SAMPLE={N_SAMPLE}\n{"=" * 100}')
    rng = np.random.default_rng(2026)
    rows = []
    for M_local in M_SWEEP:
        t0 = time.time()
        images, img_dim = load_dataset('mnist', M_local)
        mh_true, mh_delta = modern_hopfield_delta(images, M_local, rng)
        mh_eff = MODERN_BETA * mh_delta
        ours_true, ours_delta_val = ours_delta(M_local, rng)
        ours_eff = OUR_BETA * ours_delta_val
        ratio = ours_eff / mh_eff
        rows.append({'M': M_local, 'mh_true_sim': mh_true, 'mh_delta': mh_delta,
                      'mh_beta_x_delta': mh_eff, 'ours_true_sim': ours_true,
                      'ours_delta': ours_delta_val, 'ours_beta_x_delta': ours_eff,
                      'ratio_ours_over_mh': ratio})
        print(f'  M={M_local:5d}  MH: Delta={mh_delta:.4f} beta*Delta={mh_eff:7.2f}  |  '
              f'Ours: Delta={ours_delta_val:7.2f} beta*Delta={ours_eff:7.2f}  |  '
              f'ratio(ours/MH)={ratio:5.2f}x  [{time.time()-t0:.1f}s]')
    df = pd.DataFrame(rows)
    df.to_csv('beta_delta_fairness_full_M_sweep.csv', index=False)
    print('\nSaved beta_delta_fairness_full_M_sweep.csv')


if __name__ == '__main__':
    main()
