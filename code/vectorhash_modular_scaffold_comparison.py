"""
Adds an actual Vector-HaSH-style MODULAR scaffold to the hopfield_modern_sdm_dam_comparison.py
baseline set. Everything else in this project's cross-modal section (heteroassoc_vectorhash_*)
borrows Vector-HaSH's bidirectional pseudo-inverse HETEROASSOCIATION RULE but keeps our own
flat, non-modular scaffold (W_SH random projection + k-WTA). This script instead replaces the
scaffold itself with Vector-HaSH's actual defining mechanism: several small, pairwise-coprime
periodic modules (grid-cell-like), each contributing a one-hot code for its own residue of the
stored item's index; the combined code (concatenation across modules) addresses a space of size
PRODUCT(moduli) using only SUM(moduli) total units -- the "expansion via modularity" property
that is Vector-HaSH's central claim, tested here directly against our own flat random-projection
scaffold and against DAM/SDM/modern-Hopfield/ours, all on the identical M-sweep, noise-sweep,
images, and (where the RNG usage matches) the identical trial-by-trial noise draws.

Moduli are chosen as 10 primes (97..139) summing to ~1027 -- deliberately close to our own
H=1000, so this is a same-total-neuron-budget comparison of scaffold ORGANIZATION (flat sparse
random code vs. modular periodic code), not a comparison confounded by one method simply having
more units than the other. Per-code sparsity is far higher for the modular scaffold (10 active
of 1027, ~1%) than ours (150 of 1000, 15%), a direct structural consequence of one-hot-per-module
coding, not a tuned choice.

UPDATE: the first version of this script cleaned up a noisy grid-code estimate with independent
per-module argmax -- snap each module to its own nearest phase, with no interaction between
modules. That is NOT what Vector-HaSH actually does, and it showed: recovery collapsed far
faster with M than every other method tested (7.5% at M=800 vs. 57.5-100% for everything else),
because success required all 10 independent argmax decisions to be simultaneously correct, with
no mechanism for confident modules to help correct a noisy one.

This version adds genuine recurrent, cross-module correction: a Hopfield-style iterative
attractor over the space of the M known VALID grid codes (not over independent module phases).
Each valid code is, by construction, a globally-consistent combination of all 10 modules'
residues for one specific stored item -- so relaxing toward the nearest attractor state in code
space automatically lets every module's evidence inform every other module's correction, exactly
the cross-module consistency the real model's recurrent dynamics provide, without requiring an
explicit exact per-module vote. This reuses the same softmax-relaxation mathematical form as
modern_hopfield_recall() in hopfield_modern_sdm_dam_comparison.py, applied to the space of grid
codes instead of the space of raw images -- the discrete "attractor" analogue that a real
recurrent module network converges to on its own, made explicit as an iterative fixed point here.
Success is exact grid-code recovery, unchanged from before.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import binom
import sys
sys.path.insert(0, '.')
from heteroassoc_vectorhash_generalized import load_dataset, train_pseudo_inverse_dual, add_noise

SEED = 42
MODULI = [97, 101, 103, 107, 109, 113, 127, 131, 137, 139]
TOTAL_UNITS = sum(MODULI)
NUM_MODULES = len(MODULI)
RIDGE_LAMBDA = 1.0
# Validated (not carried over) exactly as MODERN_BETA/DAM_N were in
# hopfield_modern_sdm_dam_comparison.py: swept {1,5,20,60,100,150} against zero-noise and
# sigma=0.3 self-recovery across M=100/400/800, and against sigma=0.6-1.0 at M=50 for
# stability at the high-noise end. beta<20 fails to even lock onto a clean cue (M>=400);
# beta=100 gave the best balance across the whole range (M=800, sigma=0.3: 90%; M=50,
# sigma=1.0: 88%, both better than beta=60 or beta=150 at the corresponding extreme).
ATTRACTOR_BETA = 100.0
ATTRACTOR_MAX_ITERS = 30

M_SWEEP = [100, 200, 400, 800]
SIGMA_FOR_M_SWEEP = 0.3
N_TRIALS = 200

NOISE_SWEEP_M = 50
SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - halfwidth, center + halfwidth)


def grid_code(m, moduli=MODULI):
    code = np.zeros(sum(moduli), dtype=np.float32)
    offset = 0
    for p in moduli:
        code[offset + (m % p)] = 1.0
        offset += p
    return code


def grid_codes_all(M, moduli=MODULI):
    return np.stack([grid_code(m, moduli) for m in range(M)])


def clean_grid_code(raw, moduli=MODULI):
    """One-shot, independent per-module argmax -- no cross-module interaction. Kept only as
    the (deliberately weaker) reference point for measuring what recurrence adds."""
    out = np.zeros_like(raw)
    offset = 0
    for p in moduli:
        seg = raw[offset:offset + p]
        out[offset + int(np.argmax(seg))] = 1.0
        offset += p
    return out


def recurrent_attractor_decode(codes_norm, raw, beta, max_iters=30, tol=1e-6):
    """Hopfield-style relaxation over the space of the M known valid grid codes. Each valid
    code encodes one globally-consistent combination of all module residues, so converging
    toward the nearest one lets confident modules' evidence correct a noisy module's -- the
    cross-module correction independent per-module argmax cannot provide. Returns the decoded
    item index (argmax similarity to the fixed point)."""
    xi = raw.astype(np.float64).copy()
    xi = xi / (np.linalg.norm(xi) + 1e-12)
    for _ in range(max_iters):
        sims = codes_norm @ xi
        sims = sims - sims.max()
        w = np.exp(beta * sims)
        w = w / w.sum()
        xi_new = codes_norm.T @ w
        xi_new = xi_new / (np.linalg.norm(xi_new) + 1e-12)
        if np.max(np.abs(xi_new - xi)) < tol:
            xi = xi_new
            break
        xi = xi_new
    return int(np.argmax(codes_norm @ xi))


def product(vals):
    r = 1
    for v in vals:
        r *= v
    return r


def run_one_M(M_local, sigma_levels, n_trials, label_prefix):
    t0 = time.time()
    images, img_dim = load_dataset('mnist', M_local)
    codes = grid_codes_all(M_local)
    codes_norm = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-12)
    W_SI = train_pseudo_inverse_dual(images, codes, RIDGE_LAMBDA)

    rows = []
    for sigma in sigma_levels:
        rng = np.random.default_rng(SEED + 999)
        results_oneshot, results_recurrent = [], []
        for _ in range(n_trials):
            m = rng.integers(M_local)
            x_noisy = add_noise(images[m:m + 1], sigma, rng)[0]
            raw = W_SI @ x_noisy
            results_oneshot.append(int(np.array_equal(clean_grid_code(raw), codes[m])))
            m_decoded = recurrent_attractor_decode(codes_norm, raw, ATTRACTOR_BETA,
                                                     ATTRACTOR_MAX_ITERS)
            results_recurrent.append(int(m_decoded == m))
        lo_o, hi_o = wilson_ci(sum(results_oneshot), n_trials)
        lo_r, hi_r = wilson_ci(sum(results_recurrent), n_trials)
        rows.append({'M': M_local, 'sigma': sigma,
                     'vectorhash_oneshot_exact': np.mean(results_oneshot),
                     'vectorhash_oneshot_ci_lo': lo_o, 'vectorhash_oneshot_ci_hi': hi_o,
                     'vectorhash_exact': np.mean(results_recurrent),
                     'vectorhash_ci_lo': lo_r, 'vectorhash_ci_hi': hi_r})
        print(f'  {label_prefix} sigma={sigma:.2f}  '
              f'one-shot={np.mean(results_oneshot):.3f}  '
              f'recurrent={np.mean(results_recurrent):.3f} [{lo_r:.3f},{hi_r:.3f}]  '
              f'[{time.time()-t0:.1f}s]')
    return rows


def main():
    print(f'Modular scaffold: {NUM_MODULES} modules, moduli={MODULI}')
    print(f'Total units={TOTAL_UNITS} (cf. our H=1000), active units per code={NUM_MODULES} '
          f'(cf. our K=150), addressable space=PRODUCT(moduli)={product(MODULI):.3e}')

    print(f'\n{"=" * 90}\nM-SWEEP at fixed sigma={SIGMA_FOR_M_SWEEP}, N={N_TRIALS}/point\n{"=" * 90}')
    m_rows = []
    for M_local in M_SWEEP:
        m_rows.extend(run_one_M(M_local, [SIGMA_FOR_M_SWEEP], N_TRIALS, f'M={M_local}'))
    df_m = pd.DataFrame(m_rows)
    df_m.to_csv('vectorhash_modular_scaffold_M_sweep.csv', index=False)

    print(f'\n{"=" * 90}\nNOISE-SWEEP at fixed M={NOISE_SWEEP_M}, N={N_TRIALS}/point\n{"=" * 90}')
    noise_rows = run_one_M(NOISE_SWEEP_M, SIGMA_LEVELS, N_TRIALS, f'M={NOISE_SWEEP_M}')
    df_n = pd.DataFrame(noise_rows)
    df_n.to_csv('vectorhash_modular_scaffold_noise_sweep.csv', index=False)

    print('\nSaved vectorhash_modular_scaffold_M_sweep.csv, _noise_sweep.csv')

    try:
        existing_m = pd.read_csv('hopfield_modern_sdm_dam_M_sweep.csv')
        merged_m = existing_m.merge(df_m, on=['M', 'sigma'], how='inner')
        merged_m.to_csv('full_hetero_comparison_M_sweep.csv', index=False)
        cols = ['M', 'outer_exact', 'cov_exact', 'pinv_exact', 'modern_exact', 'sdm_exact',
                'dam_exact', 'ours_exact', 'vectorhash_exact']
        print(f'\n{"=" * 90}\nFULL M-SWEEP COMPARISON (sigma={SIGMA_FOR_M_SWEEP})\n{"=" * 90}')
        print(merged_m[cols].round(3).to_string(index=False))

        existing_n = pd.read_csv('hopfield_modern_sdm_dam_noise_sweep.csv')
        merged_n = existing_n.merge(df_n, on=['M', 'sigma'], how='inner')
        merged_n.to_csv('full_hetero_comparison_noise_sweep.csv', index=False)
        cols_n = ['sigma', 'outer_exact', 'cov_exact', 'pinv_exact', 'modern_exact', 'sdm_exact',
                  'dam_exact', 'ours_exact', 'vectorhash_exact']
        print(f'\n{"=" * 90}\nFULL NOISE-SWEEP COMPARISON (M={NOISE_SWEEP_M})\n{"=" * 90}')
        print(merged_n[cols_n].round(3).to_string(index=False))
    except FileNotFoundError:
        print('\nhopfield_modern_sdm_dam_*.csv not found in this directory -- run '
              'hopfield_modern_sdm_dam_comparison.py first (or copy its CSVs here) to get '
              'the full merged comparison table.')


if __name__ == '__main__':
    main()
