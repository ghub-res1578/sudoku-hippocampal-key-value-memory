"""
Statistically robust, PAIRED partial-cue completion sweep (Table 6 in the paper): initial
guess / fixed-weight spiking dynamics / hard-coded numpy conflict mask, all evaluated on the
IDENTICAL target grid + known-cell mask at each trial, so the comparison is paired (removes
trial-to-trial sampling noise). Reports a two-sided Wilcoxon signed-rank test (dynamics vs.
hard-coded rule) per clue fraction.

Requires: hippocampus_drive_readout.py, separated_edit_no_plasticity.py,
feedback_iterate_highM_pool.npy (or regenerate via feedback_iterate_highM.py) in the same
directory / on sys.path.
"""
import time
import numpy as np
import pandas as pd
from scipy import stats
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network
from hippocampus_drive_readout import (
    H, N, K, SEED as HSEED, TAU, relation, make_projection, make_engrams, drive_vectors,
    readout_attention, decode_digits, clean_engram, conflict_matrix, coincidence,
)

SEED = 42
THETA_MS = 2.0
POOL_FILE = 'feedback_iterate_highM_pool.npy'
M = 200
CLUE_FRACTIONS = [0.15, 0.19, 0.21, 0.25, 0.27, 0.30, 0.35]
N_TRIALS = 20

W_IE = 0.005
W_EE = 0.001
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
OSC_PERIOD_MS = 100.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])


def build_fixed_weight_net():
    params = dict(DEFAULT_PARAMS)
    params['w_IE'] = W_IE
    params['w_EE'] = W_EE
    params['runtime_ms'] = WINDOW_MS
    params['drive_duration_ms'] = WINDOW_MS + 1
    net, comps = build_network(params=params)
    return net, comps, params


def run_fixed_weight_trial(drive, seed):
    net, comps, params = build_fixed_weight_net()
    rng = np.random.default_rng(seed)
    E = comps['E']
    low, high = params['init_perturb_low'], params['init_perturb_high']
    E.v = rng.random(N) * 0.1 + rng.uniform(low, high, size=N)
    comps['I'].v = 0.0
    E.I_d = drive
    E.osc_on = 1.0
    net.run(WINDOW_MS * ms)
    spkE = comps['spkE']
    t_all = np.asarray(spkE.t / ms); i_all = np.asarray(spkE.i)
    return [np.sort(t_all[i_all == n]) for n in range(N)]


def circular_diff(p1, p2, period=OSC_PERIOD_MS):
    d = np.abs(p1 - p2)
    return np.minimum(d, period - d)


def pair_phase_offset(train_i, train_j):
    t_i = train_i[train_i >= STEADY_STATE_START_MS]
    t_j = train_j[train_j >= STEADY_STATE_START_MS]
    if len(t_i) == 0 or len(t_j) == 0:
        return np.nan, 0
    offsets = []
    for t in t_i:
        nearest_j = t_j[np.argmin(np.abs(t_j - t))]
        if abs(nearest_j - t) < OSC_PERIOD_MS:
            offsets.append(circular_diff(t % OSC_PERIOD_MS, nearest_j % OSC_PERIOD_MS))
    if len(offsets) < 3:
        return np.nan, len(offsets)
    return np.mean(offsets), len(offsets)


def build_emergent_coincidence(trains, theta=THETA_MS):
    C_hat = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            m, n = pair_phase_offset(trains[i], trains[j])
            if n >= 3 and not np.isnan(m) and m <= theta:
                C_hat[i, j] = C_hat[j, i] = True
    return C_hat


def main():
    pool_full = np.load(POOL_FILE)[:M]
    digits_true_all = pool_full.reshape(M, N)

    print('Building numpy engram...')
    rng_proj = np.random.default_rng(HSEED + 50)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool_full], dtype=np.float32)
    B_drive = drive_vectors(pool_full)
    H_keys = make_engrams(C_all, W_SH, K)
    conf = conflict_matrix()

    print('\n' + '=' * 78)
    print(f'PAIRED SWEEP: {len(CLUE_FRACTIONS)} clue fractions x {N_TRIALS} trials, '
          f'3 methods per trial (paired)')
    print('=' * 78)

    trng = np.random.default_rng(SEED + 12345)
    rows = []
    t0 = time.time()
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        targets = trng.choice(M, size=min(N_TRIALS, M), replace=False)
        for ti, t in enumerate(targets):
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
            digits_true = digits_true_all[t]

            x = pool_full[t].ravel()
            Cp = (x[:, None] == x[None, :])
            Cp = Cp & (known_mask[:, None] & known_mask[None, :])
            np.fill_diagonal(Cp, False)
            a = np.asarray(W_SH @ Cp.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx] = 1
            b_hat = readout_attention(H_keys, B_drive, h_q, tau=TAU)
            digits_engram_guess = decode_digits(b_hat)
            digits_drive_input = digits_engram_guess.copy()
            digits_drive_input[known_mask] = digits_true[known_mask]
            drive = DESIGNATED_RANGE[digits_drive_input - 1]

            base_seed = SEED + 999 + t * 7 + ti
            trains_fixed = run_fixed_weight_trial(drive, base_seed)
            C_hat_fixed = build_emergent_coincidence(trains_fixed)

            def decode_from_C(C):
                h_clean = clean_engram(C, W_SH, K)
                b = readout_attention(H_keys, B_drive, h_clean, tau=TAU)
                return decode_digits(b)

            digits_fixed = decode_from_C(C_hat_fixed)
            C_numpy_clean = coincidence(digits_drive_input, conf)
            digits_numpy = decode_from_C(C_numpy_clean)

            unknown_mask = ~known_mask; n_unk = unknown_mask.sum()

            def unk_acc(d):
                return np.mean(d[unknown_mask] == digits_true[unknown_mask]) if n_unk else np.nan

            rows.append({
                'clue_fraction': clue_frac, 'target': int(t),
                'acc_initial': unk_acc(digits_drive_input),
                'acc_fixed': unk_acc(digits_fixed),
                'acc_numpy': unk_acc(digits_numpy),
                'exact_initial': int(np.all(digits_drive_input == digits_true)),
                'exact_fixed': int(np.all(digits_fixed == digits_true)),
                'exact_numpy': int(np.all(digits_numpy == digits_true)),
            })
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        acc_fixed_arr = np.array([r['acc_fixed'] for r in sub])
        acc_numpy_arr = np.array([r['acc_numpy'] for r in sub])
        diff = acc_fixed_arr - acc_numpy_arr
        if np.any(diff != 0):
            wstat, wp = stats.wilcoxon(diff)
        else:
            wstat, wp = (np.nan, 1.0)
        print(f'clue_frac={clue_frac:.2f}  '
              f'initial={np.nanmean([r["acc_initial"] for r in sub]):.3f}  '
              f'fixed={np.nanmean(acc_fixed_arr):.3f}  '
              f'numpy={np.nanmean(acc_numpy_arr):.3f}  '
              f'wilcoxon_p(fixed vs numpy)={wp:.4f}  '
              f'[{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('spiking_paired_significance_sweep.csv', index=False)
    print('\nSaved spiking_paired_significance_sweep.csv')
    return df


if __name__ == '__main__':
    main()
