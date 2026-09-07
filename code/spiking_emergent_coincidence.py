"""
Reframing: the spiking network is not asked to CORRECT a wrong per-cell digit guess. It is
asked to produce a COINCIDENCE MATRIX -- the same 81x81 same-digit relation the rest of the
architecture (W_SH projection -> k-WTA engram -> attention readout) already consumes -- via
its own dynamics, replacing the hardcoded numpy conflict_matrix()/coincidence() step.

Readout: run the network at steady state (long window, transient discarded, established in
spiking_steady_state_phase_test.py), and for EVERY pair of cells, classify them as "same
cluster" (synchronized, phase offset <= theta) or not, via the coincidence-window threshold
validated in the real-Sudoku test (same-digit offset ~0.2ms, different-digit ~13ms -- clean
gap around 2-3ms). This gives an emergent Ĉ, built from dynamics, not decoded digits.

Ĉ is then fed through the EXISTING, already-characterized engram pipeline exactly like the
partial-coincidence-matrix experiments (sudoku_partial_grid_completion.py etc): project
through W_SH, k-WTA to an engram, attention-readout against the M stored (engram, drive)
pairs, decode digits. The robustness to a noisy/imperfect Ĉ is expected to come from THAT
already-validated attention readout, not from the spiking dynamics being a perfect corrector.
"""
import time
import numpy as np
import pandas as pd
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network
from hippocampus_drive_readout import (
    H, N, K, SEED as HSEED, TAU,
    relation, make_projection, make_engrams, drive_vectors,
    readout_attention, decode_digits, clean_engram, conflict_matrix, coincidence,
)

SEED = 42
W_IE = 0.005
W_EE = 0.001
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
OSC_PERIOD_MS = 100.0
THETA_MS = 2.0                 # coincidence window: same-digit ~0.2ms, diff-digit ~13ms in
                                # the real-grid test -- 2ms sits cleanly in the gap
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])

POOL_FILE = 'feedback_iterate_highM_pool.npy'
M = 200
CLUE_FRACTIONS = [0.10, 0.15, 0.19, 0.21, 0.23, 0.25, 0.27, 0.30, 0.35, 0.40, 0.50]
N_TRIALS = 25


def build_fresh():
    params = dict(DEFAULT_PARAMS)
    params['w_IE'] = W_IE
    params['w_EE'] = W_EE
    params['runtime_ms'] = WINDOW_MS
    params['drive_duration_ms'] = WINDOW_MS + 1
    net, comps = build_network(params=params)
    return net, comps, params


def run_trial(drive, rng):
    net, comps, params = build_fresh()
    E = comps['E']
    low, high = params['init_perturb_low'], params['init_perturb_high']
    E.v = rng.random(N) * 0.1 + rng.uniform(low, high, size=N)
    comps['I'].v = 0.0
    E.I_d = drive
    E.osc_on = 1.0
    net.run(WINDOW_MS * ms)
    spkE = comps['spkE']
    t_all = np.asarray(spkE.t / ms); i_all = np.asarray(spkE.i)
    trains = [np.sort(t_all[i_all == n]) for n in range(N)]
    return trains


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

    print('Building numpy engram (for the initial per-cell drive guess)...')
    rng_proj = np.random.default_rng(HSEED + 50)
    W_SH = make_projection(rng_proj)
    C_all = np.asarray([relation(g).ravel() for g in pool_full], dtype=np.float32)
    B_drive = drive_vectors(pool_full)
    H_keys = make_engrams(C_all, W_SH, K)
    conf = conflict_matrix()

    print('\n' + '=' * 78)
    print(f'EMERGENT COINCIDENCE MATRIX PIPELINE: theta={THETA_MS}ms, w_IE={W_IE}, w_EE={W_EE}, '
          f'window={WINDOW_MS}ms')
    print('=' * 78)

    trng = np.random.default_rng(SEED + 12345)
    rng_net = np.random.default_rng(SEED + 999)
    rows = []
    t0 = time.time()
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        targets = trng.choice(M, size=min(N_TRIALS, M), replace=False)
        for t in targets:
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
            digits_true = digits_true_all[t]
            C_true = relation(pool_full[t])

            x = pool_full[t].ravel()
            Cp = (x[:, None] == x[None, :])
            known2d = known_mask[:, None] & known_mask[None, :]
            Cp = Cp & known2d
            np.fill_diagonal(Cp, False)
            a = np.asarray(W_SH @ Cp.astype(np.float32).ravel()).ravel()
            idx = np.argpartition(a, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx] = 1
            b_hat = readout_attention(H_keys, B_drive, h_q, tau=TAU)
            digits_engram_guess = decode_digits(b_hat)

            digits_drive_input = digits_engram_guess.copy()
            digits_drive_input[known_mask] = digits_true[known_mask]
            drive = DESIGNATED_RANGE[digits_drive_input - 1]

            trains = run_trial(drive, rng_net)
            C_hat = build_emergent_coincidence(trains)

            # quality of the emergent coincidence matrix itself, vs. ground truth
            iu, ju = np.triu_indices(N, k=1)
            true_pairs = C_true[iu, ju]
            hat_pairs = C_hat[iu, ju]
            tp = np.sum(true_pairs & hat_pairs); fp = np.sum(~true_pairs & hat_pairs)
            fn = np.sum(true_pairs & ~hat_pairs); tn = np.sum(~true_pairs & ~hat_pairs)
            c_acc = (tp + tn) / len(true_pairs)
            c_sens = tp / max(tp + fn, 1)   # of true same-digit pairs, how many found
            c_spec = tn / max(tn + fp, 1)   # of true different-digit pairs, how many correctly excluded

            # feed the emergent (dynamics-derived) coincidence matrix into the EXISTING
            # engram pipeline, exactly like the numpy conflict-mask cleanup step
            h_clean = clean_engram(C_hat, W_SH, K)
            b_hat2 = readout_attention(H_keys, B_drive, h_clean, tau=TAU)
            digits_from_dynamics = decode_digits(b_hat2)

            # for comparison: what the OLD hardcoded numpy conflict-mask cleanup gives,
            # starting from the SAME initial digit guess
            C_numpy_clean = coincidence(digits_drive_input, conf)
            h_clean_numpy = clean_engram(C_numpy_clean, W_SH, K)
            b_hat3 = readout_attention(H_keys, B_drive, h_clean_numpy, tau=TAU)
            digits_numpy_clean = decode_digits(b_hat3)

            unknown_mask = ~known_mask
            n_unk = unknown_mask.sum()

            def unk_acc(d):
                return np.mean(d[unknown_mask] == digits_true[unknown_mask]) if n_unk else np.nan

            rows.append({
                'clue_fraction': clue_frac, 'target': int(t),
                'coinc_acc': c_acc, 'coinc_sens': c_sens, 'coinc_spec': c_spec,
                'unk_acc_initial_guess': unk_acc(digits_drive_input),
                'unk_acc_dynamics_coinc': unk_acc(digits_from_dynamics),
                'unk_acc_numpy_conflict_mask': unk_acc(digits_numpy_clean),
                'exact_initial': int(np.all(digits_drive_input == digits_true)),
                'exact_dynamics': int(np.all(digits_from_dynamics == digits_true)),
                'exact_numpy': int(np.all(digits_numpy_clean == digits_true)),
            })
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        print(f'clue_frac={clue_frac:.2f}  '
              f'coinc_acc={np.mean([r["coinc_acc"] for r in sub]):.3f}  '
              f'(sens={np.mean([r["coinc_sens"] for r in sub]):.3f} '
              f'spec={np.mean([r["coinc_spec"] for r in sub]):.3f})  '
              f'initial={np.nanmean([r["unk_acc_initial_guess"] for r in sub]):.3f}  '
              f'dynamics_coinc={np.nanmean([r["unk_acc_dynamics_coinc"] for r in sub]):.3f}  '
              f'numpy_conflict_mask={np.nanmean([r["unk_acc_numpy_conflict_mask"] for r in sub]):.3f}  '
              f'[{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('spiking_emergent_coincidence_full.csv', index=False)
    print('\nSaved spiking_emergent_coincidence_full.csv')
    return df


if __name__ == '__main__':
    main()
