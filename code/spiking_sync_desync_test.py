"""
Isolated test of the synchrony/desynchrony hypothesis for the conflict-mask circuit,
BEFORE wiring anything back into the partial-cue pipeline.

S_IE.w = 0.005 (down from 0.02 -- may have been too strong before). Initial membrane
potentials are left RANDOM (the default +-0.1 perturbation), not zeroed out, per request.

IMPORTANT methodological fix vs. the first version of this test: the oscillation term in
the neuron equation (`I = osc_on*(0.5-0.2*cos(2*pi*10*Hz*t))`) depends on Brian2's ABSOLUTE
simulation clock t, which does not reset between net.run() calls on a reused Network. A
persistent network run across many trials therefore starts each trial at an uncontrolled,
drifting oscillation phase -- a real confound for anything comparing trials (it likely
explains some of the earlier "noise" attributed to the initial-v perturbation). Fix: build
a FRESH network per trial so every trial starts at the same absolute t=0 phase. Network
rebuild is cheap once Brian2's codegen cache is warm (~0.5-1s), so this is affordable for
a moderate number of trials.

Digit identity is tested as a RELATIVE, synchrony-based code, not decoded from one neuron's
absolute spike time against a fixed calibration table: same-digit cells (never conflict-mask
-connected, since two cells sharing a digit are by construction never in the same row/col/box)
should fire in sync when driven identically; conflict-mask-connected cells given the SAME
(ambiguous/wrong) drive should be pushed apart in time by inhibition; complement-mask
-connected (non-conflicting) cells given the same drive should be pulled together by
excitation (w_EE, off by default, turned on for this test).

Part 1: one real solved Sudoku, true digit-based drive. Sanity check: do the 9 same-digit
         neurons in each cluster fire at (approximately) the same time?

Part 2: random per-neuron drive (independent of Sudoku legality), same drive range, w_EE on,
         aggregated over multiple independent (fresh-network) trials. For same-driven pairs,
         do conflict-mask (inhibitory) pairs desynchronize while complement-mask (excitatory)
         pairs synchronize?
"""
import time
import numpy as np
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network

POOL_FILE = 'feedback_iterate_highM_pool.npy'
N = 81
SEED = 42

W_IE_TEST = 0.005
W_EE_TEST = 0.005          # symmetric first guess -- excitation was OFF (0.0) before
WINDOW_MS = 400.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])  # linspace(0.58,0.67,9)
N_PART2_TRIALS = 12


def build_fresh(w_ie, w_ee):
    params = dict(DEFAULT_PARAMS)
    params['w_IE'] = w_ie
    params['w_EE'] = w_ee
    params['runtime_ms'] = WINDOW_MS
    params['drive_duration_ms'] = WINDOW_MS + 1
    net, comps = build_network(params=params)
    return net, comps, params


def run_trial(w_ie, w_ee, drive, rng):
    net, comps, params = build_fresh(w_ie, w_ee)
    E = comps['E']
    low, high = params['init_perturb_low'], params['init_perturb_high']
    E.v = rng.random(N) * 0.1 + rng.uniform(low, high, size=N)
    comps['I'].v = 0.0
    E.I_d = drive
    E.osc_on = 1.0
    net.run(WINDOW_MS * ms)   # t_start is always 0 -- fresh network, controlled phase

    spkE = comps['spkE']
    t_all = np.asarray(spkE.t / ms); i_all = np.asarray(spkE.i)
    trains = [np.sort(t_all[i_all == n]) for n in range(N)]
    conflict_mask = comps['conflict_mask'].astype(bool)
    return trains, conflict_mask


def mean_spike_time(trains):
    return np.array([t.mean() if len(t) else np.nan for t in trains])


def first_spike_time(trains):
    return np.array([t[0] if len(t) else WINDOW_MS + 1.0 for t in trains])


def part1_cluster_sync(pool):
    print('=' * 70)
    print('PART 1: same-digit cluster synchrony sanity check (real Sudoku, real digits)')
    print(f'(fresh network, window={WINDOW_MS}ms, w_IE={W_IE_TEST}, w_EE={W_EE_TEST})')
    print('=' * 70)
    grid = pool[0]
    digits = grid.reshape(N)
    drive = DESIGNATED_RANGE[digits - 1]
    rng = np.random.default_rng(SEED)

    trains, _ = run_trial(W_IE_TEST, W_EE_TEST, drive, rng)
    mean_t = mean_spike_time(trains)
    first_t = first_spike_time(trains)
    counts = np.array([len(t) for t in trains])

    print(f'{"digit":>5} {"n_cells":>8} {"mean_count":>11} {"mean(firstT) ms":>16} '
          f'{"std(firstT) ms":>15}')
    within_std = []
    cluster_means = []
    for d in range(1, 10):
        idx = np.where(digits == d)[0]
        ft = first_t[idx]
        print(f'{d:>5} {len(idx):>8} {counts[idx].mean():>11.2f} {np.nanmean(ft):>16.2f} '
              f'{np.nanstd(ft):>15.2f}')
        within_std.append(np.nanstd(ft))
        cluster_means.append(np.nanmean(ft))

    overall_within_std = np.nanmean(within_std)
    span = max(cluster_means) - min(cluster_means)
    print(f'\nMean within-cluster std of first-spike time: {overall_within_std:.2f} ms')
    print(f'Across-cluster mean-first-spike-time span: {span:.2f} ms '
          f'(range {min(cluster_means):.2f}-{max(cluster_means):.2f} ms)')
    passed = overall_within_std < span / 2
    print(f'\n{"PASS" if passed else "FAIL"}: within-cluster std '
          f'{"<" if passed else ">="} half the across-cluster span.')
    return passed


def part2_sync_desync_causal_test(w_ee_test, n_trials=N_PART2_TRIALS, seed_offset=1, w_ie_test=W_IE_TEST):
    print()
    print('=' * 70)
    print(f'PART 2: causal sync/desync test with RANDOM drive, w_IE={w_ie_test}, w_EE={w_ee_test}')
    print(f'({n_trials} independent fresh-network trials, window={WINDOW_MS}ms)')
    print('=' * 70)

    rng = np.random.default_rng(SEED + seed_offset)
    dt_first_inhib, dt_first_excit, dt_first_uncon = [], [], []
    dt_mean_inhib, dt_mean_excit = [], []
    t0 = time.time()
    for trial_i in range(n_trials):
        digit_assignment = rng.integers(1, 10, size=N)   # RANDOM, ignores Sudoku legality
        drive = DESIGNATED_RANGE[digit_assignment - 1]
        trains, conflict_mask = run_trial(w_ie_test, w_ee_test, drive, rng)
        complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))

        first_t = first_spike_time(trains)
        mean_t = mean_spike_time(trains)

        same_drive = (digit_assignment[:, None] == digit_assignment[None, :])
        iu, ju = np.triu_indices(N, k=1)
        same_drive_pairs = same_drive[iu, ju]
        inhib_pairs = conflict_mask[iu, ju] & same_drive_pairs
        excit_pairs = complement_mask[iu, ju] & same_drive_pairs
        uncon_pairs = same_drive_pairs & ~inhib_pairs & ~excit_pairs

        dtf = np.abs(first_t[iu] - first_t[ju])
        dtm = np.abs(np.nan_to_num(mean_t[iu], nan=WINDOW_MS) -
                      np.nan_to_num(mean_t[ju], nan=WINDOW_MS))

        dt_first_inhib.append(dtf[inhib_pairs]); dt_first_excit.append(dtf[excit_pairs])
        dt_first_uncon.append(dtf[uncon_pairs])
        dt_mean_inhib.append(dtm[inhib_pairs]); dt_mean_excit.append(dtm[excit_pairs])
        print(f'  trial {trial_i+1}/{N_PART2_TRIALS}: '
              f'inhib_pairs={inhib_pairs.sum()} excit_pairs={excit_pairs.sum()} '
              f'uncon_pairs={uncon_pairs.sum()}  [{time.time()-t0:.1f}s cumulative]')

    dt_first_inhib = np.concatenate(dt_first_inhib)
    dt_first_excit = np.concatenate(dt_first_excit)
    dt_first_uncon = np.concatenate(dt_first_uncon)
    dt_mean_inhib = np.concatenate(dt_mean_inhib)
    dt_mean_excit = np.concatenate(dt_mean_excit)

    print(f'\nPooled over {N_PART2_TRIALS} trials:')
    for label, arr in [('inhibition-connected (should DESYNC)', dt_first_inhib),
                        ('excitation-connected (should SYNC)', dt_first_excit),
                        ('unconnected (baseline)', dt_first_uncon)]:
        print(f'  {label}: n={len(arr):5d}  |first-spike diff| mean={arr.mean():7.2f} ms  '
              f'median={np.median(arr):7.2f} ms  std={arr.std():7.2f} ms')

    print(f'\n  (mean-spike-time diff, same categories) inhib={dt_mean_inhib.mean():.2f} ms  '
          f'excit={dt_mean_excit.mean():.2f} ms')

    inhib_dt = dt_first_inhib.mean(); excit_dt = dt_first_excit.mean()
    uncon_dt = dt_first_uncon.mean() if len(dt_first_uncon) else float('nan')
    print(f'\nHypothesis check: inhibition desync ({inhib_dt:.2f} ms) vs '
          f'excitation desync ({excit_dt:.2f} ms) vs baseline ({uncon_dt:.2f} ms)')
    passed = inhib_dt > excit_dt
    print(f'{"PASS" if passed else "FAIL"}: inhibition-connected same-drive pairs desynchronize '
          f'more than excitation-connected same-drive pairs.')
    return passed, inhib_dt, excit_dt


def w_ee_scan():
    print()
    print('#' * 70)
    print('W_EE SCAN: each cell has ~60 excitatory (complement-mask) partners vs ~20')
    print('inhibitory (conflict-mask) partners -- equal per-synapse weight gives excitation')
    print('~3x the aggregate influence. Scanning w_EE (w_IE fixed at', W_IE_TEST, ') to find')
    print('where local desync/sync structure survives instead of being swamped by global sync.')
    print('#' * 70)
    print('\n--- fully uncoupled control (w_IE=0, w_EE=0): are conflict-mask-labeled and')
    print('complement-mask-labeled pairs different AT ALL absent any real coupling? ---')
    part2_sync_desync_causal_test(0.0, n_trials=6, seed_offset=100, w_ie_test=0.0)

    for w_ee in [0.0, 0.0005, 0.001, 0.0017, 0.003, 0.005]:
        part2_sync_desync_causal_test(w_ee, n_trials=6, seed_offset=100)


def main():
    pool = np.load(POOL_FILE)[:5]
    passed1 = part1_cluster_sync(pool)
    passed2, inhib_dt, excit_dt = part2_sync_desync_causal_test(W_EE_TEST)
    w_ee_scan()

    print('\n' + '=' * 70)
    print(f'Part 1 (cluster synchrony sanity check): {"PASSED" if passed1 else "FAILED"}')
    print(f'Part 2 @ w_EE={W_EE_TEST} (inhibition desync > excitation desync): '
          f'{"PASSED" if passed2 else "FAILED"}')


if __name__ == '__main__':
    main()
