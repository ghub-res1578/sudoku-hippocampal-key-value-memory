"""
Correction to the earlier sync/desync test: that test measured raw first-spike time (and
whole-window mean spike time) over a short 400ms window -- only ~4 cycles of the 10Hz
oscillation. That statistic is dominated by TRANSIENT settling behavior (driven partly by
the random initial-v perturbation), not by the STEADY-STATE relationship the conflict-mask
mechanism should actually rely on.

The correct claim to test: two identically-driven, mutually INHIBITORY-connected neurons
should settle into a stable ANTI-PHASE lock at steady state (each cycle, consistently ~half
a period apart) -- a real, reproducible desynchronization, low-variance cycle to cycle, not
just "some spike-time difference averaged over a noisy transient". Two identically-driven,
EXCITATORY-connected neurons should settle into a stable IN-PHASE lock (near-zero, low-variance
offset). This IS the correct spiking analogue of the hard conflict mask: not "average timing
differs", but "the relative PHASE reliably locks apart (conflict) or together (compatible)".

Test: run long enough (several seconds) for transients to die out, discard the early cycles,
and measure each same-driven pair's phase offset (spike time mod oscillation period) using
only the LATE, steady-state spikes -- both its mean offset and its cycle-to-cycle std (a
genuine lock should have LOW std; transient/noise-dominated pairs would have high std).
"""
import time
import numpy as np
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network

N = 81
SEED = 42
W_IE_TEST = 0.005
WINDOW_MS = 3000.0          # 30 cycles at 10Hz -- long enough to separate transient from steady state
STEADY_STATE_START_MS = 1500.0   # discard the first 15 cycles as transient
OSC_PERIOD_MS = 100.0        # 10 Hz
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])


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
    net.run(WINDOW_MS * ms)
    spkE = comps['spkE']
    t_all = np.asarray(spkE.t / ms); i_all = np.asarray(spkE.i)
    trains = [np.sort(t_all[i_all == n]) for n in range(N)]
    return trains, comps['conflict_mask'].astype(bool)


def steady_state_phases(train):
    """Phase (ms within a 100ms cycle) of each spike after the transient window."""
    late = train[train >= STEADY_STATE_START_MS]
    return late % OSC_PERIOD_MS


def circular_diff(p1, p2, period=OSC_PERIOD_MS):
    d = np.abs(p1 - p2)
    return np.minimum(d, period - d)


def pair_phase_offset(train_i, train_j):
    """Mean and std of the cycle-by-cycle circular phase offset between two spike trains,
    using only steady-state spikes. Pairs cycles by nearest spike time (both fire ~once/cycle
    once entrained)."""
    ph_i = steady_state_phases(train_i)
    ph_j = steady_state_phases(train_j)
    if len(ph_i) == 0 or len(ph_j) == 0:
        return np.nan, np.nan, 0
    # match each spike in i to the nearest spike in j (by absolute time, not just phase,
    # to correctly pair same-cycle spikes rather than spurious phase-only matches)
    t_i = np.sort(train_i[train_i >= STEADY_STATE_START_MS])
    t_j = np.sort(train_j[train_j >= STEADY_STATE_START_MS])
    if len(t_i) == 0 or len(t_j) == 0:
        return np.nan, np.nan, 0
    offsets = []
    for t in t_i:
        nearest_j = t_j[np.argmin(np.abs(t_j - t))]
        if abs(nearest_j - t) < OSC_PERIOD_MS:  # same or adjacent cycle
            offsets.append(circular_diff(t % OSC_PERIOD_MS, nearest_j % OSC_PERIOD_MS))
    if len(offsets) < 3:
        return np.nan, np.nan, len(offsets)
    offsets = np.array(offsets)
    return offsets.mean(), offsets.std(), len(offsets)


def coincidence_window_test(w_ie, w_ee, n_trials=15, thresholds=None):
    """Test the refined hypothesis: not full anti-phase locking, but a SMALL coincidence
    window that cleanly separates inhibition-connected (desync, offset > theta) from
    excitation-connected (sync, offset <= theta) same-drive pairs. Saves the FULL per-pair
    offset distribution (not just pooled means) and scans classification accuracy over
    candidate theta."""
    if thresholds is None:
        thresholds = np.arange(0.5, 10.25, 0.25)

    rng = np.random.default_rng(SEED + 2024)
    inhib_offsets, excit_offsets = [], []
    t0 = time.time()
    for trial_i in range(n_trials):
        digit_assignment = rng.integers(1, 10, size=N)
        drive = DESIGNATED_RANGE[digit_assignment - 1]
        trains, conflict_mask = run_trial(w_ie, w_ee, drive, rng)
        complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))
        same_drive = (digit_assignment[:, None] == digit_assignment[None, :])

        iu, ju = np.triu_indices(N, k=1)
        same_drive_pairs = same_drive[iu, ju]
        inhib_pairs = np.where(conflict_mask[iu, ju] & same_drive_pairs)[0]
        excit_pairs = np.where(complement_mask[iu, ju] & same_drive_pairs)[0]

        rng_sub = np.random.default_rng(trial_i + 777)
        inhib_sample = rng_sub.choice(inhib_pairs, size=min(40, len(inhib_pairs)), replace=False)
        excit_sample = rng_sub.choice(excit_pairs, size=min(40, len(excit_pairs)), replace=False)

        for p in inhib_sample:
            m, s, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
            if n >= 3:
                inhib_offsets.append(m)
        for p in excit_sample:
            m, s, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
            if n >= 3:
                excit_offsets.append(m)
        print(f'  trial {trial_i+1}/{n_trials}  [{time.time()-t0:.1f}s cumulative]')

    inhib_offsets = np.array(inhib_offsets); excit_offsets = np.array(excit_offsets)
    print(f'\nw_IE={w_ie}, w_EE={w_ee}: n_inhib={len(inhib_offsets)}, n_excit={len(excit_offsets)}')
    print(f'  inhib offsets: mean={inhib_offsets.mean():.2f}  median={np.median(inhib_offsets):.2f}  '
          f'p10={np.percentile(inhib_offsets,10):.2f}  p90={np.percentile(inhib_offsets,90):.2f}')
    print(f'  excit offsets: mean={excit_offsets.mean():.2f}  median={np.median(excit_offsets):.2f}  '
          f'p10={np.percentile(excit_offsets,10):.2f}  p90={np.percentile(excit_offsets,90):.2f}')

    best = None
    print(f'\n  {"theta(ms)":>10} {"excit<=theta(sync)":>20} {"inhib>theta(desync)":>21} {"balanced_acc":>13}')
    for theta in thresholds:
        excit_correct = np.mean(excit_offsets <= theta)   # excitatory pairs correctly "synced"
        inhib_correct = np.mean(inhib_offsets > theta)     # inhibitory pairs correctly "desynced"
        bal_acc = (excit_correct + inhib_correct) / 2
        print(f'  {theta:10.2f} {excit_correct:20.2%} {inhib_correct:21.2%} {bal_acc:13.2%}')
        if best is None or bal_acc > best[1]:
            best = (theta, bal_acc, excit_correct, inhib_correct)

    print(f'\nBest theta={best[0]:.2f}ms: balanced accuracy={best[1]:.2%} '
          f'(excit sync-detect={best[2]:.2%}, inhib desync-detect={best[3]:.2%})')
    return inhib_offsets, excit_offsets, best


def scan_w_ie(w_ie_values, w_ee=0.0, n_trials=8):
    print('=' * 78)
    print('STEADY-STATE PHASE LOCK TEST -- scanning w_IE (w_EE held at', w_ee, ')')
    print(f'window={WINDOW_MS}ms, steady-state region >= {STEADY_STATE_START_MS}ms')
    print('=' * 78)

    for w_ie in w_ie_values:
        rng = np.random.default_rng(SEED + 500)
        inhib_means, inhib_stds, excit_means, excit_stds = [], [], [], []
        t0 = time.time()
        for trial_i in range(n_trials):
            digit_assignment = rng.integers(1, 10, size=N)
            drive = DESIGNATED_RANGE[digit_assignment - 1]
            trains, conflict_mask = run_trial(w_ie, w_ee, drive, rng)
            complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))
            same_drive = (digit_assignment[:, None] == digit_assignment[None, :])

            iu, ju = np.triu_indices(N, k=1)
            same_drive_pairs = same_drive[iu, ju]
            inhib_pairs = np.where(conflict_mask[iu, ju] & same_drive_pairs)[0]
            excit_pairs = np.where(complement_mask[iu, ju] & same_drive_pairs)[0]

            # subsample for speed (pairwise loop is O(pairs)); this is diagnostic, not final
            rng_sub = np.random.default_rng(trial_i)
            inhib_sample = rng_sub.choice(inhib_pairs, size=min(30, len(inhib_pairs)), replace=False)
            excit_sample = rng_sub.choice(excit_pairs, size=min(30, len(excit_pairs)), replace=False)

            for p in inhib_sample:
                m, s, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
                if n >= 3:
                    inhib_means.append(m); inhib_stds.append(s)
            for p in excit_sample:
                m, s, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
                if n >= 3:
                    excit_means.append(m); excit_stds.append(s)

        im, ist = np.array(inhib_means), np.array(inhib_stds)
        em, est = np.array(excit_means), np.array(excit_stds)
        print(f'\nw_IE={w_ie}  [{time.time()-t0:.1f}s]  n_inhib_pairs={len(im)}  n_excit_pairs={len(em)}')
        if len(im):
            print(f'  inhibition-connected: mean phase offset={im.mean():6.2f}ms  '
                  f'(pair-level std of offset={ist.mean():6.2f}ms)  '
                  f'  [anti-phase target = {OSC_PERIOD_MS/2:.0f}ms]')
        if len(em):
            print(f'  excitation-connected: mean phase offset={em.mean():6.2f}ms  '
                  f'(pair-level std of offset={est.mean():6.2f}ms)  [in-phase target = 0ms]')


if __name__ == '__main__':
    print('=' * 78)
    print('COINCIDENCE-WINDOW CLASSIFIER TEST')
    print('Not full anti-phase locking -- testing whether a SMALL coincidence window')
    print('separates inhib-connected (desync) from excit-connected (sync) same-drive pairs.')
    print('=' * 78)
    for w_ie, w_ee, label in [(0.005, 0.0, 'weak inhib only'),
                               (0.005, 0.0005, 'weak inhib + light excit'),
                               (0.005, 0.001, 'sweet spot'),
                               (0.4, 0.0, 'strong inhib only')]:
        print(f'\n--- {label}: w_IE={w_ie}, w_EE={w_ee} ---')
        coincidence_window_test(w_ie, w_ee, n_trials=15)
