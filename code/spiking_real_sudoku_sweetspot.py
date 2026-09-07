"""
Real-Sudoku raster + sync/desync analysis at the sweet spot (w_IE=0.005, w_EE=0.001),
steady state (long window, transient discarded).

Important structural note: in a REAL solved Sudoku, inhibition-connected (same row/col/box)
cells are NEVER the same digit -- so "same-drive + inhibited" pairs (the artificial random-
assignment test used to isolate the pure connection-type effect) cannot occur here. What DOES
occur, and is the natural real-puzzle analogue:
  - same-digit pairs (always either excitation-connected or unconnected) -- should synchronize
  - conflicting different-digit pairs (always inhibition-connected, by construction) -- already
    separated by their differing drive current; this asks whether inhibition adds EXTRA
    separation beyond what the current difference alone would produce
  - non-conflicting different-digit pairs (excitation-connected) -- a reference category
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network, build_conflict_mask

N = 81
SEED = 42
W_IE = 0.005
W_EE = 0.001
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
OSC_PERIOD_MS = 100.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])
POOL_FILE = 'feedback_iterate_highM_pool.npy'

PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']


def build_fresh(w_ie, w_ee):
    params = dict(DEFAULT_PARAMS)
    params['w_IE'] = w_ie
    params['w_EE'] = w_ee
    params['runtime_ms'] = WINDOW_MS
    params['drive_duration_ms'] = WINDOW_MS + 1
    net, comps = build_network(params=params)
    return net, comps, params


def run_trial(w_ie, w_ee, drive, seed):
    net, comps, params = build_fresh(w_ie, w_ee)
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
    trains = [np.sort(t_all[i_all == n]) for n in range(N)]
    return trains


def circular_diff(p1, p2, period=OSC_PERIOD_MS):
    d = np.abs(p1 - p2)
    return np.minimum(d, period - d)


def pair_phase_offset(train_i, train_j):
    t_i = np.sort(train_i[train_i >= STEADY_STATE_START_MS])
    t_j = np.sort(train_j[train_j >= STEADY_STATE_START_MS])
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


def main():
    pool = np.load(POOL_FILE)
    grid = pool[0]
    digits = grid.reshape(N)
    drive = DESIGNATED_RANGE[digits - 1]
    conflict_mask = build_conflict_mask(N).astype(bool)
    complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))

    print('Running real-Sudoku trial at sweet spot (w_IE=0.005, w_EE=0.001)...')
    trains = run_trial(W_IE, W_EE, drive, SEED + 999)

    # ---------------- Raster ----------------
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    order = np.argsort(digits)
    colors = [mcolors.to_rgb(PALETTE[(digits[i] - 1) % 9]) for i in order]
    axes[0].eventplot([trains[i] for i in order], lineoffsets=np.arange(N), colors=colors,
                       linelengths=0.8, linewidths=0.6)
    axes[0].axvline(STEADY_STATE_START_MS, color='gray', linestyle=':', linewidth=1.2)
    axes[0].set_ylabel('neuron (sorted by digit)')
    axes[0].set_title('Full 3000ms window, real Sudoku drive, sweet spot (w_IE=.005, w_EE=.001)')
    axes[0].set_xlim(0, WINDOW_MS)

    tail = WINDOW_MS - 500
    axes[1].eventplot([trains[i][trains[i] >= tail] for i in order], lineoffsets=np.arange(N),
                       colors=colors, linelengths=0.8, linewidths=1.0)
    for cyc_start in np.arange(tail, WINDOW_MS, OSC_PERIOD_MS):
        axes[1].axvline(cyc_start, color='lightgray', linewidth=0.5, zorder=0)
    axes[1].set_ylabel('neuron (sorted by digit)')
    axes[1].set_xlabel('time (ms)')
    axes[1].set_title('Zoomed: last 500ms (steady state), cycle boundaries marked')
    axes[1].set_xlim(tail, WINDOW_MS)

    fig.tight_layout()
    fig.savefig('spiking_real_sudoku_sweetspot_raster.png', dpi=150)
    print('Saved spiking_real_sudoku_sweetspot_raster.png')

    # ---------------- Sync/desync analysis ----------------
    print('\nComputing steady-state phase offsets by pair category...')
    same_digit = (digits[:, None] == digits[None, :])
    iu, ju = np.triu_indices(N, k=1)

    same_digit_pairs = same_digit[iu, ju] & ~conflict_mask[iu, ju]  # always true (never inhibited)
    conflicting_pairs = conflict_mask[iu, ju]                       # always different digit
    noncon_diff_digit_pairs = complement_mask[iu, ju] & ~same_digit[iu, ju]

    def offsets_for(mask):
        offs = []
        idx = np.where(mask)[0]
        for p in idx:
            m, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
            if n >= 3 and not np.isnan(m):
                offs.append(m)
        return np.array(offs)

    same_digit_offsets = offsets_for(same_digit_pairs)
    conflicting_offsets = offsets_for(conflicting_pairs)
    noncon_diff_offsets = offsets_for(noncon_diff_digit_pairs)

    for label, offs in [('same-digit (excit/unconnected -- should SYNC)', same_digit_offsets),
                         ('conflicting different-digit (inhibited, real neighbors)', conflicting_offsets),
                         ('non-conflicting different-digit (excited)', noncon_diff_offsets)]:
        if len(offs) == 0:
            print(f'  {label}: no pairs'); continue
        print(f'  {label}: n={len(offs):4d}  mean={offs.mean():6.2f}ms  median={np.median(offs):6.2f}ms  '
              f'p10={np.percentile(offs,10):5.2f}  p90={np.percentile(offs,90):5.2f}')

    np.savez('spiking_real_sudoku_sweetspot_offsets.npz',
             same_digit=same_digit_offsets, conflicting=conflicting_offsets,
             noncon_diff=noncon_diff_offsets)
    print('\nSaved spiking_real_sudoku_sweetspot_offsets.npz')


if __name__ == '__main__':
    main()
