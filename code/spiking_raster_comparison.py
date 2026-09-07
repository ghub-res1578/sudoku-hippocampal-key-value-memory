"""
Raster plots for the steady-state phase-lock question, across the conditions discussed:
weak inhibition only, strong inhibition only, the earlier "sweet spot" (w_IE=0.005,
w_EE=0.001), and the global-sync-collapse case (w_IE=0.005, w_EE=0.005).

Same random digit assignment reused across all conditions (fair comparison). One reference
cell (index 0) is tracked against a same-digit neighbor it is INHIBITED-connected to
(conflict mask) and a same-digit neighbor it is EXCITED-connected to (complement mask) --
these are the pair types the phase-lock test measured. All 81 neurons are shown, sorted by
digit and color-coded, with the three reference cells highlighted and labeled.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network

N = 81
SEED = 42
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])

CONDITIONS = [
    ('w_IE=0.005, w_EE=0 (weak inhib only)', 0.005, 0.0),
    ('w_IE=0.4, w_EE=0 (strong inhib only)', 0.4, 0.0),
    ('w_IE=0.005, w_EE=0.001 (sweet spot)', 0.005, 0.001),
    ('w_IE=0.005, w_EE=0.005 (global sync collapse)', 0.005, 0.005),
]

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
    return trains, comps['conflict_mask'].astype(bool)


def pick_reference_trio(seed):
    """Find a reference cell (index 0) with a same-digit inhibited partner AND a
    same-digit excited partner, using the conflict_mask (same for every trial)."""
    rng = np.random.default_rng(seed)
    from separated_edit_no_plasticity import build_conflict_mask
    conflict_mask = build_conflict_mask(N).astype(bool)
    complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))
    for attempt in range(200):
        digit_assignment = rng.integers(1, 10, size=N)
        ref = 0
        d0 = digit_assignment[ref]
        inhib_candidates = np.where(conflict_mask[ref] & (digit_assignment == d0))[0]
        excit_candidates = np.where(complement_mask[ref] & (digit_assignment == d0))[0]
        if len(inhib_candidates) and len(excit_candidates):
            return digit_assignment, ref, inhib_candidates[0], excit_candidates[0]
    raise RuntimeError('could not find a reference trio -- widen search')


def main():
    digit_assignment, ref, inhib_partner, excit_partner = pick_reference_trio(SEED + 777)
    print(f'Reference cell {ref} (digit={digit_assignment[ref]})')
    print(f'  inhibited same-digit partner: cell {inhib_partner}')
    print(f'  excited same-digit partner:   cell {excit_partner}')
    drive = DESIGNATED_RANGE[digit_assignment - 1]

    print('Running all conditions once, reusing results across the three figures...')
    all_trains = {}
    for label, w_ie, w_ee in CONDITIONS:
        print(f'  {label}...')
        trains, _ = run_trial(w_ie, w_ee, drive, SEED + 999)
        all_trains[label] = trains

    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(13, 4 * len(CONDITIONS)), sharex=True)

    for ax, (label, w_ie, w_ee) in zip(axes, CONDITIONS):
        trains = all_trains[label]

        order = np.argsort(digit_assignment)
        colors = [mcolors.to_rgb(PALETTE[(digit_assignment[i] - 1) % 9]) for i in order]
        raster_data = [trains[i] for i in order]
        ax.eventplot(raster_data, lineoffsets=np.arange(N), colors=colors, linelengths=0.8,
                     linewidths=0.6)

        rank_ref = int(np.where(order == ref)[0][0])
        rank_inhib = int(np.where(order == inhib_partner)[0][0])
        rank_excit = int(np.where(order == excit_partner)[0][0])
        for rank, name, color in [(rank_ref, f'ref (cell {ref})', 'black'),
                                   (rank_inhib, f'inhib partner (cell {inhib_partner})', 'red'),
                                   (rank_excit, f'excit partner (cell {excit_partner})', 'blue')]:
            ax.axhline(rank, color=color, linewidth=1.0, alpha=0.5, linestyle='--')
            ax.text(WINDOW_MS + 30, rank, name, color=color, fontsize=8, va='center')

        ax.axvline(STEADY_STATE_START_MS, color='gray', linestyle=':', linewidth=1.2)
        ax.text(STEADY_STATE_START_MS, N + 2, 'steady-state region starts',
                fontsize=8, color='gray', ha='left')
        ax.set_ylabel('neuron (sorted by digit)')
        ax.set_title(label, fontsize=11, loc='left')
        ax.set_xlim(0, WINDOW_MS + 220)
        ax.set_ylim(-1, N + 4)

    axes[-1].set_xlabel('time (ms)')
    fig.suptitle(f'Reference cell {ref} (digit {digit_assignment[ref]}) vs same-digit '
                 f'inhibited partner (cell {inhib_partner}) and excited partner '
                 f'(cell {excit_partner}) -- colors = the random digit assigned to each cell',
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig('spiking_raster_comparison.png', dpi=150)
    print('Saved spiking_raster_comparison.png')

    # ---- zoomed-in panel: just the 3 reference cells, full window, for precise comparison ----
    fig2, axes2 = plt.subplots(len(CONDITIONS), 1, figsize=(13, 2.0 * len(CONDITIONS)), sharex=True)
    for ax, (label, w_ie, w_ee) in zip(axes2, CONDITIONS):
        trains = all_trains[label]
        ax.eventplot([trains[ref], trains[inhib_partner], trains[excit_partner]],
                     lineoffsets=[2, 1, 0],
                     colors=['black', 'red', 'blue'], linelengths=0.7, linewidths=1.2)
        ax.axvline(STEADY_STATE_START_MS, color='gray', linestyle=':', linewidth=1.2)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([f'ref {ref}', f'inhib {inhib_partner}', f'excit {excit_partner}'],
                            fontsize=8)
        ax.set_title(label, fontsize=10, loc='left')
        ax.set_xlim(0, WINDOW_MS)
    axes2[-1].set_xlabel('time (ms)')
    fig2.suptitle('Zoomed: reference trio only, full 3000ms window', fontsize=11)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    fig2.savefig('spiking_raster_comparison_zoomed.png', dpi=150)
    print('Saved spiking_raster_comparison_zoomed.png')

    # ---- zoomed-in on just the steady-state tail (last 500ms) for the clearest read ----
    fig3, axes3 = plt.subplots(len(CONDITIONS), 1, figsize=(13, 2.0 * len(CONDITIONS)), sharex=True)
    for ax, (label, w_ie, w_ee) in zip(axes3, CONDITIONS):
        trains = all_trains[label]
        tail = WINDOW_MS - 500
        tr = [t[t >= tail] for t in (trains[ref], trains[inhib_partner], trains[excit_partner])]
        ax.eventplot(tr, lineoffsets=[2, 1, 0], colors=['black', 'red', 'blue'],
                     linelengths=0.7, linewidths=1.5)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([f'ref {ref}', f'inhib {inhib_partner}', f'excit {excit_partner}'],
                            fontsize=8)
        ax.set_title(label, fontsize=10, loc='left')
        ax.set_xlim(tail, WINDOW_MS)
        for cyc_start in np.arange(tail, WINDOW_MS, 100.0):
            ax.axvline(cyc_start, color='lightgray', linewidth=0.5, zorder=0)
    axes3[-1].set_xlabel('time (ms)  (thin gray lines = 100ms oscillation cycle boundaries)')
    fig3.suptitle('Zoomed: last 500ms only (steady state), cycle boundaries marked', fontsize=11)
    fig3.tight_layout(rect=[0, 0, 1, 0.97])
    fig3.savefig('spiking_raster_comparison_tail.png', dpi=150)
    print('Saved spiking_raster_comparison_tail.png')


if __name__ == '__main__':
    main()
