"""
Multi-grid replication of the same-digit-sync / different-digit-desync effect, FIXED-WEIGHT
network (w_IE=0.005, w_EE=0.001, no plasticity) -- across several distinct solved Sudoku
grids. (The original single-grid demonstration used only pool[0]; this generalizes it,
and uses the non-plastic network that is the paper's actual reported mechanism.)
"""
import numpy as np
import pandas as pd
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
N_GRIDS = 8


def build_fresh():
    params = dict(DEFAULT_PARAMS)
    params['w_IE'] = W_IE
    params['w_EE'] = W_EE
    params['runtime_ms'] = WINDOW_MS
    params['drive_duration_ms'] = WINDOW_MS + 1
    net, comps = build_network(params=params)
    return net, comps, params


def run_trial(drive, seed):
    net, comps, params = build_fresh()
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


def main():
    pool = np.load(POOL_FILE)[:N_GRIDS]
    conflict_mask = build_conflict_mask(N).astype(bool)
    complement_mask = (~conflict_mask) & (~np.eye(N, dtype=bool))
    iu, ju = np.triu_indices(N, k=1)

    rows = []
    for gi in range(N_GRIDS):
        digits = pool[gi].reshape(N)
        drive = DESIGNATED_RANGE[digits - 1]
        trains = run_trial(drive, SEED + 999 + gi * 13)

        same_digit = (digits[:, None] == digits[None, :])
        same_digit_pairs = same_digit[iu, ju] & ~conflict_mask[iu, ju]
        conflicting_pairs = conflict_mask[iu, ju]
        noncon_diff_pairs = complement_mask[iu, ju] & ~same_digit[iu, ju]

        def offs(mask):
            out = []
            for p in np.where(mask)[0]:
                m, n = pair_phase_offset(trains[iu[p]], trains[ju[p]])
                if n >= 3 and not np.isnan(m):
                    out.append(m)
            return np.array(out)

        o_same = offs(same_digit_pairs); o_conf = offs(conflicting_pairs); o_non = offs(noncon_diff_pairs)
        row = {'grid': gi, 'same_digit_mean': o_same.mean(), 'same_digit_n': len(o_same),
               'conflicting_mean': o_conf.mean(), 'conflicting_n': len(o_conf),
               'noncon_diff_mean': o_non.mean(), 'noncon_diff_n': len(o_non)}
        rows.append(row)
        print(f'grid {gi}: same-digit={row["same_digit_mean"]:.2f}ms  '
              f'conflicting={row["conflicting_mean"]:.2f}ms  '
              f'noncon-diff={row["noncon_diff_mean"]:.2f}ms')

    df = pd.DataFrame(rows)
    df.to_csv('spiking_multigrid_replication_fixedweight.csv', index=False)
    print('\n=== SUMMARY across', N_GRIDS, 'grids (fixed-weight, w_IE=0.005, w_EE=0.001) ===')
    print(f'same-digit:      mean={df.same_digit_mean.mean():.3f} +- {df.same_digit_mean.std():.3f} ms')
    print(f'conflicting:     mean={df.conflicting_mean.mean():.3f} +- {df.conflicting_mean.std():.3f} ms')
    print(f'non-conf diff:   mean={df.noncon_diff_mean.mean():.3f} +- {df.noncon_diff_mean.std():.3f} ms')
    print('Saved spiking_multigrid_replication_fixedweight.csv')


if __name__ == '__main__':
    main()
