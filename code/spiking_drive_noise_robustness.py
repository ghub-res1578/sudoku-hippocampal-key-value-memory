"""
Noise robustness of the spiking mechanism, complementing the partial-cue (missing-information)
results: rather than some cells being entirely unknown, every KNOWN cell's drive current is
corrupted with Gaussian jitter (sigma as a fraction of the inter-digit current spacing), and
unknown cells still use the numpy engram's own guess as before. Fixed clue_fraction=0.30
(mid-range, already well characterized elsewhere in the paper). Tests whether the emergent-
coincidence-matrix pipeline degrades gracefully under continuous current noise, and whether it
still improves on the noisy input it started from.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network
from hippocampus_drive_readout import (
    H, N, K, SEED as HSEED, TAU, relation, make_projection, make_engrams, drive_vectors,
    readout_attention, decode_digits, clean_engram,
)

SEED = 42
W_IE = 0.005
W_EE = 0.001
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
OSC_PERIOD_MS = 100.0
THETA_MS = 2.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])
DIGIT_SPACING = (DESIGNATED_RANGE.max() - DESIGNATED_RANGE.min()) / 8   # ~0.0225

POOL_FILE = 'feedback_iterate_highM_pool.npy'
M = 200
CLUE_FRACTION = 0.30
N_TRIALS = 20
NOISE_SIGMAS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05]   # in drive-current units


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

    trng = np.random.default_rng(SEED + 9999)
    n_known = max(1, int(round(CLUE_FRACTION * N)))
    targets = trng.choice(M, size=min(N_TRIALS, M), replace=False)

    # precompute the (noise-free) trial setup once -- known_mask, engram guess for unknowns,
    # true digits -- shared across all noise levels (paired design)
    trials = []
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
        base_drive = DESIGNATED_RANGE[digits_drive_input - 1]

        trials.append({'t': t, 'ti': ti, 'known_mask': known_mask, 'digits_true': digits_true,
                       'digits_drive_input': digits_drive_input, 'base_drive': base_drive})

    print('\n' + '=' * 78)
    print(f'DRIVE-NOISE ROBUSTNESS: clue_fraction={CLUE_FRACTION}, sigmas={NOISE_SIGMAS}')
    print('=' * 78)

    rows = []
    t0 = time.time()
    for sigma in NOISE_SIGMAS:
        rng_noise = np.random.default_rng(SEED + int(sigma * 100000) + 1)
        for trial in trials:
            known_mask = trial['known_mask']; digits_true = trial['digits_true']
            noisy_drive = trial['base_drive'].copy()
            noise = rng_noise.normal(0, sigma, size=N)
            noisy_drive[known_mask] += noise[known_mask]   # only corrupt KNOWN cells' currents
            noisy_drive = np.clip(noisy_drive, 0.0, None)

            trains = run_trial(noisy_drive, SEED + 999 + trial['t'] * 7 + trial['ti'])
            C_hat = build_emergent_coincidence(trains)
            h_clean = clean_engram(C_hat, W_SH, K)
            b_hat = readout_attention(H_keys, B_drive, h_clean, tau=TAU)
            digits_dyn = decode_digits(b_hat)

            unknown_mask = ~known_mask
            acc_input = np.mean(trial['digits_drive_input'][unknown_mask] == digits_true[unknown_mask])
            acc_dyn = np.mean(digits_dyn[unknown_mask] == digits_true[unknown_mask])
            exact_dyn = int(np.all(digits_dyn == digits_true))

            rows.append({'sigma': sigma, 'target': trial['t'],
                         'acc_input_noisy': acc_input, 'acc_dynamics': acc_dyn,
                         'exact_dynamics': exact_dyn})
        sub = [r for r in rows if r['sigma'] == sigma]
        print(f'sigma={sigma:.3f} ({sigma/DIGIT_SPACING:.2f}x digit spacing)  '
              f'acc_input={np.mean([r["acc_input_noisy"] for r in sub]):.3f}  '
              f'acc_dynamics={np.mean([r["acc_dynamics"] for r in sub]):.3f}  '
              f'exact={np.mean([r["exact_dynamics"] for r in sub]):.2%}  '
              f'[{time.time()-t0:.1f}s]')

    df = pd.DataFrame(rows)
    df.to_csv('spiking_drive_noise_robustness.csv', index=False)
    print('\nSaved spiking_drive_noise_robustness.csv')
    plot(df)
    return df


def plot(df):
    """Regenerates Figure 7 (graceful degradation under drive-current noise)."""
    s = df.groupby('sigma').agg(
        acc_input=('acc_input_noisy', 'mean'), acc_dynamics=('acc_dynamics', 'mean'),
        acc_dynamics_se=('acc_dynamics', lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()
    s['sigma_norm'] = s.sigma / DIGIT_SPACING

    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=150)
    ax.errorbar(s.sigma_norm, s.acc_dynamics, yerr=s.acc_dynamics_se, fmt='o-', color='#2f7350',
                label='dynamics pipeline (noisy drive), n=20', linewidth=1.8, markersize=6,
                capsize=3, zorder=3)
    ax.axhline(s.acc_input.iloc[0], color='#7c8697', linestyle='--', linewidth=1.5,
               label='noise-free symbolic guess (baseline)', zorder=2)
    ax.set_xlabel('Drive-current noise sigma (units of inter-digit current spacing)')
    ax.set_ylabel('Unknown-cell accuracy')
    ax.set_title(f'Graceful degradation under drive-current noise (clue fraction = {CLUE_FRACTION})')
    ax.grid(alpha=0.25)
    ax.legend(loc='lower left', fontsize=9)
    fig.tight_layout()
    fig.savefig('spiking_drive_noise_clean.png', dpi=150, bbox_inches='tight')
    print('Saved spiking_drive_noise_clean.png')


if __name__ == '__main__':
    main()
