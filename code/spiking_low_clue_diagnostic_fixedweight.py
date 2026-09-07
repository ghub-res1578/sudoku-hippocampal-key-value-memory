"""
Low-clue-fraction diagnostic, FIXED-WEIGHT network (no STDP): why does the emergent
coincidence matrix's engram-activation strength degrade at low clue fractions? Measures,
per trial: total active (predicted-True) pairs in the emergent Ĉ vs the true count (324),
engram overlap with the true stored key, attention mass on the true target, and its rank
among the M stored memories.
"""
import numpy as np
import pandas as pd
from brian2 import ms
import sys
sys.path.insert(0, '.')
from separated_edit_no_plasticity import DEFAULT_PARAMS, build_network
from hippocampus_drive_readout import (
    H, N, K, SEED as HSEED, TAU, relation, make_projection, make_engrams, drive_vectors,
    readout_attention, decode_digits, clean_engram,
)
from scipy.special import softmax

SEED = 42
W_IE = 0.005
W_EE = 0.001
WINDOW_MS = 3000.0
STEADY_STATE_START_MS = 1500.0
OSC_PERIOD_MS = 100.0
THETA_MS = 2.0
DESIGNATED_RANGE = np.array(DEFAULT_PARAMS['cluster_drive_amplitudes'])
POOL_FILE = 'feedback_iterate_highM_pool.npy'
M = 200
CLUE_FRACTIONS = [0.19, 0.25, 0.30]
N_TRIALS = 8


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

    trng = np.random.default_rng(SEED + 12345)
    rows = []
    for clue_frac in CLUE_FRACTIONS:
        n_known = max(1, int(round(clue_frac * N)))
        targets = trng.choice(M, size=min(N_TRIALS, M), replace=False)
        for ti, t in enumerate(targets):
            known_idx = trng.choice(N, size=n_known, replace=False)
            known_mask = np.zeros(N, dtype=bool); known_mask[known_idx] = True
            digits_true = digits_true_all[t]
            C_true = relation(pool_full[t])
            true_active_pairs = int(C_true.sum() / 2)

            x = pool_full[t].ravel()
            Cp = (x[:, None] == x[None, :])
            Cp = Cp & (known_mask[:, None] & known_mask[None, :])
            np.fill_diagonal(Cp, False)
            a0 = np.asarray(W_SH @ Cp.astype(np.float32).ravel()).ravel()
            idx0 = np.argpartition(a0, -K)[-K:]
            h_q = np.zeros(H, np.uint8); h_q[idx0] = 1
            b_hat = readout_attention(H_keys, B_drive, h_q, tau=TAU)
            digits_engram_guess = decode_digits(b_hat)
            digits_drive_input = digits_engram_guess.copy()
            digits_drive_input[known_mask] = digits_true[known_mask]
            drive = DESIGNATED_RANGE[digits_drive_input - 1]

            trains = run_trial(drive, SEED + 999 + t * 7 + ti)
            C_hat = build_emergent_coincidence(trains)
            hat_active_pairs = int(C_hat.sum() / 2)

            def engram_diagnostics(C):
                h = clean_engram(C, W_SH, K)
                overlap = float(np.sum(h * H_keys[t]) / K)
                sim = H_keys.astype(np.float32) @ h.astype(np.float32)
                p = softmax(sim / TAU)
                attn_true = float(p[t])
                attn_rank = int((sim > sim[t]).sum())
                b = p @ B_drive
                digits_final = decode_digits(b)
                unk_acc = np.mean(digits_final[~known_mask] == digits_true[~known_mask])
                return overlap, attn_true, attn_rank, unk_acc

            ov_dyn, attn_dyn, rank_dyn, acc_dyn = engram_diagnostics(C_hat)
            ov_oracle, attn_oracle, rank_oracle, acc_oracle = engram_diagnostics(C_true)

            rows.append({
                'clue_fraction': clue_frac, 'target': int(t),
                'true_active_pairs': true_active_pairs, 'hat_active_pairs': hat_active_pairs,
                'engram_overlap_dyn': ov_dyn, 'attn_mass_true_dyn': attn_dyn, 'attn_rank_dyn': rank_dyn,
                'unk_acc_dyn': acc_dyn,
                'engram_overlap_oracle': ov_oracle, 'attn_mass_true_oracle': attn_oracle,
                'attn_rank_oracle': rank_oracle, 'unk_acc_oracle': acc_oracle,
            })
            print(f'clue={clue_frac:.2f} trial={ti+1}/{len(targets)}  '
                  f'active_pairs(true/hat)={true_active_pairs}/{hat_active_pairs}  '
                  f'overlap(dyn/oracle)={ov_dyn:.3f}/{ov_oracle:.3f}  '
                  f'attn_true(dyn/oracle)={attn_dyn:.4f}/{attn_oracle:.4f}  '
                  f'rank(dyn/oracle)={rank_dyn}/{rank_oracle}  '
                  f'unk_acc(dyn/oracle)={acc_dyn:.3f}/{acc_oracle:.3f}')
        sub = [r for r in rows if r['clue_fraction'] == clue_frac]
        print(f'--- clue={clue_frac:.2f} SUMMARY: '
              f'hat_active_pairs={np.mean([r["hat_active_pairs"] for r in sub]):.1f} (true=324)  '
              f'overlap_dyn={np.mean([r["engram_overlap_dyn"] for r in sub]):.3f}  '
              f'attn_true_dyn={np.mean([r["attn_mass_true_dyn"] for r in sub]):.4f}  '
              f'attn_rank_dyn={np.mean([r["attn_rank_dyn"] for r in sub]):.1f}  '
              f'unk_acc_dyn={np.mean([r["unk_acc_dyn"] for r in sub]):.3f}  '
              f'unk_acc_oracle={np.mean([r["unk_acc_oracle"] for r in sub]):.3f} ---')

    df = pd.DataFrame(rows)
    df.to_csv('spiking_low_clue_diagnostic_fixedweight.csv', index=False)
    print('\nSaved spiking_low_clue_diagnostic_fixedweight.csv')


if __name__ == '__main__':
    main()
