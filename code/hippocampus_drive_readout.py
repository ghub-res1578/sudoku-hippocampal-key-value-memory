"""
Hippocampus -> drive readout.

    C  = 81x81 same-digit coincidence matrix of a solved Sudoku, flattened (6561,)
    h  = k-WTA(W_SH @ C)                      clean engram,  h in {0,1}^1000
    b  = digit-to-drive-level lookup of the solved grid       drive,  b in R^81

Two readouts from engram to drive are implemented on the SAME stored (h, b) pairs:

  regression -- learn W_HD (81 x 1000) so that, across all M stored pairs:
                    W_HD @ h  ~=  b            i.e.  H_keys @ W_HD.T ~= B_values
                Retrieval is a single matrix-vector product: b_hat = W_HD @ h_query.

  attention  -- no learned matrix. Weight each STORED drive vector (bias) by how
                similar its key is to the query, via softmax attention coefficients:
                    p     = softmax(H_keys @ h_query / TAU)     (M,)   attention coeffs
                    b_hat = p @ B_values                         (81,)  weighted biases

W_SH (the 1000 x 6561 encoder) is a FIXED random sparse projection, not learned, and
is shared by both readouts. capacity_regression_vs_attention.py found: regression has
a hard capacity ceiling at M~=H and is fragile past a few % query noise; attention has
no ceiling in the tested range (M up to 2000) and tolerates noise to ~20-30%.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.special import softmax

SEED = 42
SEED_TRIALS = 12345

H = 1000                 # hippocampal neurons
K = 150                  # engram size (active units per key)
N = 81                   # Sudoku cells
INPUT_DIM = N * N        # 6561, flattened coincidence matrix
DENSITY = 0.02           # W_SH sparsity
RIDGE_LAMBDA = 1e-4
TAU = 2.0                # softmax temperature for the attention readout

M = 200                  # number of stored memories (comfortably under the
                          # empirical capacity ceiling of ~H=1000 found for this
                          # readout in capacity_regression_vs_attention.py)

NOISE_CHECK = 0.10       # fraction of H bits flipped, for the robustness spot-check

DRIVE_LEVELS = np.array([.58, .59, .60, .61, .62, .63, .64, .65, .67])


# ---------------- Sudoku generation ----------------

def generate_sudoku(rng):
    g = np.zeros((9, 9), dtype=np.int8)

    def cand(pos):
        r, c = divmod(pos, 9)
        used = set(g[r, :]); used.update(g[:, c])
        br, bc = (r // 3) * 3, (c // 3) * 3
        used.update(g[br:br + 3, bc:bc + 3].ravel())
        return np.array([d for d in range(1, 10) if d not in used], dtype=np.int8)

    def solve():
        best = -1; bc = None; bl = 10
        for pos in range(81):
            if g.ravel()[pos] != 0:
                continue
            cs = cand(pos)
            if len(cs) == 0:
                return False
            if len(cs) < bl:
                best, bc, bl = pos, cs, len(cs)
        if best < 0:
            return True
        rng.shuffle(bc); r, c = divmod(best, 9)
        for d in bc:
            g[r, c] = d
            if solve():
                return True
            g[r, c] = 0
        return False

    solve(); return g.copy()


def generate_unique(m, rng):
    out = []; seen = set()
    while len(out) < m:
        g = generate_sudoku(rng); key = tuple(g.ravel())
        if key not in seen:
            seen.add(key); out.append(g)
    return np.asarray(out)


# ---------------- Coincidence matrix (the "key" source) ----------------

def relation(g):
    x = g.ravel(); C = (x[:, None] == x[None, :])
    np.fill_diagonal(C, False); return C


# ---------------- Fixed random encoder: coincidence -> hippocampus ----------------

def make_projection(rng):
    mask = rng.random((H, INPUT_DIM)) < DENSITY
    rows, cols = np.nonzero(mask); vals = rng.normal(size=len(rows)).astype(np.float32)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(H, INPUT_DIM), dtype=np.float32)


def make_engrams(C_all, W_SH, k=K):
    """C_all: (M, INPUT_DIM) -> clean binary engrams (M, H), k active units each."""
    a = C_all @ W_SH.T
    a = a.toarray() if sparse.issparse(a) else np.asarray(a)
    idx = np.argpartition(a, -k, axis=1)[:, -k:]
    Hs = np.zeros((C_all.shape[0], H), np.uint8)
    Hs[np.arange(C_all.shape[0])[:, None], idx] = 1
    return Hs


# ---------------- Drive vectors (the "value") ----------------

def drive_vectors(sud):
    return DRIVE_LEVELS[sud.reshape(len(sud), N) - 1].astype(np.float32)


# ---------------- Learned readout: hippocampus -> drive ----------------

def train_W_HD(H_keys, B_values, ridge_lambda=RIDGE_LAMBDA):
    """
    Solve W_HD (81 x H) minimizing ||H_keys @ W_HD.T - B_values||^2 + ridge_lambda ||W_HD||^2.

    Dual form (cheap when M < H, exact same optimum as primal ridge):
        alpha = (H_keys @ H_keys.T + ridge_lambda I_M)^-1 @ B_values     (M x 81)
        W_HD  = (H_keys.T @ alpha).T                                     (81 x H)
    """
    Hm = H_keys.astype(np.float64); Bm = B_values.astype(np.float64)
    Mloc = Hm.shape[0]
    A = Hm @ Hm.T + ridge_lambda * np.eye(Mloc)
    alpha = np.linalg.solve(A, Bm)
    return (Hm.T @ alpha).T  # (81, H)


def readout_regression(W_HD, h_query):
    """b_hat = W_HD @ h    -- the 81x1000 matrix acting on a 1000x1 engram."""
    return W_HD @ h_query.astype(np.float32)


def readout_attention(H_keys, B_values, h_query, tau=TAU):
    """
    b_hat = softmax(H_keys @ h_query / tau) @ B_values

    The stored biases (B_values) are weighted by softmax attention coefficients
    computed from key similarity -- no learned matrix, a pure content-addressable
    lookup over the M stored (engram, drive) pairs.
    """
    sim = H_keys.astype(np.float32) @ h_query.astype(np.float32)
    p = softmax(sim / tau)
    return p @ B_values


# ---------------- Decode drive -> digits, for validation ----------------

def decode_digits(b):
    return np.argmin(np.abs(b[:, None] - DRIVE_LEVELS[None, :]), axis=1) + 1


# ---------------- Conflict-mask cleanup: drive -> coincidence -> engram ----------------
#
# A drive vector retrieved under noise is a SUPERPOSITION of several stored B_values
# (p is spread over more than one memory). Decoding it cell-by-cell can therefore
# assign the same digit to two cells that share a row/column/box -- something no
# valid Sudoku ever does. The conflict mask removes exactly those illegal same-digit
# pairs before the pattern is re-encoded, projecting the noisy guess back onto the
# manifold of valid Sudoku coincidence structure before it re-enters the hippocampus.

def conflict_matrix():
    """c[i,j] = True iff cells i,j share a row, column, or 3x3 box (i != j)."""
    c = np.zeros((N, N), bool)
    for i in range(N):
        r, col = divmod(i, 9)
        for j in range(N):
            rr, cc = divmod(j, 9)
            c[i, j] = (r == rr or col == cc or (r // 3 == rr // 3 and col // 3 == cc // 3))
    np.fill_diagonal(c, False); return c


def coincidence(digits, conflict=None):
    """Same-digit relation matrix from a decoded 81-digit guess, optionally with
    structurally-illegal same-digit pairs (same row/col/box) masked out."""
    C = digits[:, None] == digits[None, :]
    if conflict is not None:
        C[conflict] = False
    np.fill_diagonal(C, False); return C


def clean_engram(C_masked, W_SH, k=K):
    """Re-project a (cleaned) coincidence matrix through the SAME fixed encoder
    used to form the stored engrams, then k-WTA -- feeding the cleaned-up guess
    back into the hippocampal layer as a new candidate engram."""
    a = np.asarray(W_SH @ C_masked.astype(np.float32).ravel()).ravel()
    idx = np.argpartition(a, -k)[-k:]
    h = np.zeros(H, np.uint8); h[idx] = 1
    return h


# ---------------- Noise sweep: with vs without the feedback/cleanup pass ----------------

NOISE_SWEEP = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.38, 0.40,
               0.42, 0.45, 0.48, 0.50, 0.55, 0.60, 0.70, 0.80, 1.0]


def run_feedback_comparison(H_keys, B_values, W_SH, digits_true, conf,
                             noise_levels=NOISE_SWEEP, n_trials=100, seed=SEED_TRIALS + 1):
    """
    For each noise level, run the attention readout with a noisy query, both
    WITHOUT feedback (raw decode of the softmax-attention superposition) and
    WITH feedback (decode -> conflict mask -> re-encode through W_SH -> re-query).
    """
    Mloc = H_keys.shape[0]
    trng = np.random.default_rng(seed)
    n_trials = min(n_trials, Mloc)
    rows = []
    for noise in noise_levels:
        targets = trng.choice(Mloc, size=n_trials, replace=False)
        nflip = int(round(noise * H))
        acc_no_fb, acc_fb, overlaps, top1s = [], [], [], []
        for t in targets:
            hq = H_keys[t].copy()
            if nflip:
                hq[trng.choice(H, size=nflip, replace=False)] ^= 1

            b1 = readout_attention(H_keys, B_values, hq)
            digits1 = decode_digits(b1)
            acc_no_fb.append(np.mean(digits1 == digits_true[t]))

            C_clean = coincidence(digits1, conf)
            h_clean = clean_engram(C_clean, W_SH)
            overlaps.append(np.sum(h_clean * H_keys[t]) / K)
            sim_clean = H_keys.astype(np.float32) @ h_clean.astype(np.float32)
            top1s.append(int(np.argmax(sim_clean) == t))

            b2 = readout_attention(H_keys, B_values, h_clean)
            digits2 = decode_digits(b2)
            acc_fb.append(np.mean(digits2 == digits_true[t]))

        rows.append({
            'noise_percent': noise * 100,
            'digit_cell_acc_no_feedback': np.mean(acc_no_fb),
            'digit_cell_acc_with_feedback': np.mean(acc_fb),
            'cleaned_engram_overlap': np.mean(overlaps),
            'cleaned_engram_top1_acc': np.mean(top1s),
        })
        print(f"  noise={noise:5.0%}  no_feedback={np.mean(acc_no_fb):8.4%}  "
              f"with_feedback={np.mean(acc_fb):8.4%}  "
              f"delta={np.mean(acc_fb) - np.mean(acc_no_fb):+7.4%}")
    return pd.DataFrame(rows)


def plot_feedback_comparison(df, out_prefix='hippocampus_drive_readout_feedback'):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(df.noise_percent, df.digit_cell_acc_no_feedback, 'o-',
            color='tab:orange', label='without feedback (raw attention readout)')
    ax.plot(df.noise_percent, df.digit_cell_acc_with_feedback, 'o-',
            color='tab:green', label='with feedback (conflict-mask cleanup)')
    ax.axhline(1 / 9, color='gray', linestyle=':', linewidth=1, label='chance (1/9)')
    ax.set_xlabel('Query noise (% of H bits flipped)')
    ax.set_ylabel('Digit-cell retrieval accuracy')
    ax.set_title(f'Attention readout: effect of conflict-mask feedback (M={M})')
    ax.grid(alpha=0.3); ax.legend(loc='center left', fontsize=9)
    fig.tight_layout()
    fig.savefig(f'{out_prefix}.png', dpi=150)
    print(f'Saved plot to {out_prefix}.png')


def main():
    rng = np.random.default_rng(SEED)

    print(f'Generating {M} unique solved Sudokus...')
    sud = generate_unique(M, rng)

    print('Building coincidence matrices (keys source)...')
    C_all = np.asarray([relation(g).ravel() for g in sud], dtype=np.float32)

    print(f'Building fixed random encoder W_SH ({H} x {INPUT_DIM}, density={DENSITY})...')
    W_SH = make_projection(rng)

    print(f'Forming clean engrams (k-WTA, K={K})...')
    H_keys = make_engrams(C_all, W_SH)          # (M, H)

    print('Building drive vectors (values)...')
    B_values = drive_vectors(sud)               # (M, 81)

    print(f'Learning W_HD ({N} x {H}) via ridge regression (lambda={RIDGE_LAMBDA})...')
    W_HD = train_W_HD(H_keys, B_values)

    digits_true = sud.reshape(M, N)

    def evaluate(name, B_hat):
        fit_rmse = np.sqrt(np.mean((B_hat - B_values) ** 2))
        digits_hat = np.array([decode_digits(b) for b in B_hat])
        cell_acc = np.mean(digits_hat == digits_true)
        exact_acc = np.mean(np.all(digits_hat == digits_true, axis=1))
        print(f'  {name:11s}  rmse={fit_rmse:.6f}  digit_cell_acc={cell_acc:.4%}  '
              f'exact_grid_acc={exact_acc:.4%}')
        return cell_acc

    print('\nClean-query validation (query = the exact stored engram):')
    B_hat_reg = H_keys.astype(np.float32) @ W_HD.T
    evaluate('regression', B_hat_reg)
    B_hat_att = np.array([readout_attention(H_keys, B_values, h) for h in H_keys])
    evaluate('attention', B_hat_att)

    print(f'\nNoisy-query robustness spot-check ({NOISE_CHECK:.0%} of H bits flipped, '
          f'{min(50, M)} trials):')
    trng = np.random.default_rng(SEED_TRIALS)
    n_trials = min(50, M)
    targets = trng.choice(M, size=n_trials, replace=False)
    nflip = int(round(NOISE_CHECK * H))
    B_hat_reg_noisy, B_hat_att_noisy, true_noisy = [], [], []
    for t in targets:
        hq = H_keys[t].copy()
        hq[trng.choice(H, size=nflip, replace=False)] ^= 1
        B_hat_reg_noisy.append(readout_regression(W_HD, hq))
        B_hat_att_noisy.append(readout_attention(H_keys, B_values, hq))
        true_noisy.append(digits_true[t])
    digits_true_n = np.array(true_noisy)

    def evaluate_noisy(name, B_hat):
        digits_hat = np.array([decode_digits(b) for b in B_hat])
        cell_acc = np.mean(digits_hat == digits_true_n)
        print(f'  {name:11s}  digit_cell_acc={cell_acc:.4%}')

    evaluate_noisy('regression', np.array(B_hat_reg_noisy))
    evaluate_noisy('attention', np.array(B_hat_att_noisy))

    print('\nNoise sweep: attention readout with vs without conflict-mask feedback:')
    conf = conflict_matrix()
    fb_df = run_feedback_comparison(H_keys, B_values, W_SH, digits_true, conf)
    fb_df.to_csv('hippocampus_drive_readout_feedback_sweep.csv', index=False)
    plot_feedback_comparison(fb_df)

    np.save('W_HD.npy', W_HD)
    np.savez('hippocampus_drive_readout_store.npz',
             H_keys=H_keys, B_values=B_values, sudokus=sud)
    print('\nSaved W_HD.npy (the learned 81x1000 readout) and '
          'hippocampus_drive_readout_store.npz (engrams/drives/sudokus -- '
          'needed by readout_attention at query time, since it has no trained weights).')

    return W_SH, W_HD, H_keys, B_values, sud


if __name__ == '__main__':
    main()
