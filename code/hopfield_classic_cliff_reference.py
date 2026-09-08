"""
A genuine, first-hand demonstration of the classic Hopfield memory cliff, to compare
against our own (measured, not assumed) smooth-degradation curves.

Standard outer-product (Hebbian) rule on N=784 bipolar (+-1) neurons -- matched to
MNIST's pixel dimension for a size-comparable reference point. Textbook capacity is
~0.138*N =~ 108 patterns for this rule; well past that point, recall is known to
collapse catastrophically (a discontinuous cliff), unlike the smooth decay we measured
for the pseudo-inverse heteroassociation on real image data.

Patterns are RANDOM independent bipolar vectors (the standard, textbook setting for this
capacity formula) -- NOT MNIST images, since the formula assumes uncorrelated patterns;
using correlated real images would not test the textbook claim cleanly. This is a
reference/control, not a drop-in replacement for the image experiments elsewhere in this
project.
"""
import numpy as np
import pandas as pd

N_NEURONS = 784
M_VALUES = [20, 40, 60, 80, 100, 108, 120, 140, 160, 200, 300, 500]
NOISE_FRACTION = 0.05    # fraction of bits flipped in the recall cue
N_TRIALS = 40
MAX_ITERS = 15
SEED = 42


def train_hopfield(patterns):
    """patterns: (M, N) in {-1,+1}. Standard outer-product rule, zero diagonal."""
    M_local, N_local = patterns.shape
    W = (patterns.T @ patterns) / N_local
    np.fill_diagonal(W, 0.0)
    return W


def recall(W, cue, max_iters=MAX_ITERS):
    s = cue.copy()
    for _ in range(max_iters):
        s_new = np.sign(W @ s)
        s_new[s_new == 0] = 1
        if np.array_equal(s_new, s):
            break
        s = s_new
    return s


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    for M_local in M_VALUES:
        patterns = rng.choice([-1, 1], size=(M_local, N_NEURONS)).astype(np.float64)
        W = train_hopfield(patterns)

        nflip = int(round(NOISE_FRACTION * N_NEURONS))
        overlaps, exacts = [], []
        for _ in range(N_TRIALS):
            m = rng.integers(M_local)
            cue = patterns[m].copy()
            flip_idx = rng.choice(N_NEURONS, size=nflip, replace=False)
            cue[flip_idx] *= -1

            recovered = recall(W, cue)
            overlap = np.dot(recovered, patterns[m]) / N_NEURONS  # in [-1, 1]
            overlaps.append(overlap)
            exacts.append(int(np.array_equal(recovered, patterns[m])))

        mean_overlap = np.mean(overlaps)
        exact_rate = np.mean(exacts)
        rows.append({'M': M_local, 'load_ratio_M_over_N': M_local / N_NEURONS,
                     'mean_overlap': mean_overlap, 'exact_rate': exact_rate})
        print(f'M={M_local:4d}  (M/N={M_local/N_NEURONS:.3f})  '
              f'mean_overlap={mean_overlap:+.3f}  exact_rate={exact_rate:.2%}')

    df = pd.DataFrame(rows)
    df.to_csv('hopfield_classic_cliff_reference.csv', index=False)
    print('\nSaved hopfield_classic_cliff_reference.csv')
    print('\nTextbook capacity for this rule: ~0.138*N =~', round(0.138 * N_NEURONS), 'patterns')


if __name__ == '__main__':
    main()
