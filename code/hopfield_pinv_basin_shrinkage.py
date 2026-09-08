"""
Supplementary data point for Section 4.5: the pseudo-inverse Hopfield rule is
algebraically exact at zero noise (verified separately), but its basin of attraction
under a fixed 5% bit-flip cue shrinks fast as M grows. This traces out that shrinkage at
M=100/150/200 on the same bipolarized MNIST images used throughout Section 4.5 -- the
M=100 and M=200 points are also in hopfield_vs_our_network_M_sweep.csv (produced by
hopfield_vs_our_network_comparison.py); M=150 (the illustrative midpoint, 25% exact) is
reported only in the paper text and is reproduced here on its own for completeness.
"""
import numpy as np
import sys
sys.path.insert(0, '.')
from heteroassoc_vectorhash_generalized import load_dataset
from hopfield_vs_our_network_comparison import bipolarize, train_hopfield_pinv, hopfield_recall, flip_bits

M_VALUES = [100, 150, 200]
FLIP_FRACTION = 0.05
N_TRIALS = 20
SEED = 999


def main():
    for M in M_VALUES:
        images, _ = load_dataset('mnist', M)
        images_bp = bipolarize(images)
        W_pinv = train_hopfield_pinv(images_bp, ridge=1.0)
        rng = np.random.default_rng(SEED)
        exact_ct = 0
        for _ in range(N_TRIALS):
            m = rng.integers(M)
            p = images_bp[m]
            cue = flip_bits(p, FLIP_FRACTION, rng)
            recovered = hopfield_recall(W_pinv, cue)
            exact_ct += int(np.array_equal(recovered, p))
        print(f'M={M}: exact={exact_ct}/{N_TRIALS} = {exact_ct/N_TRIALS:.2%}')


if __name__ == '__main__':
    main()
