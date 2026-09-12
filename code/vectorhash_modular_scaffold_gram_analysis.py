"""
Reproduces the Gram-matrix condition number and effective-rank numbers cited in Section 4.8
("Does image dimensionality limit capacity here?"): condition number growing from 3.0e3
(M=100) to 2.8e12 (M=800), effective rank (99.9% energy) of only 387 at M=800. These numbers
existed only as an unsaved ad hoc computation when first reported; this script makes them
reproducible, matching this project's standard of an exact command for every number in the
paper.

The Gram matrix here is the raw MNIST image Gram matrix (images @ images.T, M x M) -- the
same matrix the pseudo-inverse encoder W_SI must invert to fit the modular grid codes
(vectorhash_modular_scaffold_comparison.py) or the flat engram (heteroassoc_* scripts). Its
condition number and effective rank are a property of the image data alone, independent of
which scaffold (flat or modular) is being fit to it -- which is the point: both scaffolds'
content-mapping step is bounded by the same ceiling.
"""
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, '.')
from heteroassoc_vectorhash_generalized import load_dataset

M_SWEEP = [100, 200, 400, 800]
ENERGY_THRESHOLD = 0.999


def main():
    rows = []
    for M in M_SWEEP:
        images, img_dim = load_dataset('mnist', M)
        G = images @ images.T
        cond = np.linalg.cond(G)
        s = np.linalg.svd(G, compute_uv=False)
        cumulative = np.cumsum(s) / np.sum(s)
        eff_rank = int(np.searchsorted(cumulative, ENERGY_THRESHOLD) + 1)
        rows.append({'M': M, 'condition_number': cond, 'effective_rank_99.9pct': eff_rank})
        print(f'M={M}: condition number={cond:.3e}, effective rank (99.9% energy)={eff_rank}')

    df = pd.DataFrame(rows)
    df.to_csv('vectorhash_modular_scaffold_gram_analysis.csv', index=False)
    print('\nSaved vectorhash_modular_scaffold_gram_analysis.csv')


if __name__ == '__main__':
    main()
