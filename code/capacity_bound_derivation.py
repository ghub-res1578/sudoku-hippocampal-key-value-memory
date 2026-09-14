"""
Reproduces the approximate capacity bound of Section 2.3 ("A division of labor between sensing
and memory, and a combinatorial capacity bound") and validates it against Table 1's empirically
measured collapse thresholds (feedback_vs_M_sweep.py / feedback_vs_K_sweep.py and friends).

Model: each of M stored keys is an independent, uniformly random K-sparse binary vector in H
dimensions. A query formed by flipping a fraction eta of all H engram bits (this project's noise
model throughout) has expected active-unit count K_q(eta) = K(1-2*eta) + eta*H, and similarity
S_true(eta) = K(1-eta) to its own true target. Similarity to any other, unrelated stored key is
hypergeometric with mean mu_other and variance sigma_other^2 (exact hypergeometric formulas, not
a Gaussian approximation, used for computing these two moments). Retrieval fails when the LARGEST
of the M-1 competing similarities exceeds the true one -- an extreme-value criterion approximated
via the standard leading-order expectation for the max of n i.i.d. Gaussians, mu + sigma*sqrt(2 ln n).
Solving S_true(eta) = E[max of M-1] for eta gives the predicted collapse threshold.

This is presented in the paper as an approximate, honestly-caveated bound, not a proof: the
Gaussian approximation to the hypergeometric tail and the leading-order-only extreme-value
asymptotic are both known sources of finite-n bias (largest at eta=0, where the independently
measured similarity ratios of the separation-operator ablation are over-predicted), and the
independence assumption is false by construction (real engram keys are correlated through the
shared random projection W_SH and through Sudoku's own combinatorial structure).
"""
import numpy as np

H_DEFAULT = 1000


def hypergeom_overlap_mean_var(H, Ka, Kb):
    """Mean/variance of the overlap between two random subsets of sizes Ka, Kb in [H]."""
    mean = Ka * Kb / H
    if H <= 1:
        return mean, 0.0
    var = Ka * (Kb / H) * (1 - Kb / H) * ((H - Ka) / (H - 1))
    return mean, var


def predict_Mmax(H, K, eta):
    """Approximate capacity bound M_max(eta) at fixed H, K (Eq. 6 in the paper)."""
    Kq = K * (1 - 2 * eta) + eta * H
    S_true = K * (1 - eta)
    mu_other, var_other = hypergeom_overlap_mean_var(H, Kq, K)
    sigma_other = np.sqrt(max(var_other, 1e-12))
    z = (S_true - mu_other) / sigma_other
    if z <= 0:
        return 0.0, z
    return 1 + np.exp(z**2 / 2), z


def collapse_threshold(H, K, target_M, eta_grid=None):
    """Smallest eta (on a fine grid) at which the predicted M_max drops to/below target_M."""
    if eta_grid is None:
        eta_grid = np.arange(0.05, 0.60, 0.001)
    for eta in eta_grid:
        Mmax, _ = predict_Mmax(H, K, eta)
        if Mmax <= target_M:
            return eta
    return None


def main():
    print('=== Sanity check against the separation-operator ablation\'s measured similarity ratios ===')
    print('(H=1000, K=150, M=200 -- these ratios motivated the extreme-value step in the derivation)')
    H, K, M = 1000, 150, 200
    for eta, measured_ratio in [(0.0, 3.39), (0.20, 1.92), (0.30, 1.45)]:
        Kq = K * (1 - 2 * eta) + eta * H
        S_true = K * (1 - eta)
        mu_other, var_other = hypergeom_overlap_mean_var(H, Kq, K)
        sigma_other = np.sqrt(var_other)
        n = M - 1
        Emax = mu_other + sigma_other * np.sqrt(2 * np.log(n))
        print(f'  eta={eta:.2f}  predicted ratio S_true/E[max]={S_true/Emax:.2f}  '
              f'measured={measured_ratio}')

    print('\n=== Predicted vs. observed collapse threshold (Table 1 in the paper) ===')
    print(f'{"":30s} {"M=50":>8s} {"M=1000":>8s} {"M=10,000":>10s} {"K=10":>8s} {"K=150":>8s}')
    pred = [
        collapse_threshold(1000, 150, 50),
        collapse_threshold(1000, 150, 1000),
        collapse_threshold(1000, 150, 10000),
        collapse_threshold(1000, 10, 1000),
        collapse_threshold(1000, 150, 1000),
    ]
    print(f'{"Predicted":30s}' + ''.join(f'{p*100:7.1f}%' for p in pred))
    observed = [45, 45, 40, 25, None]
    print(f'{"Observed (Table 1)":30s}' + '   45.0%   45.0%     40.0%   25.0%   40-45%')


if __name__ == '__main__':
    main()
