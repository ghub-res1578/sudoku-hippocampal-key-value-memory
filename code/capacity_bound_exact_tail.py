"""
Exact hypergeometric tail bound for the sparse combinatorial key-value memory (Section 2.3 of
the main paper; full derivation and proofs in paper/capacity_bound_supplement.tex, Section 8).

Where capacity_bound_derivation.py approximates retrieval failure via the *expected* maximum of
M-1 competing similarities (a first-moment approximation), this script computes the failure
PROBABILITY itself, exactly, from the hypergeometric distribution -- no Gaussian approximation,
no extreme-value asymptotics.

Model: query corruption flips a fraction eta of all H engram bits. R = number of the true key's
K active bits that are among the flipped ones is exactly Hypergeometric(H, K, eta*H). Conditional
on the realized R=r, BOTH the query's active-bit count k(r) = K + eta*H - 2r AND its similarity
to the true key s(r) = K - r are deterministic -- they are perfectly correlated through the same
r. (A naive shortcut that holds s(r) fixed at its unconditional mean K(1-eta) while still letting
k(r) vary with r breaks this correlation and understates the true failure probability by 4-5
orders of magnitude for eta in [0.30, 0.40] -- see capacity_bound_supplement.tex Section 8.1.)

p_k(s) = P(Hypergeometric(H, k, K) >= s) is the exact probability that one specific, independent
other stored key has similarity >= s to a k-sparse query. The exact unconditional failure
probability, averaging over R, is
    P_fail(M, eta) = sum_r P(R=r) * (1 - (1 - p_{k(r)}(s(r)))**(M-1))
A pointwise union bound (Bernoulli's inequality, applied before averaging over r) gives a
rigorous sufficient bound: with Pbar(eta) := E_R[p_{k(R)}(s(R))],
    M_max(eta, delta) = 1 + floor(delta / Pbar(eta))
guarantees P_fail(M, eta) <= delta exactly, for any finite M -- unlike the Gaussian expected-
maximum formula, which only matches a first moment.
"""
import numpy as np
from scipy.stats import hypergeom

H_DEFAULT = 1000
K_DEFAULT = 150


def r_distribution(H, K, eta):
    """Pr(R=r) for r=0..K, R ~ Hypergeometric(H, K, eta*H)."""
    flips = int(round(eta * H))
    r_vals = np.arange(0, K + 1)
    probs = hypergeom.pmf(r_vals, H, K, flips)
    return r_vals, probs


def p_tail(H, k, K, s):
    """P(Hypergeometric(H, k, K) >= s) -- exact tail probability for one competitor."""
    if s > min(k, K):
        return 0.0
    if s <= 0:
        return 1.0
    return hypergeom.sf(s - 1, H, k, K)


def pbar(H, K, eta):
    """Pbar(eta) = E_R[p_{k(R)}(s(R))], the per-realization tail probability averaged over R."""
    r_vals, r_probs = r_distribution(H, K, eta)
    total = 0.0
    for r, pr in zip(r_vals, r_probs):
        if pr == 0:
            continue
        k = int(round(K + eta * H - 2 * r))
        s = int(round(K - r))
        total += pr * p_tail(H, k, K, s)
    return total


def exact_pfail(H, K, eta, M):
    """Exact P_fail(M, eta) -- no union bound, no Gaussian approximation."""
    r_vals, r_probs = r_distribution(H, K, eta)
    total = 0.0
    n = M - 1
    for r, pr in zip(r_vals, r_probs):
        if pr == 0:
            continue
        k = int(round(K + eta * H - 2 * r))
        s = int(round(K - r))
        p = p_tail(H, k, K, s)
        fail_given_r = 1.0 if p >= 1.0 else -np.expm1(n * np.log1p(-p))
        total += pr * fail_given_r
    return total


def mmax_exact(H, K, eta, delta):
    """Rigorous sufficient bound: M_max(eta, delta) = 1 + floor(delta / Pbar(eta))."""
    pb = pbar(H, K, eta)
    if pb <= 0:
        return float('inf')
    return 1 + np.floor(delta / pb)


def mmax_gaussian(H, K, eta):
    """Original expected-maximum approximation (capacity_bound_derivation.py), for comparison."""
    Kq = K * (1 - 2 * eta) + eta * H
    S_true = K * (1 - eta)
    mean_o = Kq * K / H
    var_o = Kq * (K / H) * (1 - K / H) * (H - Kq) / (H - 1)
    sigma_o = np.sqrt(max(var_o, 1e-12))
    z = (S_true - mean_o) / sigma_o
    if z <= 0:
        return 0.0
    return 1 + np.exp(z ** 2 / 2)


def main():
    H, K = H_DEFAULT, K_DEFAULT
    print('=== Table 1 (paper Section 2.3 / supplement Section 8.4): exact sufficient bound ===')
    print('=== vs. Gaussian expected-maximum approximation, H=1000, K=150 ===')
    print(f'{"eta":>6s} {"Pbar(eta)":>14s} {"Mmax(d=0.01)":>14s} {"Mmax(d=0.05)":>14s} {"Gaussian Mmax":>14s}')
    for eta in [0.20, 0.30, 0.35, 0.40, 0.42, 0.45]:
        pb = pbar(H, K, eta)
        m01 = mmax_exact(H, K, eta, 0.01)
        m05 = mmax_exact(H, K, eta, 0.05)
        mg = mmax_gaussian(H, K, eta)
        print(f'{eta:6.2f} {pb:14.4e} {m01:14.4e} {m05:14.4e} {mg:14.4e}')

    print('\n=== Table 2 (supplement Section 8.4): tightness of the sufficient bound ===')
    print('=== exact P_fail vs. sufficient-bound estimate (M-1)*Pbar(eta) ===')
    for eta in [0.30, 0.40, 0.42]:
        pb = pbar(H, K, eta)
        for M in [200, 1000, 4000]:
            exact = exact_pfail(H, K, eta, M)
            bound = (M - 1) * pb
            ratio = bound / exact if exact > 0 else float('nan')
            print(f'  eta={eta:.2f}  M={M:5d}  exact={exact:.4e}  bound={bound:.4e}  ratio={ratio:.2f}')


if __name__ == '__main__':
    main()
