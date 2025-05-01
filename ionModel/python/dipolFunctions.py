import numpy as np
from numba import njit, typed, types, prange

@njit(parallel=True, fastmath=True, cache=False)
def factorial(n):
    if n < 0:
        return 0
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

@njit(parallel=True, fastmath=True, cache=False)
def gamma(z):
    # Lanczos approximation for Gamma(z)
    g = 7
    p = [
        0.99999999999980993, 
        676.5203681218851, 
        -1259.1392167224028, 
        771.32342877765313,
        -176.61502916214059, 
        12.507343278686905,
        -0.13857109526572012, 
        9.9843695780195716e-6,
        1.5056327351493116e-7
    ]
    if z < 0.5:
        return np.pi / (np.sin(np.pi * z) * gamma(1 - z))
    else:
        z -= 1
        x = p[0]
        for i in range(1, len(p)):
            x += p[i] / (z + i)
        t = z + g + 0.5
        return np.sqrt(2 * np.pi) * t**(z + 0.5) * np.exp(-t) * x

@njit(parallel=True, fastmath=True, cache=False)
def binom(n, k):
    if k < 0 or k > n:
        return 0.0
    if k == 0 or k == n:
        return 1.0
    # Use symmetry
    if k > n - k:
        k = n - k
    result = 1.0
    for i in range(1, k + 1):
        result *= (n - (k - i))
        result /= i
    return result

@njit(parallel=True, fastmath=True, cache=False)
def hyp2f1(a, b, c, z, n_terms=50):
    # Series expansion for |z| < 1
    result = 0.0
    term = 1.0
    for n in range(n_terms):
        if n > 0:
            term *= (a + n - 1) * (b + n - 1) / ((c + n - 1) * n) * z
        result += term
        if np.abs(term) < 1e-15:
            break
    return result

@njit(parallel=True, fastmath=True, cache=False)
def sph_harm(m, l, phi, theta):
    abs_m = abs(m)
    norm = np.sqrt(
        ((2 * l + 1) / (4 * np.pi)) *
        factorial(l - abs_m) / factorial(l + abs_m)
    )
    # Condon-Shortley phase
    cs_phase = 1.0 if m >= 0 else (-1) ** abs_m
    # Value
    plm = legendre_p(l, abs_m, np.cos(theta))
    ylm = cs_phase * norm * plm * np.exp(1j * m * phi)
    return ylm

@njit(parallel=True, fastmath=True, cache=False)
def legendre_p(l, m, x):
    pmm = 1.0
    if m > 0:
        somx2 = np.sqrt((1.0 - x) * (1.0 + x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm *= -fact * somx2
            fact += 2.0
    if l == m:
        return pmm
    pmmp1 = x * (2 * m + 1) * pmm
    if l == m + 1:
        return pmmp1
    pll = 0.0
    for ll in range(m + 2, l + 1):
        pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m)
        pmm = pmmp1
        pmmp1 = pll
    return pmmp1