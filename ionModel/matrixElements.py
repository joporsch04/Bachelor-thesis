import numpy as np
from scipy.special import factorial, gamma, hyp2f1, sph_harm, binom
import cmath
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def dipolElement_vec(n, l, m, p, thetap, phip):
    return np.array([phip_analytical_Dp(n, l, m, p, thetap, phip), phip_analytical_Dthetap(n, l, m, p, thetap, phip), phip_analytical_Dphip(n, l, m, p, thetap, phip)])

def phip_analytical_Dp(n, l, m, p, thetap, phip):
    prefactor = np.sqrt(2) * (1/n)**(-3 - l) * np.sqrt(gamma(-l + n) / (n**4 * gamma(1 + l + n)))
    sph = sph_harm(m, l, phip, thetap)
    sum_term = 0
    for iota in range(0, -1 - l + n + 1):
        binomial = binom(l + n, -1 - l + n - iota)
        gamma1 = gamma(3 + 2*l + iota)
        gamma2 = gamma(5 + 2*l + iota)
        denom = gamma(1 + iota)
        I_l = cmath.exp(1j * np.pi/2 * l)

        num1 = ((-2)**iota) * I_l * l * p**(-1 + l) * binomial * gamma1
        hyp1 = hyp2f1(2 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, -n**2 * p**2) / gamma(1.5 + l)
        term1 = num1 * hyp1 / denom

        num2 = ((-1)**iota) * I_l * 2**(-1 + iota) * n**2 * p**(1 + l) * binomial * gamma2
        hyp2 = hyp2f1(3 + l + iota/2, 1 + 0.5 * (3 + 2*l + iota), 2.5 + l, -n**2 * p**2) / gamma(2.5 + l)
        term2 = num2 * hyp2 / denom
        sum_term += (term1 - term2)
    return prefactor * sph * sum_term

def phip_analytical_Dthetap(n, l, m, p, thetap, phip):
    prefactor = (
        np.sqrt(2)
        * (1/n)**(-3 - l)
        * np.sqrt(gamma(-l + n) / (n**4 * factorial(l + n)))
    )
    sphY_lm = sph_harm(m, l, phip, thetap)
    angular = m * 1/np.tan(thetap) * sphY_lm
    if np.abs(gamma(l - m)) > 0 and np.abs(gamma(1 + l + m)) > 0:
        sphY_lmp1 = sph_harm(m + 1, l, phip, thetap)
        angular += (
            np.exp(-1j * phip)
            * np.sqrt(gamma(1 + l - m))
            * np.sqrt(gamma(2 + l + m))
            * sphY_lmp1
            / (np.sqrt(gamma(l - m)) * np.sqrt(gamma(1 + l + m)))
        )
    sum_term = 0
    for iota in range(0, -1 - l + n + 1):
        coeff = ((-2)**iota) * (cmath.exp(1j * np.pi/2 * l)) * (p**l)
        binomial = binom(l + n, -1 - l + n - iota)
        gamma_val = gamma(3 + 2*l + iota)
        a = 2 + l + iota/2
        b = 0.5 * (3 + 2*l + iota)
        c = 1.5 + l
        z = -n**2 * p**2
        hyp = hyp2f1(a, b, c, z) / gamma(c)
        denom = factorial(iota)
        sum_term += coeff * binomial * gamma_val * hyp / denom
    return prefactor * angular * sum_term

def phip_analytical_Dphip(n, l, m, p, thetap, phip):
    prefactor = (
        1j * np.sqrt(2) * m
        * (1/n)**(-3 - l)
        * np.sqrt(gamma(-l + n) / (n**4 * gamma(1 + l + n)))
    )
    sphY_lm = sph_harm(m, l, phip, thetap)
    sum_term = 0
    for iota in range(0, -1 - l + n + 1):
        coeff = ((-2)**iota) * (cmath.exp(1j * np.pi/2 * l)) * (p**l)
        binomial = binom(l + n, -1 - l + n - iota)
        gamma_val = gamma(3 + 2*l + iota)
        a = 2 + l + iota/2
        b = 0.5 * (3 + 2*l + iota)
        c = 1.5 + l
        z = -n**2 * p**2
        hyp = hyp2f1(a, b, c, z) / gamma(c)
        denom = factorial(iota)
        sum_term += coeff * binomial * gamma_val * hyp / denom
    return prefactor * sphY_lm * sum_term