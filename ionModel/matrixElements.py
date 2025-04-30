import numpy as np
from scipy.special import factorial, gamma, hyp2f1, sph_harm, binom
import cmath
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def regularized_hypergeometric2F1(a, b, c, z):
    return hyp2f1(a, b, c, z) / gamma(c)

def dipoleElement(l, n, px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    result = 0.0 + 0.0j
    for m in range(-l, l+1):
        # Pre-factors
        sqrt2 = np.sqrt(2)
        n_factor = (1/n)**(-3-l)
        sqrt_fact = np.sqrt(factorial(-1-l+n) / (n**4 * factorial(l+n)))
        theta = np.arccos(pz/p)
        phi = np.arctan2(py, px)
        Ylm = sph_harm(m, l, phi, theta)
        Yl1m = sph_harm(1+m, l, phi, theta) if (1+m <= l) else 0

        # First sum over iota
        sum1 = 0.0 + 0.0j
        for iota in range(0, -1-l+n+1):
            binomial = binom(l+n, -1-l+n-iota)
            gamma_term = gamma(3+2*l+iota)
            hyp = regularized_hypergeometric2F1(2+l+iota/2, 0.5*(3+2*l+iota), 1.5+l, -n**2*p**2)
            term = ((-2)**iota) * (1j)**l * (p**2)**(l/2) * binomial * gamma_term * hyp / factorial(iota)
            sum1 += term

        # Second sum over iota
        sum2 = 0.0 + 0.0j
        for iota in range(0, -1-l+n+1):
            binomial = binom(l+n, -1-l+n-iota)
            gamma_term = gamma(3+2*l+iota)
            hyp1 = regularized_hypergeometric2F1(2+l+iota/2, 0.5*(3+2*l+iota), 1.5+l, -n**2*p**2)
            hyp2 = regularized_hypergeometric2F1(3+l+iota/2, 1+0.5*(3+2*l+iota), 2.5+l, -n**2*p**2)
            term1 = ((-2)**iota) * (1j)**l * l * (p**2)**(-1+l/2) * pz * binomial * gamma_term * hyp1 / factorial(iota)
            term2 = ((-2)**iota) * (1j)**l * n**2 * (p**2)**(l/2) * pz * (2+l+iota/2) * (3+2*l+iota) * binomial * gamma_term * hyp2 / factorial(iota)
            sum2 += (term1 - term2)

        # Main expression
        prefactor1 = -1 / np.sqrt(1 - (pz**2)/(p**2))
        bracket1 = (1/np.sqrt(p**2) - (pz**2)/(p**3))
        termA = sqrt2 * n_factor * bracket1 * sqrt_fact
        part1 = (m * pz * Ylm) / (np.sqrt(p**2) * np.sqrt(1 - (pz**2)/(p**2)))
        part2 = (np.exp(-1j*phi) * np.sqrt(gamma(1+l-m)) * np.sqrt(gamma(2+l+m)) * Yl1m) / (np.sqrt(gamma(l-m)) * np.sqrt(gamma(1+l+m)))
        sum_part = (part1 + part2) * sum1

        termB = sqrt2 * n_factor * sqrt_fact * Ylm * sum2

        result += prefactor1 * termA * sum_part + termB

    return result

