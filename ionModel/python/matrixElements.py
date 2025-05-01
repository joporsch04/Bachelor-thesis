import numpy as np
from scipy.special import gamma, hyp2f1, sph_harm, binom, factorial
import plotly.graph_objects as go

def regularized_hypergeometric2F1(a, b, c, z):
    return hyp2f1(a, b, c, z) / gamma(c)

def hydrogenic_momentum_wavefunction(l, n, m, p, theta_p, phi_p, Z=1):
    """
    Numerically evaluates the sum:
    Sum_{iota=0}^{2l+1} [ ... ] as in the Mathematica formula.
    """
    result = 0.0 + 0.0j
    for iota in range(0, 2*l + 2):  # Python range is exclusive at the end
        prefactor = ((-1)**iota) / factorial(iota)
        prefactor *= 2**(0.5 + iota)
        prefactor *= (1j * n)**l * n * (p**2)**(l/2) * Z**(-3 - l)
        prefactor *= binom(l + n, -1 - l + n - iota)
        prefactor *= np.sqrt(Z**3 * gamma(-l + n) / gamma(1 + l + n))
        prefactor *= gamma(3 + 2*l + iota)
        a = 2 + l + iota/2
        b = 0.5 * (3 + 2*l + iota)
        c = 1.5 + l
        z = - (n**2 * p**2) / Z**2
        hyp = regularized_hypergeometric2F1(a, b, c, z)
        Ylm = sph_harm(m, l, phi_p, theta_p)
        term = prefactor * hyp * Ylm
        result += term
    return result


fig = go.Figure()

p = np.linspace(0, 10, 100)
theta_p = np.linspace(0, np.pi, 100)
phi_p = np.linspace(0, 2*np.pi, 100)

result = hydrogenic_momentum_wavefunction(1, 1, 0, p, theta_p, phi_p)