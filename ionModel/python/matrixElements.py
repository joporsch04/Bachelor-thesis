import numpy as np
import pandas as pd
from scipy.special import gamma, binom, hyp2f1, sph_harm, factorial
from scipy.interpolate import interp1d
import plotly.graph_objects as go

#print(get_coefficients(4)[1,:])

def get_eigenEnergy(excitedStates):
    return np.array([0.5*1/(i**2) for i in range(1, excitedStates+1)])

def get_coefficients(excitedStates, t_grid):
    df=pd.read_csv("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0012/expec", sep='\s+', header=8)
    #shift every column name by one and remove the first column
    df.columns = df.columns[1:].tolist() + [""]
    #remove last column
    df = df.iloc[:, :-1]

    time = np.array(df["Time"])

    c_list = []
    for i in range(excitedStates):
        #c = np.array(df[f"Re{{<H0:{i}|psi>}}"]) + np.array(df[f"Imag{{<H0:{i}|psi>}}"]) * 1j
        c = np.array(df["<Occ{H0:0}>"])
        interp_real = interp1d(time, c.real, kind='cubic', fill_value="extrapolate")
        interp_imag = interp1d(time, c.imag, kind='cubic', fill_value="extrapolate")
        c_interp = interp_real(t_grid) + 1j * interp_imag(t_grid)
        c_list.append(c_interp)
    return np.vstack(c_list)

def hydrogen_state_generator(n_max):
    for n in range(1, n_max + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                yield (n, l, m)

def get_hydrogen_states(maxStates):
    gen = hydrogen_state_generator(maxStates)
    states = []
    for i, state in enumerate(gen):
        if i >= maxStates:
            break
        states.append(state)
    return states

def hyp2f1_regularized(a, b, c, z):
    return hyp2f1(a, b, c, z) / gamma(c)

def transitionElement(n, l, m, p, px, py, pz, Az, Ip):
    with np.errstate(invalid='raise', divide='raise'):
        sum = 0
        for iota in range(0, n-l):
            prefactor = (
                (-1)**iota
                * 2**(-(9/4) - l/2 + iota)
                * Ip**(-(7/4) - l/2)
                * n
                * (1j * n * np.sqrt((Az + p)**2))**l
                * binom(l + n, -1 - l + n - iota)
                * np.sqrt(gamma(-l + n) / gamma(1 + l + n))
                * gamma(3 + 2*l + iota)
            )

            theta = 0.5 * (np.pi - (2 * (Az + p) * np.arcsin((Az + pz)/(Az + p))) / np.sqrt((Az + p)**2))
            phi = np.arctan2(py, px)
            
            z = -((n**2 * (Az + p)**2) / (2 * Ip))
            F1 = hyp2f1_regularized(2.5 + l + iota/2, 3 + l + iota/2, 2.5 + l, z)
            F2 = hyp2f1_regularized(2 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, z)
            Ylm = sph_harm(m, l, phi, theta)
            
            Ylmp1 = sph_harm(m+1, l, phi, theta)
            
            term1 = (
                -n**2 * (Az + pz) * (3 + 2*l + iota) * (4 + 2*l + iota) * F1 * Ylm
            )
            
            sqrt1 = np.sqrt(1 - (Az + pz)**2 / (Az + p)**2)
            sqrt2 = np.sqrt(gamma(1 + l - m)) * np.sqrt(gamma(2 + l + m))
            sqrt3 = np.sqrt(gamma(l - m)) * np.sqrt(gamma(1 + l + m))
            exp_phi = np.exp(-1j * phi)
            term2 = (
                4 * Ip * F2 * (
                    ((l - m) * (Az + pz) * Ylm) / (Az + p)**2
                    - (exp_phi * sqrt1 * sqrt2 * Ylmp1) / (np.sqrt((Az + p)**2) * sqrt3)
                )
            )
            sum += prefactor * (term1 + term2)
        return sum