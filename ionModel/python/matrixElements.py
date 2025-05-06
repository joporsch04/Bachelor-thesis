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

def transitionElement(n, l, m, p, px, py, pz, VP, Ip):
    Ap = VP + p
    Apz = VP + pz
    sqrt_Ap2 = np.sqrt(Ap**2 + 1e-12)
    x = Apz / sqrt_Ap2
    phi = np.arctan2(py, px)
    theta = 0.5 * (np.pi - (2 * Ap * np.arcsin(x)) / sqrt_Ap2)
    z = -n**2 * Ap**2 / (2 * Ip)
    total = 0.0
    
    if np.any(Ap == 0):
        print("Warning: Ap is zero!")
    if np.any(Ap**2 == 0):
        print("Warning: Ap^2 is zero!")
    if np.any(1 - (Apz**2)/(Ap**2) < 0):
        print("Warning: sqrt argument negative!")

    for iota in range(0, n - l):
        prefac = (1/factorial(iota)) * (-1)**iota * 2**(-2.25 - l/2 + iota)
        prefac *= Ip**(-1.75 - l/2) * n * (1j * n * sqrt_Ap2)**l
        prefac *= binom(l + n, -1 - l + n - iota)
        prefac *= np.sqrt(gamma(-l + n) / gamma(1 + l + n))
        prefac *= gamma(3 + 2*l + iota)

        F1 = hyp2f1_regularized(2.5 + l + iota/2, 3 + l + iota/2, 2.5 + l, z)
        F2 = hyp2f1_regularized(2 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, z)

        Ylm = sph_harm(l, m, theta, phi)
        Yl1m = sph_harm(l, 1 + m, theta, phi)

        sqrt_g1 = np.sqrt(gamma(1 + l - m))
        sqrt_g2 = np.sqrt(gamma(2 + l + m))
        sqrt_g3 = np.sqrt(gamma(l - m))
        sqrt_g4 = np.sqrt(gamma(1 + l + m))

        sqrt1_arg = 1 - (Apz**2) / (Ap**2 + 1e-12)
        sqrt1_arg = np.where(sqrt1_arg < 0, 0, sqrt1_arg)
        sqrt1 = np.sqrt(sqrt1_arg)

        exp1 = np.exp(-1j * phi)

        term1 = (
            -n**2 * Apz * (3 + 2*l + iota) * (4 + 2*l + iota) * F1 * Ylm
            + 4 * Ip * F2 * (
                ((l - m) * Apz * Ylm) / (Ap**2)
                - (exp1 * sqrt1 * sqrt_g1 * sqrt_g2 * Yl1m) / (sqrt_Ap2 * sqrt_g3 * sqrt_g4)
            )
        )

        total += prefac * term1

    return total