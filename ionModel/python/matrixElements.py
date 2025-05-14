import numpy as np
import pandas as pd
from scipy.special import gamma, binom, hyp2f1, sph_harm, lpmv
from scipy.interpolate import interp1d
import plotly.graph_objects as go

#print(get_coefficients(4)[1,:])

def get_eigenEnergy(excitedStates):
    states = get_hydrogen_states(excitedStates)
    return np.array([0.5*1/(n**2) for (n,l,m) in states])

def get_coefficients(excitedStates, t_grid):
    df=pd.read_csv("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0016/expec", sep='\s+', header=8)
    #shift every column name by one and remove the first column
    df.columns = df.columns[1:].tolist() + [""]
    #remove last column
    df = df.iloc[:, :-1]

    time = np.array(df["Time"])
    eigenEnergy = get_eigenEnergy(excitedStates)
    c_list = []
    for i in range(excitedStates):
        c = np.array(df[f"Re{{<H0:{i}|psi>}}"]) + np.array(df[f"Imag{{<H0:{i}|psi>}}"]) * 1j
        interp_real = interp1d(time, c.real, kind='cubic', fill_value="extrapolate")
        interp_imag = interp1d(time, c.imag, kind='cubic', fill_value="extrapolate")
        c_interp = (interp_real(t_grid) + 1j * interp_imag(t_grid))*np.exp(-1j*eigenEnergy[i]*t_grid)
        c_list.append(c_interp)
    return np.vstack(c_list)

def hydrogen_state_generator(n_max):
    for n in range(1, n_max + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                yield (n, l, 0)

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

def transitionElement(n, l, m, p, pz, Az, Ip):
    result = 0
    Apsqrt = Az**2 + p**2 + 2*Az*pz +1e-14 #can also take the limit
    Apz = Az + pz
    sqrt_pi = np.sqrt(np.pi)
    for iota in range(0, -1 - l + n + 1):
        prefactor = (
            (-1)**iota /
            (gamma(0.5 + l) * gamma(1 + iota)) *
            2**(-9/4 - l/2 + iota) *
            Ip**(0.5 * (-5 - l)) *
            (1j * n)**l * n *
            (Apsqrt)**(0.5 * (-3 + l)) *
            binom(l + n, -1 - l + n - iota) *
            np.sqrt(Ip**(3/2) * gamma(-l + n) / gamma(l + n + 1)) *
            gamma(3 + 2*l + iota)
        )

        z = -((n**2 * Apsqrt) / (2 * Ip))
        x = Apz / np.sqrt(Apsqrt)

        Pl = lpmv(0, l, x)  # m=0 for LegendreP

        F1 = hyp2f1(1 - l, 2 + l, 2, 0.5 - Apz / (2 * np.sqrt(Apsqrt)))
        F2 = hyp2f1(2 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, z)
        F3 = hyp2f1(2 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, z)
        F4 = hyp2f1(3 + l + iota/2, 0.5 * (3 + 2*l + iota), 1.5 + l, z)

        term1 = (
            2 * Ip * l * (1 + l) * (p - pz) * (2 * Az + p + pz) * F1 * F2 /
            np.sqrt(np.pi + 2 * l * np.pi)
        )

        sqrt_Ap2 = np.sqrt(Apsqrt)
        term2 = (
            4 * Ip * np.sqrt(Apsqrt / (np.pi + 2 * l * np.pi)) * (Az + pz) *
            (
                -(4 + l + iota) * F3 +
                (4 + 2*l + iota) * F4
            ) * Pl
        )

        result += prefactor * (term1 + term2)
    return result

def transitionElementtest(configState, p, pz, Az, Ip):      #first state and normal SFA are exactly 4pi apart
    n, l, m = configState
    termsqrt = Az**2 + p**2 + 2*Az*pz + 1e-14
    if n == 2 and l == 0:
        numerator = 128 * 2**(1/4) * Ip**2 * (Ip - termsqrt) * (Az + pz)
        denominator = (np.sqrt(Ip**(3/2)) *(Ip + 2 * termsqrt)**4 *np.pi)
        return numerator / denominator
    elif n == 1 and l == 0:
        numerator = 16 * 2**(3/4) * Ip**2 * (Az + pz)
        denominator = (np.sqrt(Ip**(3/2)) *(2 * Ip + termsqrt)**3 * np.pi)
        return numerator / denominator
    elif n==2 and l==1:
        term1 = Az + p
        term2 = Az + pz
        numerator = 16j * 2**(3/4) * Ip * np.sqrt(Ip**(3/2)) * (Ip + 2 * (term1**2 - 6 * term2**2))
        denominator = ((Ip + 2 * term1**2)**4) * np.pi
        return numerator / denominator
    elif n==3 and l==0:
        term1 = Az + p
        term2 = Az + pz
        numerator = -432 * 2**(3/4) * np.sqrt(3) * Ip**2 * (44 * Ip**2 - 324 * Ip * term1**2 + 243 * term1**4) * term2
        denominator = (np.sqrt(Ip**(3/2)) * (2 * Ip + 9 * term1**2)**5 * np.pi)
        return numerator / denominator
    elif n==3 and l==1:
        term1 = Az + p
        term2 = Az + pz
        numerator = -864j * 2**(3/4) * Ip * np.sqrt(Ip**(3/2)) * (
            4 * Ip**2
            - 180 * Ip * term2**2
            - 81 * (term1**4 - 6 * term1**2 * term2**2)
        )
        denominator = (2 * Ip + 9 * term1**2)**5 * np.pi
        return numerator / denominator