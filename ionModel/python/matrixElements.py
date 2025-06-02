import numpy as np
import pandas as pd
from scipy.special import gamma, binom, hyp2f1, sph_harm, lpmv
from scipy.interpolate import interp1d
import plotly.graph_objects as go

from coefficientsNumerical import HydrogenSolver
from tRecXdata import tRecXdata

def get_eigenEnergy(excitedStates, get_p_states):
    states = get_hydrogen_states(excitedStates, get_p_states)
    return np.array([0.5*1/(n**2) for (n,l,m) in states])

def get_coefficientsNumerical(excitedStates, t_grid, get_only_p_states, Gauge, params):
    laser_params = (np.real(params['lam0']), np.real(params['intensity']), np.real(params['cep']))
    
    solver = HydrogenSolver(max_n=3, laser_params=laser_params)
    print(f"Basis states ({len(solver.states)}): {solver.states}")
    
    solutions = solver.solve(gauge=Gauge)

    gauges = list(solutions.keys())
    eigenEnergy = get_eigenEnergy(excitedStates+5, get_only_p_states)
        
    for gauge_idx, gauge in enumerate(gauges):
        solution = solutions[gauge]

        c_list = []
        for state_idx in range(excitedStates):
            if get_only_p_states:
                if state_idx == 0:
                    c = solution.y[0, :]    #1s state
                if state_idx == 1:
                    c = solution.y[2, :]    #2p state
                if state_idx == 2:
                    c = solution.y[4, :]    #3p state
            else:
                c = solution.y[state_idx, :]
            interp_real = interp1d(solution.t, c.real, kind='cubic', fill_value="extrapolate")
            interp_imag = interp1d(solution.t, c.imag, kind='cubic', fill_value="extrapolate")
            c_interp = (interp_real(t_grid) + 1j * interp_imag(t_grid))
            c_list.append(c_interp)
        return np.vstack(c_list)

def get_coefficientstRecX(excitedStates, t_grid, get_p_states, params):
    data = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0042")

    if float(data.laser_params['lam0']) != float(np.real(params['lam0'])) or float(data.laser_params['intensity']) != float(np.real(params['intensity'])):
        raise ValueError("Laser parameters do not match the expected values.")
    
    time = data.coefficients['Time']

    c_list = []
    eigenEnergy = get_eigenEnergy(excitedStates+5, get_p_states)

    for state_idx in range(excitedStates):
        if get_p_states:
            if state_idx == 2:
                state_idx = 4  # skip the 2p state
        #print(data.coefficients[f"Re{{<H0:{state_idx}|psi>}}"].head())
        c = np.array(data.coefficients[f"Re{{<H0:{state_idx}|psi>}}"]) + np.array(data.coefficients[f"Imag{{<H0:{state_idx}|psi>}}"]) * 1j
        
        unique_time, unique_indices = np.unique(time, return_index=True)
        c_unique = c[unique_indices]

        interp_real = interp1d(unique_time, c_unique.real, kind='cubic', fill_value="extrapolate")
        interp_imag = interp1d(unique_time, c_unique.imag, kind='cubic', fill_value="extrapolate")

        c_interp = (interp_real(t_grid) + 1j * interp_imag(t_grid))*np.exp(-1j*eigenEnergy[state_idx]*t_grid)        #*np.exp(+1j*eigenEnergy[i]*t_grid)
        c_list.append(c_interp)

    return np.vstack(c_list)

def hydrogen_state_generator(n_max, get_p_states):
    for n in range(1, n_max + 1):
        for l in range(0, n):
            if get_p_states and l != 1 and n != 1:
                continue
            yield (n, l, 0)

def get_hydrogen_states(maxStates, get_p_states):
    gen = hydrogen_state_generator(maxStates, get_p_states)
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
    thetap = np.arccos(pz/(p+1e14))
    termsqrt = Az**2 + p**2 + 2*Az*pz + 1e-14       #(Az+p)^2
    if n == 1 and l == 0:
        numerator = 16 * 2**(3/4) * Ip**2 * (Az + pz)
        denominator = (np.sqrt(Ip**(3/2)) *(2 * Ip + termsqrt)**3 * np.pi)
        return numerator / denominator
    elif n == 2 and l == 0:
        numerator = 128 * 2**(1/4) * Ip**2 * (Ip - termsqrt) * (Az + pz)
        denominator = (np.sqrt(Ip**(3/2)) * (Ip + 2 * termsqrt)**4 * np.pi)
        return numerator / denominator
    elif n == 3 and l == 0:
        numerator = -432 * 2**(3/4) * np.sqrt(3) * Ip**2 * (44 * Ip**2 - 324 * Ip * (Az + p)**2 + 243 * (Az + p)**4) * (Az + pz)
        denominator = (np.sqrt(Ip**(3/2)) * (2 * Ip + 9 * termsqrt)**5 * np.pi)
        return numerator / denominator
    elif n == 2 and l == 1:
        phi_p = 1
        exp_minus_i_phi = np.exp(-1j * phi_p)
        exp_i_phi = np.exp(1j * phi_p)
        exp_2i_phi = np.exp(2j * phi_p)
        cos_theta = np.cos(thetap)
        sin_theta = np.sin(thetap)
        numerator = (
            32 * exp_minus_i_phi * (
                -2 * exp_i_phi * (-1 + 20 * p**2) * cos_theta**2 +
                np.sqrt(2) * ( -1 + exp_2i_phi ) * ( -1 + p + 4 * p**2 * (5 + p) ) * cos_theta * sin_theta +
                2 * exp_i_phi * p * (1 + 4 * p**2) * sin_theta**2
            )
        )
        denominator = (1 + 4 * p**2)**4 * np.pi
        return 1/3*numerator / denominator
    elif n==3 and l==1:
        phi_p = 1
        exp_minus_i_phi = np.exp(-1j * phi_p)
        exp_i_phi = np.exp(1j * phi_p)
        exp_2i_phi = np.exp(2j * phi_p)
        cos_theta = np.cos(thetap)
        sin_theta = np.sin(thetap)
        numerator = (
            216 * exp_minus_i_phi * (
                -2 * exp_i_phi * (1 - 90 * p**2 + 405 * p**4) * cos_theta**2 +
                np.sqrt(2) * ( -1 + exp_2i_phi ) * (1 + p * (-1 - 90 * p + 81 * p**3 * (5 + p))) * cos_theta * sin_theta +
                2 * exp_i_phi * p * (-1 + 81 * p**4) * sin_theta**2
            )
        )
        denominator = (1 + 9 * p**2)**5 * np.pi
        return 1/3*numerator / denominator