import numpy as np
from scipy.special import gamma, binom, hyp2f1, sph_harm, lpmv
from scipy.interpolate import interp1d
from line_profiler import profile
from numba import njit
import os, os.path
import matplotlib.pyplot as plt
import pandas as pd

from coefficientsODE import HydrogenSolver
from tRecXdata import tRecXdata

def get_eigenEnergy(excitedStates, get_p_states):
    states = get_hydrogen_states(excitedStates, get_p_states)
    return np.array([0.5*1/(n**2) for (n,l,m) in states])

#@profile
def get_coefficientsNumerical(excitedStates, t_grid, get_only_p_states, Gauge, laser_pulses):
    
    solver = HydrogenSolver(max_n=3, laser_pulses=laser_pulses)
    print(f"Basis states ({len(solver.states)}): {solver.states}")
    
    solutions = solver.solve(gauge=Gauge)

    gauges = list(solutions.keys())
    eigenEnergy = get_eigenEnergy(excitedStates+5, get_only_p_states)
        
    for gauge_idx, gauge in enumerate(gauges):
        solution = solutions[gauge]

        time = solution.t

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

            df = pd.DataFrame({'time': time, 'c_real': c.real, 'c_imag': c.imag})
            df = df.groupby('time').mean().reset_index().sort_values('time')
            
            t_min_data, t_max_data = df['time'].min(), df['time'].max()
            
            t_grid_clipped = np.clip(t_grid, t_min_data, t_max_data)
            
            interp_real = interp1d(df['time'], df['c_real'], kind='cubic', 
                                    bounds_error=False, fill_value='extrapolate')
            interp_imag = interp1d(df['time'], df['c_imag'], kind='cubic', 
                                    bounds_error=False, fill_value='extrapolate')
            
            c_real_interp = interp_real(t_grid_clipped)
            c_imag_interp = interp_imag(t_grid_clipped)
            
            mask_before = t_grid < t_min_data
            mask_after = t_grid > t_max_data
            
            if np.any(mask_before):
                c_real_interp[mask_before] = df['c_real'].iloc[0]
                c_imag_interp[mask_before] = df['c_imag'].iloc[0]
            
            if np.any(mask_after):
                c_real_interp[mask_after] = df['c_real'].iloc[-1]
                c_imag_interp[mask_after] = df['c_imag'].iloc[-1]
            
            c_interp = (c_real_interp + 1j * c_imag_interp)
            c_list.append(c_interp)
            
            # interp_real = interp1d(solution.t, c.real, kind='cubic', fill_value="extrapolate")
            # interp_imag = interp1d(solution.t, c.imag, kind='cubic', fill_value="extrapolate")
            # c_interp = (interp_real(t_grid) + 1j * interp_imag(t_grid))
            # c_list.append(c_interp)

        return np.vstack(c_list)

def get_coefficientstRecX_delay(excitedStates, t_grid, get_p_states, params, delay):

    delay_files_path = "/home/user/TIPTOE/new_data/450nm_short_length_gauge/250nm/I_8.00e+13"

    count_files = [eintrag for eintrag in os.listdir(delay_files_path) if os.path.isdir(os.path.join(delay_files_path, eintrag)) and eintrag.isdigit()]
    files_number = max([int(eintrag) for eintrag in count_files])


    for i in range(0, files_number+1):
        dir_path = os.path.join(delay_files_path, str(i))
        data = tRecXdata(dir_path)
        if float(data.extractDelay()) != float(delay):
            continue
        else:
            print(f"Found matching delay: {delay} in file {dir_path}")

            laser_params = data.laser_params
            if isinstance(laser_params, list):
                laser_params = laser_params[0]
            
            if float(laser_params['lam0']) != float(np.real(params['lam0'])) or float(laser_params['intensity']) != float(np.real(params['intensity'])):
                raise ValueError("Laser parameters do not match the expected values.")
            
            time =  np.array(data.coefficients['Time'])

            c_list = []
            eigenEnergy = get_eigenEnergy(excitedStates+5, get_p_states)

            for state_idx in range(excitedStates):
                if get_p_states:
                    if state_idx == 2:
                        state_idx = 4  # skip the 2p state
                #print(data.coefficients[f"Re{{<H0:{state_idx}|psi>}}"].head())

                c_real = np.array(data.coefficients[f"Re{{<H0:{state_idx}|psi>}}"]) 
                c_imag = np.array(data.coefficients[f"Imag{{<H0:{state_idx}|psi>}}"])
                c = c_real + 1j * c_imag

                df = pd.DataFrame({'time': time, 'c_real': c.real, 'c_imag': c.imag})
                df = df.groupby('time').mean().reset_index().sort_values('time')
                
                t_min_data, t_max_data = df['time'].min(), df['time'].max()
                
                t_grid_clipped = np.clip(t_grid, t_min_data, t_max_data)
                
                interp_real = interp1d(df['time'], df['c_real'], kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
                interp_imag = interp1d(df['time'], df['c_imag'], kind='cubic', 
                                     bounds_error=False, fill_value='extrapolate')
                
                c_real_interp = interp_real(t_grid_clipped)
                c_imag_interp = interp_imag(t_grid_clipped)
                
                mask_before = t_grid < t_min_data
                mask_after = t_grid > t_max_data
                
                if np.any(mask_before):
                    c_real_interp[mask_before] = df['c_real'].iloc[0]
                    c_imag_interp[mask_before] = df['c_imag'].iloc[0]
                
                if np.any(mask_after):
                    c_real_interp[mask_after] = df['c_real'].iloc[-1]
                    c_imag_interp[mask_after] = df['c_imag'].iloc[-1]
                
                c_interp = (c_real_interp + 1j * c_imag_interp) * np.exp(-1j*eigenEnergy[state_idx]*t_grid)
                c_list.append(c_interp)

            return np.vstack(c_list)
    raise ValueError(f"No matching delay {delay} found in the specified directory.")

def get_coefficientstRecX(excitedStates, t_grid, get_p_states, params):
    data = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0047")

    if float(data.laser_params['lam0']) != float(np.real(params['lam0'])) or float(data.laser_params['intensity']) != float(np.real(params['intensity'])):
        print(f"Expected laser parameters: {np.real(params['lam0'])}, {np.real(params['intensity'])}")
        print(f"Found laser parameters: {data.laser_params['lam0']}, {data.laser_params['intensity']}")
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

        c_interp = (interp_real(t_grid) + 1j * interp_imag(t_grid))*np.exp(-1j*eigenEnergy[state_idx]*t_grid)     #ATTENTION for 3p state is state_idx set to 4 but eigenenergy so doesnt work!!!!
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


@njit(cache=True, fastmath=True)        #
#@profile
def transitionElementtest(configState, p, pz, Az, Ip):      #first state and normal SFA are exactly 4pi apart
    n, l, m = configState
    termsqrt = Az**2 + p**2 + 2*Az*pz + 1e-20       #(Az+p)^2
    termsqrtdenom = np.sqrt(Ip**(3/2))
    if n == 1 and l == 0:
        numerator = 16 * 2**(3/4) * Ip**2 * (Az + pz)
        denominator = (termsqrtdenom *(2 * Ip + termsqrt)**3 * np.pi)
        return 1j*(numerator / denominator).astype(np.complex128)
    elif n == 2 and l == 0:
        numerator = 128 * 2**(1/4) * Ip**2 * (Ip - termsqrt) * (Az + pz)
        denominator = (termsqrtdenom * (Ip + 2 * termsqrt)**4 * np.pi)
        return 1j*(numerator / denominator).astype(np.complex128)
    elif n == 3 and l == 0:
        numerator = -432 * 2**(3/4) * np.sqrt(3) * Ip**2 * (44 * Ip**2 - 324 * Ip * (Az + p)**2 + 243 * (Az + p)**4) * (Az + pz)
        denominator = (termsqrtdenom * (2 * Ip + 9 * termsqrt)**5 * np.pi)
        return 1j*(numerator / denominator).astype(np.complex128)
    elif n == 2 and l == 1:
        Az_plus_p_squared = termsqrt
        
        numerator = (16 * 1j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * np.sqrt(2) * (Ip + 2 * (Az_plus_p_squared - 6 * (Az + pz)**2)))
        
        denominator = (Ip + 2 * Az_plus_p_squared)**4 * np.pi
        
        return 1j*(numerator / denominator).astype(np.complex128)
    elif n == 3 and l == 1:
        Az_plus_p_squared = termsqrt
        
        numerator = (864 * 1j * 2**(3/4) * Ip * np.sqrt(Ip**(3/2)) * (-4 * Ip**2 + 180 * Ip * (Az + pz)**2 + 81 * (termsqrt**2 - 6 * termsqrt * (Az + pz)**2)))
        
        denominator = (2 * Ip + 9 * Az_plus_p_squared)**5 * np.pi
        
        return 1j*(numerator / denominator).astype(np.complex128)
    
@njit(cache=True, fastmath=True)
def d10(psquared, pz, Ip):
    numerator = -16 * 2**(3/4) * Ip**2 * pz
    denominator = np.sqrt(Ip**(3/2)) *(2 * Ip + psquared)**3 * np.pi
    return (numerator / denominator).astype(np.complex128)

# def d21(psquared, pz, phi_p, Ip):
#     term1 = np.sqrt(2) * (Ip + 2 * (psquared - 6 * pz**2))
#     term2 = 24j * np.sqrt(psquared) * pz * np.sqrt(1 - pz**2/psquared) * np.sin(phi_p)
#     numerator = 16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * (term1 + term2)
#     denominator = (Ip + 2 * psquared)**4 * np.pi
#     return (1j)/3*(numerator / denominator).astype(np.complex128)

# def d31(psquared, pz, phi_p, Ip):
#     term1 = 1j * np.sqrt(2) * (-4 * Ip**2 + 180 * Ip * pz**2 + 81 * (psquared**2 - 6 * psquared * pz**2))
#     term2 = 36 * np.sqrt(psquared) * (10 * Ip - 27 * psquared) * pz * np.sqrt(1 - pz**2/psquared) * np.sin(phi_p)
#     numerator = 864 * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * (term1 + term2)
#     denominator = (2 * Ip + 9 * psquared)**5 * np.pi
#     return (1j)/3*(numerator / denominator).astype(np.complex128)

@njit(cache=True, fastmath=True)
def transitionElement_BA(configState, configStateRange, psquared_m, psquared_p, pz_m, pz_p, Ip):      #first state and normal SFA are exactly 4pi apart
    n, l, m = configState
    n_range, l_range, m_range = configStateRange

    if n == 1 and l == 0 and n_range == 1 and l_range == 0:
        return 2*np.pi*np.conjugate(d10(psquared_m, pz_m, Ip)) * d10(psquared_p, pz_p, Ip)
    
    elif n == 1 and l == 0 and n_range == 2 and l_range == 1:
        numerator_1 = 16 * 2**(3/4) * Ip**2 * pz_m
        denominator_1 = np.sqrt(Ip**(3/2)) *(2 * Ip + psquared_m)**3 * np.pi
        d10_m = np.conjugate(1/3*(numerator_1 / denominator_1)).astype(np.complex128)

        term1 = np.sqrt(2) * (Ip + 2 * (psquared_p - 6 * pz_p**2))
        numerator_2 = 16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term1
        denominator_2 = (Ip + 2 * psquared_p)**4 * np.pi
        d21_p = 1/3*(numerator_2 / denominator_2).astype(np.complex128)

        return 2*np.pi*d10_m * d21_p
    
    elif n == 2 and l == 1 and n_range == 1 and l_range == 0:
        numerator_1 = 16 * 2**(3/4) * Ip**2 * pz_m
        denominator_1 = np.sqrt(Ip**(3/2)) *(2 * Ip + psquared_m)**3 * np.pi
        d10_m = 1/3*(numerator_1 / denominator_1).astype(np.complex128)

        term1 = np.sqrt(2) * (Ip + 2 * (psquared_p - 6 * pz_p**2))
        numerator_2 = 16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term1
        denominator_2 = (Ip + 2 * psquared_p)**4 * np.pi
        d21_p = np.conjugate(1/3*(numerator_2 / denominator_2)).astype(np.complex128)

        return 2*np.pi*d21_p * d10_m

    # elif n == 2 and l == 1 and n_range == 2 and l_range == 1:
    #     term1_m = np.sqrt(2) * (Ip + 2 * (psquared_m - 6 * pz_m**2))
    #     term2_m = 24j * pz_m * np.sqrt(psquared_m - pz_m**2)

    #     term1_p = np.sqrt(2) * (Ip + 2 * (psquared_p - 6 * pz_p**2))
    #     term2_p = 24j * pz_p * np.sqrt(psquared_p - pz_p**2)

    #     numerator_1_m = np.conjugate(16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term1_m)
    #     numerator_2_m = np.conjugate(16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term2_m)
    #     denominator_m = (Ip + 2 * psquared_m)**4 * np.pi

    #     numerator_1_p = 16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term1_p
    #     numerator_2_p = 16j * 2**(1/4) * Ip * np.sqrt(Ip**(3/2)) * term2_p
    #     denominator_p = (Ip + 2 * psquared_p)**4 * np.pi
        
    #     firstterm = 2*np.pi*np.conjugate(numerator_1_m/denominator_m)*numerator_1_p/denominator_p
    #     secondterm = np.pi*np.conjugate(numerator_2_m/denominator_m)*numerator_2_p/denominator_p

    #     return 1/9*(firstterm + secondterm).astype(np.complex128)    #+1 because of i*i^*, coming from d=i\nabla_p\phi(p)

    elif n == 2 and l == 1 and n_range == 2 and l_range == 1:
        term1 = (Ip + 2 * (psquared_m - 6 * pz_m**2)) * (Ip + 2 * (psquared_p - 6 * pz_p**2))
        term2 = 144 * pz_m * pz_p * np.sqrt(psquared_p - pz_p**2) * np.sqrt(psquared_m - pz_m**2)
        
        numerator = 1024 * np.sqrt(2) * Ip**(7/2) * (term1 + term2)
        denominator = (Ip + 2 * psquared_m)**4 * (Ip + 2 * psquared_p)**4 * np.pi
        
        return 1/9*(numerator / denominator).astype(np.complex128)
    
    elif n == 3 and l == 1 and n_range == 3 and l_range == 1:
        # Calculate sqrt term: sqrt[(p1^2 - pz1^2) * (p2^2 - pz2^2)]
        sqrt_term = np.sqrt((psquared_m - pz_m**2) * (psquared_p - pz_p**2))
        
        # Terms with the sqrt factor
        term1 = 32400 * Ip**2 * pz_m * pz_p * sqrt_term
        term2 = -87480 * Ip * psquared_m * pz_m * pz_p * sqrt_term
        term3 = -87480 * Ip * psquared_p * pz_m * pz_p * sqrt_term
        term4 = 236196 * psquared_m * psquared_p * pz_m * pz_p * sqrt_term
        
        # Product of two expressions
        expr1 = 4 * Ip**2 - 180 * Ip * pz_m**2 - 81 * (psquared_m**2 - 6 * psquared_m * pz_m**2)
        expr2 = 4 * Ip**2 - 180 * Ip * pz_p**2 - 81 * (psquared_p**2 - 6 * psquared_p * pz_p**2)
        term5 = expr1 * expr2
        
        # Complete numerator
        numerator = 2985984 * np.sqrt(2) * Ip**(7/2) * (term1 + term2 + term3 + term4 + term5)
        
        # Denominator
        denominator = (2 * Ip + 9 * psquared_m)**5 * (2 * Ip + 9 * psquared_p)**5 * np.pi
        
        # Return result with normalization factor
        return 1/9 * (numerator / denominator).astype(np.complex128)