import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.integrate import cumulative_trapezoid, solve_ivp, quad
from scipy.interpolate import interp1d
from scipy.special import genlaguerre, sph_harm, factorial

from field_functions import LaserField



class AU:
    meter = 5.2917720859e-11 # atomic unit of length in meters
    nm = 5.2917721e-2 # atomic unit of length in nanometres
    second = 2.418884328e-17 # atomic unit of time in seconds
    fs = 2.418884328e-2 # atomic unit of time in femtoseconds
    Joule = 4.359743935e-18 # atomic unit of energy in Joules
    eV = 27.21138383 # atomic unit of energy in electronvolts
    Volts_per_meter = 5.142206313e+11 # atomic unit of electric field in V/m
    Volts_per_Angstrom = 51.42206313 # atomic unit of electric field in V/Angström
    speed_of_light = 137.035999 # vacuum speed of light in atomic units
    Coulomb = 1.60217646e-19 # atomic unit of electric charge in Coulombs
    PW_per_cm2_au = 0.02849451308 # PW/cm^2 in atomic units
AtomicUnits=AU

class HydrogenState:
    def __init__(self, n, l, m):
        if not (n > 0 and 0 <= l < n and -l <= m <= l):
            raise ValueError("Invalid quantum numbers")
        self.n = n
        self.l = l
        self.m = m
    @property
    def energy(self): return -0.5 / (self.n**2)
    def __repr__(self): return f"|{self.n},{self.l},{self.m}>"
    def __eq__(self, other): return self.n == other.n and self.l == other.l and self.m == other.m
    def __hash__(self): return hash((self.n, self.l, self.m))

def R_nl_func(r, n, l):
    if l >= n: return np.zeros_like(r)
    coeff = np.sqrt((2.0/n)**3 * factorial(n-l-1) / (2.0*n*factorial(n+l)))
    laguerre_poly = genlaguerre(n-l-1, 2*l+1)
    return coeff * np.exp(-r/n) * (2.0*r/n)**l * laguerre_poly(2.0*r/n)

_radial_integral_cache = {}
def radial_integral_z(n1, l1, n2, l2):
    key = tuple(sorted(((n1, l1), (n2, l2))))
    if key in _radial_integral_cache: return _radial_integral_cache[key]
    integrand = lambda r: R_nl_func(r, n1, l1) * r * R_nl_func(r, n2, l2) * r**2
    max_n_involved = max(n1, n2)
    r_max_integration = 2.5 * max_n_involved**2 + 40
    if l1 >= n1 or l2 >= n2: return 0.0
    result, error = quad(integrand, 0, r_max_integration, limit=200, epsabs=1e-9, epsrel=1e-9)
    _radial_integral_cache[key] = result
    return result

def angular_integral_z(l1, m1, l2, m2):
    if m1 != m2: return 0.0
    if abs(l1 - l2) != 1: return 0.0
    if l1 == l2 + 1: l, m = l2, m2; return np.sqrt(((l+1)**2 - m**2) / ((2*l+1)*(2*l+3)))
    elif l1 == l2 - 1: l, m = l2, m2; return np.sqrt((l**2 - m**2) / ((2*l-1)*(2*l+1)))
    return 0.0

_z_matrix_element_cache = {}
def z_matrix_element(state1: HydrogenState, state2: HydrogenState):
    key = tuple(sorted((state1, state2), key=lambda s: (s.n, s.l, s.m)))
    if key in _z_matrix_element_cache: return _z_matrix_element_cache[key]
    if state1.m != state2.m: _z_matrix_element_cache[key] = 0.0; return 0.0
    rad_int = radial_integral_z(state1.n, state1.l, state2.n, state2.l)
    ang_int = angular_integral_z(state1.l, state1.m, state2.l, state2.m)
    result = rad_int * ang_int
    _z_matrix_element_cache[key] = result
    return result

def extract_time(file_path):
    with open(file_path, 'r') as file:
        first_colomn = []
        for row, line in enumerate(file):
            if row < 6:
                continue
            line = line.strip()
            colomns = line.split()
            if len(colomns) >= 5:
                first_colomn.append(float(colomns[0]))
    return first_colomn

def extractField_0(file_path):
    with open(file_path, 'r') as file:
        fifth_colomn = []
        for row, line in enumerate(file):
            if row < 6:
                continue
            line = line.strip()
            colomns = line.split()
            if len(colomns) >= 5:
                fifth_colomn.append(float(colomns[5]))
    return fifth_colomn



def get_coeffNumerical(time, state):
    file_params = [
        ("850nm_1e+14", 850, 1e13, 350, 1e10, 1, 0, -np.pi/2),
    ]


    df_verify=pd.read_csv("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0019/expec", sep='\s+', header=8)
    df_verify.columns = df_verify.columns[1:].tolist() + [""]
    df_verify = df_verify.iloc[:, :-1]



    for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep_pump, cep_probe in file_params:
        laser_pulses = LaserField(cache_results=True)
        laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
        t_min, t_max = laser_pulses.get_time_interval()
        time_recon= np.arange(t_min, t_max+1, 1)





    # --- Basis Set ---
    basis_states = []
    max_n_basis = 3
    for n_val in range(1, max_n_basis + 1):
        for l_val in range(n_val):
            m_val = 0
            basis_states.append(HydrogenState(n_val, l_val, m_val))
    num_states = len(basis_states)
    print(f"Basis states ({num_states}): {basis_states}")

    E_j = np.array([s.energy for s in basis_states])
    z_jk_matrix = np.zeros((num_states, num_states))
    for j, state_j in enumerate(basis_states):
        for k, state_k in enumerate(basis_states):
            if j <= k:
                val = z_matrix_element(state_j, state_k)
                z_jk_matrix[j, k] = val
                z_jk_matrix[k, j] = val

    # --- 4. Coupled ODEs ---


    # --- 5. Numerical Solution ---
    s1_state = HydrogenState(1,0,0)
    idx_1s = basis_states.index(s1_state)
    c_initial = np.zeros(num_states, dtype=np.complex128)
    c_initial[idx_1s] = 1.0 + 0.0j
    A_z_initial = 0.0
    y0_complex = np.append(c_initial, A_z_initial)

    t_start, t_end = laser_pulses.get_time_interval()
    num_t_points = 500
    t_eval_points = np.linspace(t_start, t_end, num_t_points*16)

    def tdse_rhs(t, y_complex):
        c = y_complex[:-1]
        A_z_val = y_complex[-1].real
        dc_dt = np.zeros_like(c, dtype=np.complex128)
        #E_field_t = E_z_laser(t)
        E_field_t = laser_pulses.Electric_Field(t)
        for k_idx in range(num_states):
            sum_val = 0.0j
            for n_idx in range(num_states):
                if k_idx == n_idx: continue
                z_kn = z_jk_matrix[k_idx, n_idx]
                if z_kn == 0: continue
                phase_factor = np.exp(1j * (E_j[k_idx] - E_j[n_idx]) * t)
                term = c[n_idx] * (E_j[k_idx] - E_j[n_idx]) * z_kn * phase_factor
                sum_val += term
            dc_dt[k_idx] = sum_val * A_z_val
        dA_z_dt = -E_field_t
        derivatives = np.append(dc_dt, dA_z_dt)
        return derivatives

    sol = solve_ivp(tdse_rhs, [t_start, t_end], y0_complex, t_eval=t_eval_points,
                    method='RK45', rtol=1e-10, atol=1e-10) # Consider BDF or LSODA for long times if stiff


    return interp1d(sol.t, sol.y[state, :], kind='cubic', fill_value="extrapolate")(time)

