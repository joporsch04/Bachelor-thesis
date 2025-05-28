import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.integrate import solve_ivp, quad
from scipy.special import genlaguerre, factorial
from field_functions import LaserField
from scipy.interpolate import interp1d

class AtomicUnits:
    """Atomic units constants"""
    nm = 5.2917721e-2
    fs = 2.418884328e-2
    eV = 27.21138383
    speed_of_light = 137.035999

class HydrogenSolver:
    """Compact hydrogen atom TDSE solver"""
    
    def __init__(self, max_n, laser_params):
        self.max_n = max_n
        self._radial_cache = {}
        self.states = self._build_basis()
        self.energies = np.array([-0.5/n**2 for n, l, m in self.states])
        self.z_matrix = self._compute_z_matrix()
        self.laser_params = laser_params
    
    def _build_basis(self):
        """Build basis states (n,l,m) with m=0 for simplicity"""
        return [(n, l, 0) for n in range(1, self.max_n + 1) for l in range(n)]
    
    def _radial_wavefunction(self, r, n, l):
        """Hydrogen radial wavefunction R_nl(r)"""
        if l >= n:
            return np.zeros_like(r)
        
        coeff = np.sqrt((2/n)**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
        laguerre = genlaguerre(n-l-1, 2*l+1)
        return coeff * np.exp(-r/n) * (2*r/n)**l * laguerre(2*r/n)
    
    def _radial_integral(self, n1, l1, n2, l2):
        """Compute radial integral <R_n1l1|z|R_n2l2>"""
        key = tuple(sorted([(n1, l1), (n2, l2)]))
        if key in self._radial_cache:
            return self._radial_cache[key]
        
        if l1 >= n1 or l2 >= n2:
            return 0.0
        
        integrand = lambda r: (self._radial_wavefunction(r, n1, l1) * r * 
                              self._radial_wavefunction(r, n2, l2) * r**2)
        
        r_max = 2.5 * max(n1, n2)**2 + 40
        result, _ = quad(integrand, 0, r_max, limit=200, epsabs=1e-9)
        
        self._radial_cache[key] = result
        return result
    
    def _angular_integral(self, l1, m1, l2, m2):
        """Angular part of z matrix element"""
        if m1 != m2 or abs(l1 - l2) != 1:
            return 0.0
        
        if l1 == l2 + 1:
            return np.sqrt(((l2+1)**2 - m2**2) / ((2*l2+1)*(2*l2+3)))
        elif l1 == l2 - 1:
            return np.sqrt((l2**2 - m2**2) / ((2*l2-1)*(2*l2+1)))
        return 0.0
    
    def _compute_z_matrix(self):
        """Compute z matrix elements"""
        n_states = len(self.states)
        z_matrix = np.zeros((n_states, n_states))
        
        for i, (n1, l1, m1) in enumerate(self.states):
            for j, (n2, l2, m2) in enumerate(self.states):
                if i <= j:
                    radial = self._radial_integral(n1, l1, n2, l2)
                    angular = self._angular_integral(l1, m1, l2, m2)
                    z_matrix[i, j] = z_matrix[j, i] = radial * angular
        
        return z_matrix
    
    def _tdse_rhs_velocity(self, t, y):
        """TDSE right-hand side for velocity gauge"""
        c = y[:-1]
        A_z = y[-1].real
        dc_dt = np.zeros_like(c, dtype=complex)
        
        E_field = self.laser.Electric_Field(t)
        
        for k in range(len(self.states)):
            for n in range(len(self.states)):
                if k != n and self.z_matrix[k, n] != 0:
                    phase = np.exp(1j * (self.energies[k] - self.energies[n]) * t)
                    dc_dt[k] += (c[n] * (self.energies[k] - self.energies[n]) * 
                               self.z_matrix[k, n] * phase * A_z)
        
        return np.append(dc_dt, -E_field)
    
    def _tdse_rhs_length(self, t, y):
        """TDSE right-hand side for length gauge"""
        c = y
        dc_dt = np.zeros_like(c, dtype=complex)
        
        E_field = self.laser.Electric_Field(t)
        
        for k in range(len(self.states)):
            dc_dt[k] = -1j * self.energies[k] * c[k]
            for n in range(len(self.states)):
                if k != n and self.z_matrix[k, n] != 0:
                    dc_dt[k] += -1j * self.z_matrix[k, n] * E_field * c[n]
        
        return dc_dt
    
    def solve(self, gauge='both'):
        """Solve TDSE with given laser parameters"""
        lam0, intensity, cep = self.laser_params[:3]
        self.laser = LaserField(cache_results=True)
        self.laser.add_pulse(lam0, intensity, cep, 
                           lam0 / AtomicUnits.nm / AtomicUnits.speed_of_light)
        
        t_start, t_end = self.laser.get_time_interval()
        t_eval = np.linspace(t_start, t_end, 16000)
        
        results = {}
        
        if gauge in ['velocity', 'both']:
            c_init = np.zeros(len(self.states), dtype=complex)
            c_init[0] = 1.0
            y_init_vel = np.append(c_init, 0.0)
            
            results['velocity'] = solve_ivp(self._tdse_rhs_velocity, [t_start, t_end], y_init_vel, 
                                          t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-10)
        
        if gauge in ['length', 'both']:
            c_init = np.zeros(len(self.states), dtype=complex)
            c_init[0] = 1.0
            
            results['length'] = solve_ivp(self._tdse_rhs_length, [t_start, t_end], c_init, 
                                        t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-10)
        
        return results
    
    def plot_populations(self, solutions, state_indices=None, plot_type="occ"):
        fig = go.Figure()
        
        if state_indices is None:
            state_indices = list(range(1, len(self.states)))
        
        gauges = list(solutions.keys())
        
        for gauge_idx, gauge in enumerate(gauges):
            solution = solutions[gauge]
            
            for i in state_indices:
                if i >= len(self.states):
                    print(f"Warning: State index {i} is out of range. Max index is {len(self.states)-1}")
                    continue
                    
                n, l, m = self.states[i]
                
                if plot_type == "occ":
                    y_data = np.abs(solution.y[i, :])**2
                    trace_name = f'|{n},{l},{m}⟩² ({gauge})'
                    y_title = "Population |c<sub>k</sub>(t)|²"
                elif plot_type == "real":
                    y_data = solution.y[i, :].real
                    trace_name = f'Re{{c<sub>{n},{l},{m}</sub>}} ({gauge})'
                    y_title = "Real part of coefficients"
                elif plot_type == "imag":
                    y_data = solution.y[i, :].imag
                    trace_name = f'Im{{c<sub>{n},{l},{m}</sub>}} ({gauge})'
                    y_title = "Imaginary part of coefficients"
                elif plot_type == "mag":
                    y_data = solution.y[i, :].real**2 + solution.y[i, :].imag**2
                    trace_name = f'|c<sub>{n},{l},{m}</sub>| ({gauge})'
                    y_title = "Magnitude of coefficients"
                else:
                    raise ValueError("plot_type must be one of: 'occ', 'real', 'imag', 'mag'")
                
                fig.add_trace(go.Scatter(
                    x=solution.t, 
                    y=y_data,
                    mode='lines',
                    name=trace_name
                ))
        
        laser_text = "<br>".join([
            "<b>Laser Parameters:</b>",
            f"Wavelength: {self.laser_params[0]} nm",
            f"Intensity: {self.laser_params[1]:.0e} W/cm²",
            f"CEP: {self.laser_params[2]}"
        ])
        
        title_map = {
            "occ": "Hydrogen Atom State Populations",
            "real": "Real Part of Hydrogen Atom Coefficients", 
            "imag": "Imaginary Part of Hydrogen Atom Coefficients",
            "mag": "Magnitude of Hydrogen Atom Coefficients"
        }
        
        fig.update_layout(
            title=title_map.get(plot_type, "Hydrogen Atom Coefficients"),
            xaxis_title="Time (atomic units)",
            yaxis_title=y_title,
            xaxis=dict(range=[-100, 100]),
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=laser_text,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    align="left",
                    valign="top"
                )
            ]
        )
        fig.show()

def get_coefficientsNumerical(excitedStates, t_grid, get_only_p_states):
    laser_params = (850, 1e14, 0)
    
    solver = HydrogenSolver(max_n=3, laser_params=laser_params)
    print(f"Basis states ({len(solver.states)}): {solver.states}")
    
    solutions = solver.solve(gauge='length')

    gauges = list(solutions.keys())
        
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
            c_interp = (-interp_real(t_grid) + 1j * interp_imag(t_grid))
            c_list.append(c_interp)
        return np.vstack(c_list)

if __name__ == "__main__":
    laser_params = (850, 1e14, 0)
    
    solver = HydrogenSolver(max_n=3, laser_params=laser_params)
    print(f"Basis states ({len(solver.states)}): {solver.states}")
    
    solutions = solver.solve(gauge='length')
    
    fig = solver.plot_populations(solutions, [2], plot_type="real")