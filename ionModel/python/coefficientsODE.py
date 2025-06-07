import numpy as np
import plotly.graph_objects as go
from scipy.integrate import solve_ivp
from scipy.special import genlaguerre, factorial
from field_functions import LaserField
from line_profiler import profile

class AtomicUnits:
    nm = 5.2917721e-2
    fs = 2.418884328e-2
    eV = 27.21138383
    speed_of_light = 137.035999

class HydrogenSolver:
    def __init__(self, max_n, laser_params):
        self.max_n = max_n
        self._radial_cache = {}
        self.states = self._build_basis()
        self.energies = np.array([-0.5/n**2 for n, l, m in self.states])
        self.z_matrix = self._compute_z_matrix()
        #self.p_matrix = self._compute_p_matrix()
        self.laser_params = laser_params
    
    def _build_basis(self):
        return [(n, l, 0) for n in range(1, self.max_n + 1) for l in range(n)]
    
    def _radial_wavefunction(self, r, n, l):
        coeff = np.sqrt((2/n)**3 * factorial(n-l-1) / (2*n*factorial(n+l)))
        laguerre = genlaguerre(n-l-1, 2*l+1)
        return coeff * np.exp(-r/n) * (2*r/n)**l * laguerre(2*r/n)
    
    #@profile
    def _radial_integral(self, n1, l1, n2, l2):
        key = tuple(sorted([(n1, l1), (n2, l2)]))
        if key in self._radial_cache:
            return self._radial_cache[key]
        
        if l1 >= n1 or l2 >= n2:
            return 0.0
        
        integrand = lambda r: (self._radial_wavefunction(r, n1, l1) * r * 
                              self._radial_wavefunction(r, n2, l2) * r**2)
        
        r_max = 20 * max(n1, n2)**2 + 200

        # result, _ = quad(integrand, 0, r_max, limit=200, epsabs=1e-9)
        
        # self._radial_cache[key] = result
        # return result
    
        #result, err_est = quad(integrand, 0, r_max, limit=200, epsabs=1e-9)
        result = np.trapz(integrand(np.linspace(0, r_max, 10000)), np.linspace(0, r_max, 10000))
        # if err_est > 1e-7: # Or some threshold you're uncomfortable with
        #     print(f"Warning: Large error estimate ({err_est:.2e}) for radial integral with n1={n1}, l1={l1}, n2={n2}, l2={l2}")
        self._radial_cache[key] = result
        return result
    
    def _angular_integral(self, l1, m1, l2, m2):
        if m1 != m2 or abs(l1 - l2) != 1:
            return 0.0
        
        if l1 == l2 + 1:
            return np.sqrt(((l2+1)**2 - m2**2) / ((2*l2+1)*(2*l2+3)))
        elif l1 == l2 - 1:
            return np.sqrt((l2**2 - m2**2) / ((2*l2-1)*(2*l2+1)))
        return 0.0
    
    def _compute_z_matrix(self):
        n_states = len(self.states)
        z_matrix = np.zeros((n_states, n_states))
        
        for i, (n1, l1, m1) in enumerate(self.states):
            for j, (n2, l2, m2) in enumerate(self.states):
                angular = self._angular_integral(l1, m1, l2, m2)
                if angular == 0:
                    radial = 0.0
                else:
                    radial = self._radial_integral(n1, l1, n2, l2)
                z_matrix[i, j] = radial * angular
        return z_matrix
    
    # def _tdse_rhs_velocity(self, t, y):
    #     c = y
    #     dc_dt = np.zeros_like(c, dtype=complex)
        
    #     A_z = self.laser.Vector_potential(t)
        
    #     for k in range(len(self.states)):
    #         for n in range (len(self.states)):
    #             omega_kn = self.energies[k] - self.energies[n]
    #             V_kn = -1j*self.p_matrix[k, n] * A_z
    #             dc_dt[k] += -1j*np.exp(1j*omega_kn)*V_kn*c[n]
        
    #     return dc_dt
    
    def _tdse_rhs_length(self, t, y):
        c = y
        dc_dt = np.zeros_like(c, dtype=complex)
        
        E_field = self.laser.Electric_Field(t)
        
        for k in range(len(self.states)):
            for n in range(len(self.states)):
                omega_kn = self.energies[k] - self.energies[n]
                dc_dt[k] += -1j *np.exp(1j*omega_kn*t) * self.z_matrix[k, n] * E_field * c[n]
        
        return dc_dt
    
    #@profile
    def solve(self, gauge='both'):
        lam0, intensity, cep = self.laser_params[:3]
        self.laser = LaserField(cache_results=True)
        self.laser.add_pulse(lam0, intensity, cep, lam0/ AtomicUnits.nm / AtomicUnits.speed_of_light) #make complex 128, float 64
        
        t_start, t_end = self.laser.get_time_interval()
        t_eval = np.linspace(t_start, t_end, 16000)     #64000
        
        results = {}
        
        # Initial condition: ground state
        c_init = np.zeros(len(self.states), dtype=complex)
        c_init[0] = 1.0
        
        if gauge in ['length', 'both']:
            results['length'] = solve_ivp(self._tdse_rhs_length, [t_start, t_end], c_init, 
                                        t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-12)
        
        # if gauge in ['velocity', 'both']:
            
        #     results['velocity'] = solve_ivp(self._tdse_rhs_velocity, [t_start, t_end], c_init, 
        #                                   t_eval=t_eval, method='DOP853', rtol=1e-8, atol=1e-8)
        
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
                    # sumcn = np.zeros_like(solution.t)
                    # for j in range(len(self.states)):
                    #     sumcn += np.abs(solution.y[j, :])**2
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
                    y_data = np.abs(solution.y[i, :])
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
        
        laser_text = "<br>".join(["<b>Laser Parameters:</b>",f"Wavelength: {self.laser_params[0]} nm",f"Intensity: {self.laser_params[1]:.0e} W/cm²",f"CEP: {self.laser_params[2]}"])
        
        title_map = {"occ": "Hydrogen Atom State Populations","real": "Real Part of Hydrogen Atom Coefficients","imag": "Imaginary Part of Hydrogen Atom Coefficients","mag": "Magnitude of Hydrogen Atom Coefficients"}
        
        fig.update_layout(
            title=title_map.get(plot_type, "Hydrogen Atom Coefficients"),
            xaxis_title="Time (a.u.)",
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
        return fig

if __name__ == "__main__":
    laser_params = (450, 1e14, 0)
    
    solver = HydrogenSolver(max_n=3, laser_params=laser_params)
    solutions = solver.solve(gauge='length')
    solver.plot_populations(solutions, plot_type="occ")