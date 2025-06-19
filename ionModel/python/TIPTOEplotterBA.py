import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from matplotlib.backends.backend_pdf import PdfPages

from __init__ import FourierTransform

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

class TIPTOEplotterBA:

    def __init__(self, ion_tRecX, ion_SFA, ion_SFA_excited_tRecX, ion_SFA_excited_ODE, delay, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):
        
        plt.rcParams['lines.linewidth'] = 1.0
        plt.rcParams['text.usetex'] = True
        self.ion_tRecX = ion_tRecX
        self.ion_SFA = ion_SFA
        self.ion_SFA_excited_tRecX = ion_SFA_excited_tRecX
        self.ion_SFA_excited_ODE = ion_SFA_excited_ODE
        self.delay = delay
        self.time = time
        self.AU = AU
        self.lam0_pump = lam0_pump
        self.I_pump = I_pump
        self.lam0_probe = lam0_probe
        self.I_probe = I_probe
        self.FWHM_probe = FWHM_probe

    def matplot4(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        x_lim_ion_yield = 4
        phase_add = 0


        ax1.plot(self.delay*self.AU.fs, self.ion_tRecX, label=rf'$\mathrm{{tRecX}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_SFA, label=rf'$\mathrm{{SFA}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_SFA_excited_tRecX, label=rf'$\mathrm{{SFA}}_\mathrm{{excited-tRecX}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_SFA_excited_ODE, label=rf'$\mathrm{{SFA}}_\mathrm{{excited-ODE}}$')

        ax1.set_ylabel(rf'Ionization Yield $\delta N(\tau)$')
        ax1.set_xlabel(rf'Delay $\tau$ (fs)')
        ax1.set_title('Total Ionization Yield')
        ax1.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
        ax1.legend(loc='center left')
        ax1.annotate(f'$\lambda_\mathrm{{F}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{S}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{F}}={self.I_pump:.2e}\mathrm{{W}}/\mathrm{{cm^2}}$\n$\mathrm{{I}}_\mathrm{{S}}={self.I_probe:.2e}\mathrm{{W}}/\mathrm{{cm^2}}$', xy=(0.72, 0.4), xycoords='axes fraction', fontsize=9, ha='left', va='center')
        
        ion_tRecX=self.ion_tRecX-self.ion_tRecX[-1]
        ion_SFA=self.ion_SFA-self.ion_SFA[-1]
        ion_SFA_excited_tRecX=self.ion_SFA_excited_tRecX-self.ion_SFA_excited_tRecX[10]
        ion_SFA_excited_ODE=self.ion_SFA_excited_ODE-self.ion_SFA_excited_ODE[-1]

        #ax2_2 = ax2.twinx()
        #ax2_2.plot(self.time*self.AU.fs, -self.field_probe_fourier_time, label='Probe Field', linestyle='--', color='gray')
        #ax2_2.set_ylabel('Probe Field')
        #ax2_2.legend(loc='lower left')

        # ax2.plot(self.delay*self.AU.fs, ion_QS/max(abs(ion_QS))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}\cdot${max(abs(ion_y))/max(abs(ion_QS)):.2f}')
        # ax2.plot(self.delay*self.AU.fs, ion_na_GASFIR/max(abs(ion_na_GASFIR))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}\cdot${max(abs(ion_y))/max(abs(ion_na_GASFIR)):.2f}')
        ax2.plot(self.delay*self.AU.fs, ion_tRecX, label=rf'$\mathrm{{tRecX}}$')
        ax2.plot(self.delay*self.AU.fs, ion_SFA, label=rf'$\mathrm{{SFA}}$')
        ax2.plot(self.delay*self.AU.fs, ion_SFA_excited_tRecX, label=rf'$\mathrm{{SFA}}_\mathrm{{excited-tRecX}}$')
        ax2.plot(self.delay*self.AU.fs, ion_SFA_excited_ODE, label=rf'$\mathrm{{SFA}}_\mathrm{{excited-ODE}}$')
        ax2.set_xlabel(rf'Delay $\tau$ (fs)')
        ax2.set_ylabel(rf'Ionization Yield $\delta N(\tau)-N_0$')
        ax2.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
        ax2.legend()
        ax2.set_title('Without Background')



        ax3.plot(self.delay*self.AU.fs, ion_tRecX/(max(ion_tRecX)), label=rf'$\mathrm{{tRecX}}$')
        ax3.plot(self.delay*self.AU.fs, ion_SFA/(max(ion_SFA)), label=rf'$\mathrm{{SFA}}$')
        ax3.plot(self.delay*self.AU.fs, ion_SFA_excited_tRecX/(max(ion_SFA_excited_tRecX)), label=rf'$\mathrm{{SFA}}_\mathrm{{excited-tRecX}}$')
        ax3.set_xlabel(rf'Delay $\tau$ (fs)')
        ax3.set_ylabel(r'Ionization Yield $\frac{\delta N(\tau)-N_0}{N_\mathrm{max}}$')
        ax3.set_xlim(-2, 2)
        ax3.legend()
        ax3.set_title('Normalized')


        ax4.plot(self.delay*self.AU.fs, ion_tRecX/(max(ion_tRecX)), label=rf'$\mathrm{{tRecX}}$')
        ax4.plot(self.delay*self.AU.fs, ion_SFA/(max(ion_SFA)), label=rf'$\mathrm{{SFA}}$')
        ax4.plot(self.delay*self.AU.fs, ion_SFA_excited_ODE/(max(ion_SFA_excited_ODE)), label=rf'$\mathrm{{SFA}}_\mathrm{{excited-ODE}}$', color='red')
        ax4.set_xlabel(rf'Delay $\tau$ (fs)')
        ax3.set_ylabel(r'Ionization Yield $\frac{\delta N(\tau)-N_0}{N_\mathrm{max}}$')
        ax4.set_xlim(-2, 2)
        ax4.legend()
        ax4.set_title('Normalized')



        plt.tight_layout()

        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/plotIon_{self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}.pdf'
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        
        #print(f"done {self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}")

        # plt.show()
        # plt.close()





def writecsv_prob(filename, delay, ion_tRecX, ion_QS, ion_NA_GASFIR, ion_NA_SFA, ion_NA_reconstructed_GASFIR, ion_NA_reconstructed_SFA):
    """Writes delay and ionization probabilities"""
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['delay', 'ion_tRecX', 'ion_QS', 'ion_NA_GASFIR', 'ion_NA_SFA', 'ion_NA_reconstructed_GASFIR', 'ion_NA_reconstructed_SFA'])
        for i in range(len(delay)):
            writer.writerow([delay[i], ion_tRecX[i], ion_QS[i], ion_NA_GASFIR[i], ion_NA_SFA[i], ion_NA_reconstructed_GASFIR[i], ion_NA_reconstructed_SFA[i]])

def readtRecX_ion(file_path):
    """Reads delay and tRecX ionization probability"""
    data = pd.read_csv(file_path, header=None)
    delay = pd.to_numeric(data.iloc[2].iloc[2:].values)[:-1]
    ion_y = pd.to_numeric(data.iloc[1].iloc[2:].values)[:-1]
    return ion_y