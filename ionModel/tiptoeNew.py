import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.backends.backend_pdf import PdfPages

from kernels import IonProb, IonRate, analyticalRate
from field_functions import LaserField
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


def writecsv_prob(filename, delay, ion_QS, ion_NA_GASFIR, ion_NA_SFA, ion_NA_reconstructed_GASFIR, ion_NA_reconstructed_SFA):
    """Writes delay and ionization probabilities"""
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['delay', 'ion_QS', 'ion_NA_GASFIR', 'ion_NA_SFA', 'ion_NA_reconstructed', 'ion_NA_reconstructed_SFA'])
        for i in range(len(delay)):
            writer.writerow([delay[i], ion_QS[i], ion_NA_GASFIR[i], ion_NA_SFA[i], ion_NA_reconstructed_GASFIR[i], ion_NA_reconstructed_SFA[i]])


class TIPTOEplotter:

    def __init__(self, ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA, nArate, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):
        self.ion_QS = ion_QS
        self.ion_na_GASFIR = ion_na_GASFIR
        self.ion_na_SFA = ion_na_SFA
        self.ion_na_reconstructed_GASFIR = ion_na_reconstructed_GASFIR
        self.ion_na_reconstructed_SFA = ion_na_reconstructed_SFA
        self.nArate = nArate
        self.delay = delay
        self.field_probe_fourier_time = field_probe_fourier_time
        self.time = time
        self.AU = AU
        self.lam0_pump = lam0_pump
        self.I_pump = I_pump
        self.lam0_probe = lam0_probe
        self.I_probe = I_probe
        self.FWHM_probe = FWHM_probe

    


    def plot_ion_4(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        x_lim_ion_yield = 5
        phase_add = 0


        ax1.plot(self.delay*self.AU.fs, self.ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_na_GASFIR, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_na_SFA, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticSFA}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_na_reconstructed_GASFIR, label=rf'$\mathrm{{P}}_\mathrm{{nonAdRecon_GASFIR}}$')
        ax1.plot(self.delay*self.AU.fs, self.ion_na_reconstructed_SFA, label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconSFA}}$')

        ax1.set_ylabel('Ionization Yield')
        ax1.set_xlabel('Delay (fs)')
        ax1.set_title('Ionization Yield with background')
        ax1.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
        ax1.legend(loc='center left')
        ax1.annotate(f'$\lambda_\mathrm{{Pump}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={self.I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={self.I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={self.FWHM_probe}\mathrm{{fs}}$', xy=(0.72, 0.4), xycoords='axes fraction', fontsize=9, ha='left', va='center')
        
        ion_na_GASFIR=self.ion_na_GASFIR-self.ion_na_GASFIR[-1]
        ion_QS=self.ion_QS-self.ion_QS[-1] 
        ion_na_reconstructed_GASFIR=self.ion_na_reconstructed_GASFIR-self.ion_na_reconstructed_GASFIR[-1]
        ion_na_SFA=self.ion_na_SFA-self.ion_na_SFA[-1]
        ion_na_reconstructed_SFA=self.ion_na_reconstructed_SFA-self.ion_na_reconstructed_SFA[-1]

        #ax2_2 = ax2.twinx()
        #ax2_2.plot(self.time*self.AU.fs, -self.field_probe_fourier_time, label='Probe Field', linestyle='--', color='gray')
        #ax2_2.set_ylabel('Probe Field')
        #ax2_2.legend(loc='lower left')

        # ax2.plot(self.delay*self.AU.fs, ion_QS/max(abs(ion_QS))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}\cdot${max(abs(ion_y))/max(abs(ion_QS)):.2f}')
        # ax2.plot(self.delay*self.AU.fs, ion_na_GASFIR/max(abs(ion_na_GASFIR))*max(abs(ion_y)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}\cdot${max(abs(ion_y))/max(abs(ion_na_GASFIR)):.2f}')
        ax2.plot(self.delay*self.AU.fs, ion_QS, label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax2.plot(self.delay*self.AU.fs, ion_na_GASFIR, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax2.plot(self.delay*self.AU.fs, ion_na_reconstructed_GASFIR, label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconGASFIR}}$')
        ax2.plot(self.delay*self.AU.fs, ion_na_SFA, label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticSFA}}$')
        ax2.plot(self.delay*self.AU.fs, ion_na_reconstructed_SFA, label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconSFA}}$')
        ax2.set_xlabel('Delay (fs)')
        ax2.set_ylabel('Ionization Yield')
        ax2.set_xlim(-x_lim_ion_yield, x_lim_ion_yield)
        ax2.legend()
        ax2.set_title('Ionization Yield without Background')
        ax2.annotate(f'$\lambda_\mathrm{{Pump}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={self.I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={self.I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={self.FWHM_probe}\mathrm{{fs}}$', xy=(0.72, 0.16), xycoords='axes fraction', fontsize=9, ha='left', va='center')

        
        field_probe_fourier, omega = FourierTransform(self.time*self.AU.fs, self.field_probe_fourier_time, t0=0)
        field_probe_fourier=field_probe_fourier.flatten()
        omega=omega[abs(field_probe_fourier)>max(abs(field_probe_fourier))*0.05]
        omega=omega[omega>0]
        omega=np.linspace(omega[0], omega[-1], 5000)
        field_probe_fourier = FourierTransform(self.time*self.AU.fs, self.field_probe_fourier_time, omega, t0=0)



        ion_QS_fourier = FourierTransform(self.delay[::-1]*self.AU.fs, ion_QS[::-1], omega, t0=0)
        ion_nonAdiabatic_fourier_GASFIR = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_GASFIR[::-1], omega, t0=0)
        ion_nonAdiabatic_reconstructed_fourier_GASFIR = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_reconstructed_GASFIR[::-1], omega, t0=0)
        ion_nonAdiabatic_fourier_SFA = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_SFA[::-1], omega, t0=0)
        ion_nonAdiabatic_reconstructed_fourier_SFA = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_reconstructed_SFA[::-1], omega, t0=0)



        ion_QS_resp=ion_QS_fourier/field_probe_fourier
        ion_nonAdiabatic_resp_GASFIR=ion_nonAdiabatic_fourier_GASFIR/field_probe_fourier
        ion_nonAdiabatic_reconstructed_resp_GASFIR=ion_nonAdiabatic_reconstructed_fourier_GASFIR/field_probe_fourier
        ion_nonAdiabatic_resp_SFA=ion_nonAdiabatic_fourier_SFA/field_probe_fourier
        ion_nonAdiabatic_reconstructed_resp_SFA=ion_nonAdiabatic_reconstructed_fourier_SFA/field_probe_fourier
        ion_nonAdiabatic_reponse_full=FourierTransform(self.nArate[0]*self.AU.fs, self.nArate[1], omega, t0=0)



        ax3.plot(omega/2/np.pi, np.abs(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reconstructed_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticReconGASFIR}}$')
        ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reponse_full), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticReconFull}}$')


        ax3.set_xlabel('Frequency (PHz)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Spectral Response Absolute Value')
        ax3.legend()


        ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_QS_resp)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_resp_GASFIR)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_reconstructed_resp_GASFIR)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconGASFIR}}$')
        ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_resp_SFA)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticSFA}}$')
        ax4.plot(omega/2/np.pi, np.unwrap(np.angle(ion_nonAdiabatic_reconstructed_resp_SFA)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconSFA}}$')

        ax4.set_xlabel('Frequency (PHz)')
        ax4.set_ylabel('Phase')
        ax4.set_title('Spectral Response Phase')
        ax4.legend(loc='lower left')
        ax4.annotate(f'$\lambda_\mathrm{{Pump}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={self.I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={self.I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={self.FWHM_probe}\mathrm{{fs}}$', xy=(0.71, 0.16), xycoords='axes fraction', fontsize=9, ha='left', va='center')



        
        plt.tight_layout()

        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/dataOutput/plotIon_{self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}.pdf'
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        
        #print(f"done {self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}")

        # plt.show()
        # plt.close()

        ax3.clear()
        ax4.clear()





        ax3.plot(omega/2/np.pi, np.real(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$') 
        ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_reconstructed_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconGASFIR}}$')
        ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_resp_SFA), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticSFA}}$')
        ax3.plot(omega/2/np.pi, np.real(ion_nonAdiabatic_reconstructed_resp_SFA), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconSFA}}$')


        ax3.set_xlabel('Frequency (PHz)')
        ax3.set_ylabel('Real Respose')
        ax3.set_title('Spectral Response Real Value')
        ax3.legend()


        ax4.plot(omega/2/np.pi, (np.imag(ion_QS_resp)), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_resp_GASFIR)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_reconstructed_resp_GASFIR)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconGASFIR}}$')
        ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_resp_SFA)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticSFA}}$')
        ax4.plot(omega/2/np.pi, (np.imag(ion_nonAdiabatic_reconstructed_resp_SFA)), label=rf'$\mathrm{{P}}_\mathrm{{nonAdReconSFA}}$')

        ax4.set_xlabel('Frequency (PHz)')
        ax4.set_ylabel('Imaginary Response')
        ax4.set_title('Spectral Response Imaginary Value')
        ax4.legend(loc='lower left')
        ax4.annotate(f'$\lambda_\mathrm{{Pump}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{Probe}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{pump}}={self.I_pump:.2e}$\n$\mathrm{{I}}_\mathrm{{probe}}={self.I_probe:.2e}$\n$\mathrm{{FWHM}}_\mathrm{{probe}}={self.FWHM_probe}\mathrm{{fs}}$', xy=(0.7, 0.5), xycoords='axes fraction', fontsize=9, ha='left', va='center')

        plt.tight_layout()

        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/dataOutput/plotIonReIm_{self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}.pdf'
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        
        # plt.show()
        # plt.close()




file_params = [
    ("850nm_350nm_1.25e+14", 850, 1.25e14, 350, 1e10, 0.93, 0, -np.pi/2),
    # ("850nm_350nm_7.5e+13", 850, 7.50e13, 350, 6.00e09, 0.93, 0, -np.pi/2),
    # ("900nm_320nm_5e+14", 900, 5e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("1200nm_320nm_1e+14", 1200, 1e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("900nm_250nm_8e+13", 900, 8e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_9e+13", 900, 9e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1e+14", 900, 1e14, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1.1e+14", 900, 1.1e14, 250, 6e8, 0.58, 0, -np.pi/2),
]

params = {'E_g': 0.5, 'αPol': 4.51, 'tau': 2.849306230484045, 'e1': 2.2807090369952894, 't0': 0.1, 't1': 3.043736601676354, 't2': 7.270940402611973, 'e2': 0, 't3': 0, 't4': 1, "div_p":2**-4, "div_theta":1}

delay = pd.read_csv("/home/user/BachelorThesis/Bachelor-thesis/ionModel/delay.csv")

delay = np.array(delay["delay"].values) 

REDO_comp = True
for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep_pump, cep_probe in file_params:
    if REDO_comp:
        laser_pulses = LaserField(cache_results=True)
        ion_qs = []
        ion_na_GASFIR = []
        ion_na_SFA = []
        ion_na_reconstructed_GASFIR = []
        ion_na_reconstructed_SFA = []
        laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
        t_min, t_max = laser_pulses.get_time_interval()
        time_recon= np.arange(t_min, t_max+1, 1)
        dt_dE=1/np.gradient(laser_pulses.Electric_Field(time_recon),time_recon)
        ion_na_rate_GASFIR = IonRate(time_recon, laser_pulses, params, dT=0.25/16, kernel_type='GASFIR')
        ion_na_rate_SFA = IonRate(time_recon, laser_pulses, params, dT=0.25/16, kernel_type='exact_SFA')
        na_background_GASFIR=np.trapz(ion_na_rate_GASFIR, time_recon)
        na_background_SFA=np.trapz(ion_na_rate_SFA, time_recon)
        na_grad_GASFIR=np.gradient(ion_na_rate_GASFIR, laser_pulses.Electric_Field(time_recon))
        na_grad_SFA=np.gradient(ion_na_rate_SFA, laser_pulses.Electric_Field(time_recon))
        laser_pulses.reset()
        for tau in delay:
            laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            t_min, t_max = laser_pulses.get_time_interval()
            time=np.arange(t_min, t_max+1, 1)
            ion_qs.append(1-np.exp(-np.trapz(analyticalRate(time, laser_pulses, params), time)))
            ion_na_GASFIR.append(1-np.exp(-IonProb(laser_pulses, params, dt=2, dT=0.25, kernel_type='GASFIR')))
            ion_na_SFA.append(1-np.exp(-IonProb(laser_pulses, params, dt=2, dT=0.25, kernel_type='exact_SFA')))
            laser_pulses.reset()
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            ion_na_reconstructed_GASFIR.append(1-np.exp(-na_background_GASFIR-np.trapz(na_grad_GASFIR*laser_pulses.Electric_Field(time_recon), time_recon)))
            ion_na_reconstructed_SFA.append(1-np.exp(-na_background_SFA-np.trapz(na_grad_SFA*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
            laser_pulses.reset()
        output_file = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/dataOutput/ionProb_{file_name}.csv"
        writecsv_prob(output_file, delay, ion_qs, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA)

    data_rate_delay = pd.read_csv(f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/dataOutput/ionProb_{file_name}.csv")
    delay=np.array(data_rate_delay['delay'].values)
    ion_na_GASFIR=np.array(data_rate_delay['ion_NA_GASFIR'].values)
    ion_na_GASFIR=np.array(data_rate_delay['ion_NA_SFA'].values)
    ion_QS=np.array(data_rate_delay['ion_QS'].values)


    laser_pulses = LaserField(cache_results=True)
    laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
    t_min, t_max = laser_pulses.get_time_interval()
    time_recon= np.arange(t_min, t_max+1, 1)
    dt_dE=1/np.gradient(laser_pulses.Electric_Field(time_recon),time_recon)
    ion_na_rate_GASFIR = IonRate(time_recon, laser_pulses, params, dT=0.25/16, kernel_type='GASFIR')
    ion_na_rate_SFA = IonRate(time_recon, laser_pulses, params, dT=0.25/16, kernel_type='exact_SFA')
    ion_qs_rate=analyticalRate(time_recon, laser_pulses, params)
    na_grad_GASFIR=np.gradient(ion_na_rate_GASFIR, laser_pulses.Electric_Field(time_recon))
    na_grad_SFA=np.gradient(ion_na_rate_SFA, laser_pulses.Electric_Field(time_recon))
    qs_grad=np.gradient(ion_qs_rate, laser_pulses.Electric_Field(time_recon))

    try:
        ion_na_reconstructed_GASFIR=np.array(data_rate_delay['ion_NA_reconstructed_GASFIR'].values)
        ion_na_reconstructed_SFA=np.array(data_rate_delay['ion_NA_reconstructed_SFA'].values)
    except:
        ion_na_reconstructed_GASFIR = []
        ion_na_reconstructed_SFA = []
        na_background_GASFIR=np.trapz(ion_na_rate_GASFIR, time_recon)
        na_background_SFA=np.trapz(ion_na_rate_SFA, time_recon)
        na_grad_GASFIR=np.gradient(ion_na_rate_GASFIR, laser_pulses.Electric_Field(time_recon))
        na_grad_SFA=np.gradient(ion_na_rate_SFA, laser_pulses.Electric_Field(time_recon))
        for tau in delay:
            laser_pulses.reset()
            laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
            ion_na_reconstructed_GASFIR.append(1-np.exp(-na_background_GASFIR-np.trapz(na_grad_GASFIR*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
            ion_na_reconstructed_SFA.append(1-np.exp(-na_background_SFA-np.trapz(na_grad_SFA*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
            laser_pulses.reset()
        data_rate_delay['ion_NA_reconstructed_GASFIR']=np.array(ion_na_reconstructed_GASFIR)
        ion_na_reconstructed_GASFIR = np.array(data_rate_delay['ion_NA_reconstructed_GASFIR'].values)
        ion_na_reconstructed_SFA = np.array(data_rate_delay['ion_NA_reconstructed_SFA'].values)
        output_file = f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ion_prob_{file_name}.csv"
        writecsv_prob(output_file, delay, ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA)


    probe=LaserField()
    probe.add_pulse(lam0_probe, I_probe, CEP=-np.pi/2, FWHM=FWHM_probe/AtomicUnits.fs)
    tmin, tmax=probe.get_time_interval()
    time=np.arange(tmin, tmax+1, 1.)
    field_probe_fourier_time=probe.Electric_Field(time)
    nArate=[time_recon, na_grad_GASFIR, na_grad_SFA, qs_grad]

    plotter = TIPTOEplotter(ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA, nArate, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)
    plotter.plot_ion_4()
    