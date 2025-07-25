import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
from matplotlib.backends.backend_pdf import PdfPages
from field_functions import LaserField
from scipy.interpolate import interp1d

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

    def __init__(self, excitedStates, ion_tRecX, ion_SFA, ion_SFA_excited_tRecX, ion_SFA_excited_ODE, delay, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):
        
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],  # Matches lmodern package
            'font.size': 12,
            'axes.labelsize': 12,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 10,
            'figure.titlesize': 12,
            'lines.linewidth': 1.5,
            'axes.linewidth': 0.5,
            'grid.linewidth': 0.5,
            'xtick.major.width': 0.8,
            'ytick.major.width': 0.8,
            'xtick.minor.width': 0.5,
            'ytick.minor.width': 0.5,
            'savefig.dpi': 600,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        self.excitedStates = excitedStates
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
        ax1.annotate(f'$\lambda_\mathrm{{F}}={self.lam0_pump}\mathrm{{nm}}$\n$\lambda_\mathrm{{S}}={self.lam0_probe}\mathrm{{nm}}$\n$\mathrm{{I}}_\mathrm{{F}}={self.I_pump:.2e}\mathrm{{W}}/\mathrm{{cm^2}}$\n$\mathrm{{I}}_\mathrm{{S}}={self.I_probe:.2e}\mathrm{{W}}/\mathrm{{cm^2}}$\nStates: 1s,2p,3p', xy=(0.72, 0.4), xycoords='axes fraction', fontsize=9, ha='left', va='center')
        
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

        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/plotIon_{self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}_{self.excitedStates}_new.pdf'
        with PdfPages(pdf_filename) as pdf:
            pdf.savefig(fig)
        
        #print(f"done {self.lam0_pump}_{self.lam0_probe}_{self.I_pump:.2e}_{self.I_probe:.2e}")

        # plt.show()
        # plt.close()

    def plot2SFA(self):

        laser_pulses = LaserField(cache_results=True)
        laser_pulses.add_pulse(250, 8e9, -np.pi/2, 0.58/ AtomicUnits.fs)
        t_min, t_max = laser_pulses.get_time_interval()
        time_recon_1= np.arange(int(t_min), int(t_max)+1, 1.)

        fig, (ax3, ax4) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

        ion_tRecX = self.ion_tRecX - self.ion_tRecX[-1]
        ion_SFA = self.ion_SFA - self.ion_SFA[-1]
        ion_SFA_excited_tRecX = self.ion_SFA_excited_tRecX - self.ion_SFA_excited_tRecX[10]
        ion_SFA_excited_ODE = self.ion_SFA_excited_ODE - self.ion_SFA_excited_ODE[-1]

        delay_fs = self.delay * self.AU.fs
        delay_dense = np.linspace(delay_fs.min(), delay_fs.max(), len(delay_fs) * 30)
    
        interp_tRecX = interp1d(delay_fs, ion_tRecX, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA = interp1d(delay_fs, ion_SFA, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA_excited_tRecX = interp1d(delay_fs, ion_SFA_excited_tRecX, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA_excited_ODE = interp1d(delay_fs, ion_SFA_excited_ODE, kind='cubic', bounds_error=False, fill_value='extrapolate')

        ion_tRecX_dense = interp_tRecX(delay_dense)
        ion_SFA_dense = interp_SFA(delay_dense)
        ion_SFA_excited_tRecX_dense = interp_SFA_excited_tRecX(delay_dense)
        ion_SFA_excited_ODE_dense = interp_SFA_excited_ODE(delay_dense)

        ion_tRecX_dense = ion_tRecX
        ion_SFA_dense = ion_SFA
        ion_SFA_excited_tRecX_dense = ion_SFA_excited_tRecX
        ion_SFA_excited_ODE_dense = ion_SFA_excited_ODE
        delay_dense = delay_fs

        # N =18
        # ion_SFA_excited_tRecX[:N] = 0
        # ion_SFA_excited_tRecX[-N:] = 0

        ax3.plot(delay_dense, ion_tRecX_dense/(max(abs(ion_tRecX_dense))), label=r'TDSE (reference)', color='black', linestyle='-')
        ax3.plot(delay_dense, ion_SFA_dense/(max(abs(ion_SFA_dense))), label=r'Standard SFA', color='blue', linestyle='--', alpha=0.5)
        ax3.set_xlabel(r'Delay $\tau$ (fs)')
        ax3.set_ylabel(r'Normalized Ionization Yield')
        ax3.set_xlim(-2, 2)
        ax3.legend(loc='upper right', fontsize=13)
        ax3.set_title('(a) Standard SFA vs Reference')
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(delay_dense, ion_tRecX_dense/(max(abs(ion_tRecX_dense))), label=r'TDSE (reference)', color='black', linestyle='-')
        ax4.plot(delay_dense, ion_SFA_dense/(max(abs(ion_SFA_dense))), label=r'Standard SFA', color='blue', linestyle='--', alpha=0.5)
        ax4.plot(delay_dense, ion_SFA_excited_ODE_dense/(max(abs(ion_SFA_excited_ODE_dense))), label=r'Extended SFA (Sub. coeff.)', color='red', linestyle=':')
        ax4.plot(delay_dense, ion_SFA_excited_tRecX_dense/(max(abs(ion_SFA_excited_tRecX_dense))), label=r'Extended SFA (Full. coeff.)', color='green', linestyle='-.')
        ax4.set_xlabel(r'Delay $\tau$ (fs)')
        ax4.set_ylabel(r'Normalized Ionization Yield')
        ax4.set_xlim(-2, 2)
        ax4.legend(loc='upper right', fontsize=13)
        ax4.set_title('(b) SFA Models vs Reference')
        ax4.grid(True, alpha=0.3)

        param_text = (f'$\lambda_\mathrm{{F}}={self.lam0_pump}$ nm\n'
                    f'$\lambda_\mathrm{{S}}={self.lam0_probe}$ nm\n'
                    f'$I_\mathrm{{F}}={self.I_pump/1e13:.1f} \\times 10^{{13}}$ W/cm$^2$\n'
                    f'$I_\mathrm{{S}}={self.I_probe/1e9:.1f} \\times 10^{{9}}$ W/cm$^2$')
        ax3.text(0.02, 0.98, param_text, transform=ax3.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))

        plt.tight_layout()
        
        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/2plot_SFA-comparison_{self.excitedStates}_BP_laser_pulses.pdf'

        jpeg_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/2plot_SFA-comparison_{self.excitedStates}_BP_2_ax2_only.jpeg'
        
        extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        from matplotlib.transforms import Bbox
        padded_extent = Bbox.from_extents(
            extent.x0 - 0.9, extent.y0 - 0.5, 
            extent.x1 + 0.1, extent.y1 + 0.5
        )
        fig.savefig(jpeg_filename, format='jpeg', dpi=600, bbox_inches=padded_extent)

        # with PdfPages(pdf_filename) as pdf:
        #     pdf.savefig(fig, dpi=600, bbox_inches='tight')

    def plot3stark(self):

        filename = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/ionization_rates_450nm_8.0e+13Wcm2.csv"
        df = pd.read_csv(filename)

        time_recon = df['time_SFA'].values
        rate_SFA = df['rate_SFA'].values
        time_recon_1 = df['time_extended_1'].values
        rateExcited_1 = df['rate_extended_1'].values
        time_recon_2 = df['time_extended_2'].values
        rateExcited_2 = df['rate_extended_2'].values
        time_recon_3 = df['time_extended_3'].values
        rateExcited_3 = df['rate_extended_3'].values
        time_recon_4 = df['time_extended_4'].values
        rateExcited_4 = df['rate_extended_4'].values

        summary_filename = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/ionization_summary_450nm_8.0e+13Wcm2.csv"
        summary_df = pd.read_csv(summary_filename)

        print(f"Data loaded from: {filename}")
        print(f"Summary loaded from: {summary_filename}")
        print(f"DataFrame shape: {df.shape}")

        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[1, 1])
        
        ax1 = fig.add_subplot(gs[:, 0])  # Left plot spans both rows
        ax2 = fig.add_subplot(gs[0, 1])  # Top right
        ax3 = fig.add_subplot(gs[1, 1])  # Bottom right

        ion_tRecX = self.ion_tRecX - self.ion_tRecX[-1]
        ion_SFA = self.ion_SFA - self.ion_SFA[-1]
        ion_SFA_excited_tRecX = self.ion_SFA_excited_tRecX - self.ion_SFA_excited_tRecX[10]
        ion_SFA_excited_ODE = self.ion_SFA_excited_ODE - self.ion_SFA_excited_ODE[-1]

        delay_fs = self.delay * self.AU.fs
        delay_dense = np.linspace(delay_fs.min(), delay_fs.max(), len(delay_fs) * 50)

        interp_tRecX = interp1d(delay_fs, ion_tRecX, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA = interp1d(delay_fs, ion_SFA, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA_excited_tRecX = interp1d(delay_fs, ion_SFA_excited_tRecX, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_SFA_excited_ODE = interp1d(delay_fs, ion_SFA_excited_ODE, kind='cubic', bounds_error=False, fill_value='extrapolate')

        ion_tRecX_dense = interp_tRecX(delay_dense)
        ion_SFA_dense = interp_SFA(delay_dense)
        ion_SFA_excited_tRecX_dense = interp_SFA_excited_tRecX(delay_dense)
        ion_SFA_excited_ODE_dense = interp_SFA_excited_ODE(delay_dense)

        ion_tRecX_dense = ion_tRecX
        ion_SFA_dense = ion_SFA
        ion_SFA_excited_tRecX_dense = ion_SFA_excited_tRecX
        ion_SFA_excited_ODE_dense = ion_SFA_excited_ODE
        delay_dense = delay_fs

        time_dense = np.linspace(min(time_recon.min(), time_recon_1.min(), time_recon_2.min(), 
                                    time_recon_3.min(), time_recon_4.min()), 
                                max(time_recon.max(), time_recon_1.max(), time_recon_2.max(),
                                    time_recon_3.max(), time_recon_4.max()), 
                                len(time_recon) * 50)
        
        interp_rate_SFA = interp1d(time_recon, rate_SFA, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_rate_1 = interp1d(time_recon_1, rateExcited_1, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_rate_2 = interp1d(time_recon_2, rateExcited_2, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_rate_3 = interp1d(time_recon_3, rateExcited_3, kind='cubic', bounds_error=False, fill_value='extrapolate')
        interp_rate_4 = interp1d(time_recon_4, rateExcited_4, kind='cubic', bounds_error=False, fill_value='extrapolate')

        rate_SFA_dense = interp_rate_SFA(time_dense)
        rate_1_dense = interp_rate_1(time_dense)
        rate_2_dense = interp_rate_2(time_dense)
        rate_3_dense = interp_rate_3(time_dense)
        rate_4_dense = interp_rate_4(time_dense)

        rate_SFA_dense = rate_SFA
        rate_1_dense = rateExcited_1
        rate_2_dense = rateExcited_2
        rate_3_dense = rateExcited_3
        rate_4_dense = rateExcited_4
        time_dense = time_recon_2

        ax1.plot(delay_dense, ion_tRecX_dense/(max(abs(ion_tRecX_dense))), label=r'TDSE (reference)', color='black', linestyle='-')
        ax1.plot(delay_dense, ion_SFA_dense/(max(abs(ion_SFA_dense))), label=r'Standard SFA', color='blue', linestyle='--', alpha=0.5)
        ax1.plot(delay_dense, ion_SFA_excited_ODE_dense/(max(abs(ion_SFA_excited_ODE_dense))), label=r'Extended SFA (Sub. coeff.)', color='darkred', linestyle=':')
        ax1.plot(delay_dense, ion_SFA_excited_tRecX_dense/(max(abs(ion_SFA_excited_tRecX_dense))), label=r'Extended SFA (Full. coeff.)', color='darkgreen', linestyle='-.')
        ax1.set_xlabel(r'Delay $\tau$ (fs)')
        ax1.set_ylabel(r'Normalized Ionization Yield')
        ax1.set_xlim(-2, 2)
        ax1.legend(loc='upper right', fontsize=14)
        ax1.set_title('(a) Extended SFA Models vs Reference: Only Phase')
        ax1.grid(True, alpha=0.3)

        ax2.plot(time_dense, rate_SFA_dense, label=r'Standard SFA', color='blue', linestyle='-', alpha=0.5)
        ax2.plot(time_dense, rate_3_dense, label=r"Sub. SFA: only phase", color='darkred', linestyle='-.')
        ax2.plot(time_dense, rate_4_dense, label=r"Sub. SFA: only magn", color=(1.0, 0.4, 0.6), linestyle='--')
        ax2.set_xlim(-37, 37)
        ax2.set_xlabel(r'Time (fs)')
        ax2.set_ylabel(rf'Ionization Rate ($\mathrm{{fs}}^{{-1}}$)')
        ax2.legend(loc='upper right')
        ax2.set_title(r'(b) Sub Hilbertspace Extended SFA:' + '\n' +  'Only Phase vs Only Magnitude')
        ax2.grid(True, alpha=0.3)
        
        prob_SFA = 0.00011346179704394354
        prob_rate_3 = 0.00010049079038137945
        prob_rate_4 = 0.00010876240978240904
        
        prob_text_2 = (f'Ionization Probabilities:\n'
                      f'Standard SFA: {prob_SFA/1e-4:.3f}$\\times 10^{{-4}}$\n'
                      f'Sub. SFA (phase): {prob_rate_3/1e-4:.3f}$\\times 10^{{-4}}$\n'
                      f'Sub. SFA (magn): {prob_rate_4/1e-4:.3f}$\\times 10^{{-4}}$')
        ax2.text(0.02, 0.97, prob_text_2, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))

        ax3.plot(time_dense, rate_SFA_dense, label=r'Standard SFA', color='blue', linestyle='-', alpha=0.5)
        ax3.plot(time_dense, rate_1_dense, label=r"Full. SFA: only phase", color='darkgreen', linestyle='-.')
        ax3.plot(time_dense, rate_2_dense, label=r"Full. SFA: only magn", color=(0.0, 0.858, 0.528), linestyle='--')
        ax3.set_xlim(-39, 39)
        ax3.set_xlabel(r'Time (fs)')
        ax3.set_ylabel(rf'Ionization Rate ($\mathrm{{fs}}^{{-1}}$)')
        ax3.legend(loc='upper right')
        ax3.set_title(r'(c) Full Hilbertspace Extended SFA:' + '\n' +  'Only Phase vs Only Magnitude')
        ax3.grid(True, alpha=0.3)

        prob_rate_1 = 0.00014147482586329002
        prob_rate_2 = 0.00010939851885141187
        
        prob_text_3 = (f'Ionization Probabilities:\n'
                      f'Standard SFA: {prob_SFA/1e-4:.3f}$\\times 10^{{-4}}$\n'
                      f'Full. SFA (phase): {prob_rate_1/1e-4:.3f}$\\times 10^{{-4}}$\n'
                      f'Full. SFA (magn): {prob_rate_2/1e-4:.3f}$\\times 10^{{-4}}$')
        # ax3.text(0.015, 0.98, prob_text_3, transform=ax3.transAxes, fontsize=10, 
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))


        param_text = (f'$\lambda_\mathrm{{F}}={self.lam0_pump}$ nm\n'
                    f'$\lambda_\mathrm{{S}}={self.lam0_probe}$ nm\n'
                    f'$I_\mathrm{{F}}={self.I_pump/1e13:.1f} \\times 10^{{13}}$ W/cm$^2$\n'
                    f'$I_\mathrm{{S}}={self.I_probe/1e9:.1f} \\times 10^{{9}}$ W/cm$^2$')
        ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))

        plt.tight_layout()
        
        pdf_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/3plot_stark-comparison_{self.excitedStates}_BP_7.pdf'

        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        jpeg_filename = f'/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plotsTIPTOE/3plot_stark-comparison_{self.excitedStates}_BP_5_ax1_only.jpeg'

        extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        from matplotlib.transforms import Bbox
        padded_extent = Bbox.from_extents(
            extent.x0 - 0.9, extent.y0 - 0.5, 
            extent.x1 + 0.1, extent.y1 + 0.5
        )
        fig.savefig(jpeg_filename, format='jpeg', dpi=600, bbox_inches=padded_extent)

        # with PdfPages(pdf_filename) as pdf:
        #     pdf.savefig(fig, dpi=600, bbox_inches='tight')





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