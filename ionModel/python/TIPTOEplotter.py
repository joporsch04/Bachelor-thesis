import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

class TIPTOEplotter:

    def __init__(self, ion_tRecX, ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe):
        self.ion_tRecX = ion_tRecX
        self.ion_QS = ion_QS
        self.ion_na_GASFIR = ion_na_GASFIR
        self.ion_na_SFA = ion_na_SFA
        self.ion_na_reconstructed_GASFIR = ion_na_reconstructed_GASFIR
        self.ion_na_reconstructed_SFA = ion_na_reconstructed_SFA
        self.delay = delay
        self.field_probe_fourier_time = field_probe_fourier_time
        self.time = time
        self.AU = AU
        self.lam0_pump = lam0_pump
        self.I_pump = I_pump
        self.lam0_probe = lam0_probe
        self.I_probe = I_probe
        self.FWHM_probe = FWHM_probe

    def matplot4(self):

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

        x_lim_ion_yield = 5
        phase_add = 0


        ax1.plot(self.delay*self.AU.fs, self.ion_tRecX, label='tRecX')
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
        
        ion_na_tRecX=self.ion_tRecX-self.ion_tRecX[-1]
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
        ax2.plot(self.delay*self.AU.fs, ion_na_tRecX, label='tRecX')
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



        ax3.plot(omega/2/np.pi, np.abs(ion_QS_resp), label=rf'$\mathrm{{P}}_\mathrm{{QS}}$')
        ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticGASFIR}}$')
        ax3.plot(omega/2/np.pi, np.abs(ion_nonAdiabatic_reconstructed_resp_GASFIR), label=rf'$\mathrm{{P}}_\mathrm{{nonAdiabaticReconGASFIR}}$')


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

    def plotly4(self):
        x_lim_ion_yield = 2

        ion_na_tRecX = self.ion_tRecX - self.ion_tRecX[-1]
        ion_na_GASFIR = self.ion_na_GASFIR - self.ion_na_GASFIR[-1]
        ion_QS = self.ion_QS - self.ion_QS[-1]
        ion_na_reconstructed_GASFIR = self.ion_na_reconstructed_GASFIR - self.ion_na_reconstructed_GASFIR[-1]
        ion_na_SFA = self.ion_na_SFA - self.ion_na_SFA[-1]
        ion_na_reconstructed_SFA = self.ion_na_reconstructed_SFA - self.ion_na_reconstructed_SFA[-1]

        field_probe_fourier, omega = FourierTransform(self.time*self.AU.fs, self.field_probe_fourier_time, t0=0)
        field_probe_fourier = field_probe_fourier.flatten()
        omega = omega[abs(field_probe_fourier) > max(abs(field_probe_fourier)) * 0.05]
        omega = omega[omega > 0]
        omega = np.linspace(omega[0], omega[-1], 5000)
        field_probe_fourier = FourierTransform(self.time*self.AU.fs, self.field_probe_fourier_time, omega, t0=0)

        ion_QS_fourier = FourierTransform(self.delay[::-1]*self.AU.fs, ion_QS[::-1], omega, t0=0)
        ion_nonAdiabatic_fourier_GASFIR = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_GASFIR[::-1], omega, t0=0)
        ion_nonAdiabatic_reconstructed_fourier_GASFIR = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_reconstructed_GASFIR[::-1], omega, t0=0)
        ion_nonAdiabatic_fourier_SFA = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_SFA[::-1], omega, t0=0)
        ion_nonAdiabatic_reconstructed_fourier_SFA = FourierTransform(self.delay[::-1]*self.AU.fs, ion_na_reconstructed_SFA[::-1], omega, t0=0)

        ion_QS_resp = ion_QS_fourier / field_probe_fourier
        ion_nonAdiabatic_resp_GASFIR = ion_nonAdiabatic_fourier_GASFIR / field_probe_fourier
        ion_nonAdiabatic_reconstructed_resp_GASFIR = ion_nonAdiabatic_reconstructed_fourier_GASFIR / field_probe_fourier
        ion_nonAdiabatic_resp_SFA = ion_nonAdiabatic_fourier_SFA / field_probe_fourier
        ion_nonAdiabatic_reconstructed_resp_SFA = ion_nonAdiabatic_reconstructed_fourier_SFA / field_probe_fourier

        names = {
            "QS": r"$\mathrm{P}_\mathrm{QS}$",
            "GASFIR": r"$\mathrm{P}_\mathrm{nonAdiabaticGASFIR}$",
            "ReconGASFIR": r"$\mathrm{P}_\mathrm{nonAdReconGASFIR}$",
            "SFA": r"$\mathrm{P}_\mathrm{SFA}$",
            "ReconSFA": r"$\mathrm{P}_\mathrm{nonAdReconSFA}$",
            "SFA_excited": r"$\mathrm{P}_\mathrm{SFA_excited}$",
            "tRecX": r"$\mathrm{P}_{\mathrm{tRecX}}$"
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Ionization Yield with background",
                "Ionization Yield without Background and normalized",
                "Spectral Response Absolute Value",
                "Spectral Response Phase"
            )
        )

        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=self.ion_tRecX, name=names["tRecX"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=self.ion_na_GASFIR, name=names["SFA"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=self.ion_na_SFA, name=names["SFA_excited"]), row=1, col=1)
        fig.update_xaxes(title_text="Delay (fs)", row=1, col=1, range=[-x_lim_ion_yield, x_lim_ion_yield])
        fig.update_yaxes(title_text="Ionization Yield", row=1, col=1)

        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=ion_na_tRecX/(max(ion_na_tRecX)), name=names["tRecX"]), row=1, col=2)
        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=ion_na_GASFIR/(max(ion_na_GASFIR)), name=names["SFA"]), row=1, col=2)
        fig.add_trace(go.Scatter(x=self.delay*self.AU.fs, y=ion_na_SFA/(max(ion_na_SFA)), name=names["SFA_excited"]), row=1, col=2)
        fig.update_xaxes(title_text="Delay (fs)", row=1, col=2, range=[-x_lim_ion_yield, x_lim_ion_yield])
        fig.update_yaxes(title_text="Ionization Yield", row=1, col=2)

        fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_nonAdiabatic_resp_GASFIR), name=names["SFA"]), row=2, col=1)
        fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.abs(ion_nonAdiabatic_resp_SFA), name=names["SFA_excited"]), row=2, col=1)
        fig.update_xaxes(title_text="Frequency (PHz)", row=2, col=1)
        fig.update_yaxes(title_text="Amplitude", row=2, col=1)

        fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_nonAdiabatic_resp_GASFIR)), name=names["SFA"]), row=2, col=2)
        fig.add_trace(go.Scatter(x=omega/2/np.pi, y=np.unwrap(np.angle(ion_nonAdiabatic_resp_SFA)), name=names["SFA_excited"]), row=2, col=2)
        fig.update_xaxes(title_text="Frequency (PHz)", row=2, col=2)
        fig.update_yaxes(title_text="Phase", row=2, col=2)

        annotation_text = (
            r"$\lambda_\mathrm{Pump}=" + f"{self.lam0_pump}" + r"\mathrm{{nm}}$<br>"
            r"$\lambda_\mathrm{Probe}=" + f"{self.lam0_probe}" + r"\mathrm{{nm}}$<br>"
            r"$\mathrm{I}_\mathrm{pump}=" + f"{self.I_pump:.2e}" + r"$<br>"
            r"$\mathrm{I}_\mathrm{probe}=" + f"{self.I_probe:.2e}" + r"$<br>"
            r"$\mathrm{FWHM}_\mathrm{probe}=" + f"{self.FWHM_probe}" + r"\mathrm{{fs}}$"
        )
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.99, y=0.01, showarrow=False,
            font=dict(size=12), align="left"
        )

        fig.update_layout(
            height=900, width=1400,
            title_text="TIPTOE Ionization Analysis (Plotly)",
            legend=dict(x=1.05, y=1),
            font=dict(family="serif", size=16)
        )
        

        fig.for_each_trace(lambda t: t.update(hovertemplate=t.name))
        fig.update_layout(title_text="Side By Side Subplots", width=1800, height=1080)
        fig.show()




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