import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go

from kernels import IonProb, IonRate, analyticalRate
from field_functions import LaserField
from __init__ import FourierTransform
from TIPTOEplotter import TIPTOEplotter, writecsv_prob, AU
from TIPTOEplotter import AU as AtomicUnits
from scipy.integrate import simpson
from line_profiler import profile

def read_ion_Prob_data(file_path):
    data = pd.read_csv(file_path, header=None)
    delay = pd.to_numeric(data.iloc[2].iloc[2:].values)[:-1]
    ion_y = pd.to_numeric(data.iloc[1].iloc[2:].values)[:-1]
    return delay, ion_y

file_params = [
    #("850nm_350nm_1.25e+14", 850, 1.25e14, 350, 1e10, 0.93, 0, -np.pi/2),
    ("450nm_8e+13", 450, 8e13, 250, 8e9, 1, 0, -np.pi/2),
    # ("850nm_350nm_7.5e+13", 850, 7.50e13, 350, 6.00e09, 0.93, 0, -np.pi/2),
    # ("900nm_320nm_5e+14", 900, 5e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("1200nm_320nm_1e+14", 1200, 1e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("900nm_250nm_8e+13", 900, 8e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_9e+13", 900, 9e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1e+14", 900, 1e14, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1.1e+14", 900, 1.1e14, 250, 6e8, 0.58, 0, -np.pi/2),
]

def main(excitedstates):
    params = {'E_g': 0.5, 'αPol': 4.51, 'tau': 2.849306230484045, 'e1': 2.2807090369952894, 't0': 0.1, 't1': 3.043736601676354, 't2': 7.270940402611973, 'e2': 0, 't3': 0, 't4': 1, "div_p":2**-4*8, "div_theta":1*8, 'lam0': 450, 'intensity': 8e13, 'cep': 0}


    REDO_comp = True
    for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep_pump, cep_probe in file_params:

        # delaydf = pd.read_csv("/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ionProb_450nm_250nm_8e+13.csv")

        # delay = np.array(delaydf["delay"].values)
        # ion_tRecX = readtRecX_ion(f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ionProb_450nm_250nm_8e+13.csv")

        delay, ion_tRecX = read_ion_Prob_data("/home/user/TIPTOE-Hydrogen/process_all_files_output/ionProb_450nm_250nm_8e+13.csv")

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
            ion_na_rate_GASFIR = IonRate(time_recon, laser_pulses, params, dT=0.5/2, kernel_type='GASFIR')
            ion_na_rate_SFA = IonRate(time_recon, laser_pulses, params, dT=0.5/2, kernel_type='exact_SFA', excitedStates=excitedstates, coeffType="numerical", gauge="length", get_p_only=True, only_c0_is_1_rest_normal=False, delay=-264.99)
            na_background_GASFIR=np.trapz(ion_na_rate_GASFIR, time_recon)
            na_background_SFA=np.trapz(ion_na_rate_SFA, time_recon)
            na_grad_GASFIR=np.gradient(ion_na_rate_GASFIR, laser_pulses.Electric_Field(time_recon))
            na_grad_SFA=np.gradient(ion_na_rate_SFA, laser_pulses.Electric_Field(time_recon))
            laser_pulses.reset()
            for tau in delay:
                print(tau)
                laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
                laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
                t_min, t_max = laser_pulses.get_time_interval()
                time=np.arange(t_min, t_max+1, 1)
                ion_qs.append(1-np.exp(-np.trapz(analyticalRate(time, laser_pulses, params), time)))
                ion_na_GASFIR.append(1-np.exp(-IonProb(laser_pulses, params, dt=2, dT=0.5/2, kernel_type='GASFIR')))
                #ion_na_SFA.append(1-np.exp(-IonProb(laser_pulses, params, dt=2, dT=0.5, kernel_type='exact_SFA')))
                ion_na_rate_SFA_probe = IonRate(time_recon, laser_pulses, params, dT=0.5/2, kernel_type='exact_SFA', excitedStates=excitedstates, coeffType="numerical", gauge="length", get_p_only=True, delay=-tau, only_c0_is_1_rest_normal=False)
                ion_na_SFA.append(1-np.exp(-np.double(simpson(ion_na_rate_SFA_probe, x=time_recon, axis=-1, even='simpson'))))
                laser_pulses.reset()
                laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
                ion_na_reconstructed_GASFIR.append(1-np.exp(-na_background_GASFIR-np.trapz(na_grad_GASFIR*laser_pulses.Electric_Field(time_recon), time_recon)))
                ion_na_reconstructed_SFA.append(1-np.exp(-na_background_SFA-np.trapz(na_grad_SFA*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
                laser_pulses.reset()
            output_file = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_{file_name}_{excitedstates}.csv"
            writecsv_prob(output_file, delay, ion_tRecX, ion_qs, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA)

        data_rate_delay = pd.read_csv(f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_{file_name}_{excitedstates}.csv")
        delay=np.array(data_rate_delay['delay'].values)
        ion_tRecX=np.array(data_rate_delay['ion_tRecX'].values)
        ion_na_GASFIR=np.array(data_rate_delay['ion_NA_GASFIR'].values)
        print("test2")
        ion_na_SFA=np.array(data_rate_delay['ion_NA_SFA'].values)
        ion_QS=np.array(data_rate_delay['ion_QS'].values)

        try:
            print("recon try")
            ion_na_reconstructed_GASFIR=np.array(pd.to_numeric(data_rate_delay['ion_NA_reconstructed_GASFIR'].values))
            ion_na_reconstructed_SFA=np.array(pd.to_numeric(0*data_rate_delay['ion_NA_reconstructed_SFA'].values))
            print("test")
        except:
            print("recon except")
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
            writecsv_prob(output_file, delay, ion_tRecX, ion_qs, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA)


        probe=LaserField()
        probe.add_pulse(lam0_probe, I_probe, CEP=-np.pi/2, FWHM=FWHM_probe/AtomicUnits.fs)
        tmin, tmax=probe.get_time_interval()
        time=np.arange(tmin, tmax+1, 1.)
        field_probe_fourier_time=probe.Electric_Field(time)
        print("test3")

        plotter = TIPTOEplotter(ion_tRecX, ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)
        print("test4")
        fig = go.Figure()
        fig = plotter.plotly4()
        output_path = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/plot_{file_name}_{excitedstates}_maxnhigh.html"
        fig.write_html(output_path)
        print(f"Plot saved to: {output_path}")
        # fig.show()


if __name__ == "__main__":
    main(1)
    main(2)
    main(3)