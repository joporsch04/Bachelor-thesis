import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import time

from kernels import IonProb, IonRate, analyticalRate
from field_functions import LaserField
from __init__ import FourierTransform
from TIPTOEplotter import TIPTOEplotter, writecsv_prob, AU
from TIPTOEplotter import AU as AtomicUnits
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from line_profiler import profile
from TIPTOEplotterBA import TIPTOEplotterBA

def read_ion_Prob_data(file_path):
    data = pd.read_csv(file_path, header=None)
    delay = pd.to_numeric(data.iloc[2].iloc[2:].values)[:-1]
    ion_y = pd.to_numeric(data.iloc[1].iloc[2:].values)[:-1]
    return delay, ion_y

file_params = [
    #("850nm_350nm_1.25e+14", 850, 1.25e14, 350, 1e10, 0.93, 0, -np.pi/2),
    ("450nm_dense_8e+13", 450, 8e13, 250, 8e9, 0.58, 0, -np.pi/2),
    # ("850nm_350nm_7.5e+13", 850, 7.50e13, 350, 6.00e09, 0.93, 0, -np.pi/2),
    # ("900nm_320nm_5e+14", 900, 5e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("1200nm_320nm_1e+14", 1200, 1e14, 320, 4e10, 0.75, 0, -np.pi/2),
    # ("900nm_250nm_8e+13", 900, 8e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_9e+13", 900, 9e13, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1e+14", 900, 1e14, 250, 6e8, 0.58, 0, -np.pi/2),
    # ("900nm_250nm_1.1e+14", 900, 1.1e14, 250, 6e8, 0.58, 0, -np.pi/2),
]

def main(excitedstates):
    params = {
        'E_g': 0.5, 
        'αPol': 4.51, 
        'div_p':2**-4*32, 
        'div_theta':1*8,
        'lam0': 450, 
        'intensity': 8e13, 
        'cep': 0,
        'excitedStates': excitedstates, 
        'coeffType': 'trecx', 
        'gauge': 'length', 
        'get_p_only': True, 
        'only_c0_is_1_rest_normal': False, 
        'delay': -224.97,
        'plotting': False
    }
    REDO_comp = False
    for file_name, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe, cep_pump, cep_probe in file_params:

        # delaydf = pd.read_csv("/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ionProb_450nm_250nm_8e+13.csv")

        # delay = np.array(delaydf["delay"].values)
        # ion_tRecX = readtRecX_ion(f"/home/user/TIPTOE-Hydrogen/plot_ion_tau_calc_output_data/ionProb_450nm_250nm_8e+13.csv")
        delay, ion_tRecX = read_ion_Prob_data("/home/user/TIPTOE/process_all_files_output/ionProb_450nm_dense_length_gauge_250nm_8e+13.csv")

        if REDO_comp:
            laser_pulses = LaserField(cache_results=True)
            ion_qs = []
            ion_na_GASFIR = []
            ion_na_SFA = []
            ion_na_reconstructed_GASFIR = []
            ion_na_reconstructed_SFA = []
            laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
            t_min, t_max = laser_pulses.get_time_interval()
            time_recon= np.arange(t_min, t_max+1, 0.5)
            ion_na_rate_GASFIR = IonRate(time_recon, laser_pulses, params, dT=0.5/4, kernel_type='exact_SFA')
            ion_na_rate_SFA = IonRate(time_recon, laser_pulses, params, dT=0.5/4, kernel_type='exact_SFA', excitedStates=True)
            na_background_GASFIR=np.trapz(ion_na_rate_GASFIR, time_recon)
            na_background_SFA=np.trapz(ion_na_rate_SFA, time_recon)
            na_grad_GASFIR=np.gradient(ion_na_rate_GASFIR, laser_pulses.Electric_Field(time_recon))
            na_grad_SFA=np.gradient(ion_na_rate_SFA, laser_pulses.Electric_Field(time_recon))
            laser_pulses.reset()
            for tau in delay:
                params['delay'] = -tau
                print(tau)
                laser_pulses.add_pulse(lam0_pump, I_pump, cep_pump, lam0_pump/ AtomicUnits.nm / AtomicUnits.speed_of_light)
                laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
                t_min, t_max = laser_pulses.get_time_interval()
                time=np.arange(t_min, t_max+1, 0.5)
                ion_qs.append(0)
                ion_na_rate_GASFIR_probe = IonRate(time_recon, laser_pulses, params, dT=0.5/4, kernel_type='exact_SFA')
                ion_na_GASFIR.append(1-np.exp(-np.double(simpson(ion_na_rate_GASFIR_probe, x=time_recon, axis=-1, even='simpson'))))
                #ion_na_SFA.append(1-np.exp(-IonProb(laser_pulses, params, dt=2, dT=0.5, kernel_type='exact_SFA')))
                ion_na_rate_SFA_probe = IonRate(time_recon, laser_pulses, params, dT=0.5/4, kernel_type='exact_SFA', excitedStates=True)
                print(np.real(np.trapz(ion_na_rate_SFA_probe, time_recon)))
                ion_na_SFA.append(1-np.exp(-np.double(simpson(ion_na_rate_SFA_probe, x=time_recon, axis=-1, even='simpson'))))
                laser_pulses.reset()
                laser_pulses.add_pulse(lam0_probe, I_probe, cep_probe, FWHM_probe/AtomicUnits.fs, t0=-tau)
                ion_na_reconstructed_GASFIR.append(1-np.exp(-na_background_GASFIR-np.trapz(na_grad_GASFIR*laser_pulses.Electric_Field(time_recon), time_recon)))
                ion_na_reconstructed_SFA.append(1-np.exp(-na_background_SFA-np.trapz(na_grad_SFA*laser_pulses.Electric_Field(time_recon), time_recon))) #+na_grad2*laser_pulses.Electric_Field(time_recon)**2/2
                laser_pulses.reset()
            output_file = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_{file_name}_{excitedstates}_trecx_length.csv"
            writecsv_prob(output_file, delay, ion_tRecX, ion_qs, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA)

        data_rate_delay = pd.read_csv(f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_{file_name}_{excitedstates}_trecx_length.csv")
        delay=np.array(data_rate_delay['delay'].values)
        ion_tRecX=np.array(data_rate_delay['ion_tRecX'].values)
        ion_na_GASFIR=np.array(data_rate_delay['ion_NA_GASFIR'].values)
        ion_na_SFA=np.array(data_rate_delay['ion_NA_SFA'].values)
        ion_QS=np.array(data_rate_delay['ion_QS'].values)

        BA_plotting = True

        if BA_plotting:
            #data_rate_delay_ODE = pd.read_csv(f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_{file_name}_{excitedstates}_numerical_length.csv") #change numerical to ODE
            data_rate_delay_ODE = pd.read_csv("/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/ionProb_450nm_8e+13_3_numerical_newconvergence.csv")
            delay_ODE=np.array(data_rate_delay_ODE['delay'].values)
            ion_tRecX_ODE=np.array(data_rate_delay_ODE['ion_tRecX'].values)
            ion_na_GASFIR_ODE=np.array(data_rate_delay_ODE['ion_NA_GASFIR'].values)
            ion_na_SFA_ODE=np.array(data_rate_delay_ODE['ion_NA_SFA'].values)
            ion_QS_ODE=np.array(data_rate_delay_ODE['ion_QS'].values)
            ion_SFA_ODE_new = interp1d(delay_ODE, ion_na_SFA_ODE, fill_value="extrapolate")(delay)

        delay_dummy, ion_tRecX_dummy = read_ion_Prob_data("/home/user/TIPTOE/process_all_files_output/ionProb_450nm_dense_velocity_gauge_250nm_8e+13.csv")

        #ion_tRecX_new = interp1d(delay_dummy, ion_tRecX_dummy, fill_value="extrapolate")(delay)

        ion_tRecX = ion_tRecX_dummy         #now c_n in length gauge, ion_tRecX is in velocity gauge but in probably not converged parameters                                                                                                            #now

        ion_na_reconstructed_GASFIR=np.array(pd.to_numeric(data_rate_delay['ion_NA_reconstructed_GASFIR'].values))
        ion_na_reconstructed_SFA=np.array(pd.to_numeric(0*data_rate_delay['ion_NA_reconstructed_SFA'].values))



        probe=LaserField()
        probe.add_pulse(lam0_probe, I_probe, CEP=-np.pi/2, FWHM=FWHM_probe/AtomicUnits.fs)
        tmin, tmax=probe.get_time_interval()
        time=np.arange(tmin, tmax+1, 1.)
        field_probe_fourier_time=probe.Electric_Field(time)


        plotter = TIPTOEplotter(ion_tRecX, ion_QS, ion_na_GASFIR, ion_na_SFA, ion_na_reconstructed_GASFIR, ion_na_reconstructed_SFA, delay, field_probe_fourier_time, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)
        fig = go.Figure()
        fig = plotter.plotly4()
        # output_path = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/dataOutput/plot_{file_name}_{excitedstates}_maxnhigh.html"
        # fig.write_html(output_path)
        # print(f"Plot saved to: {output_path}")
        #fig.show()
        plotterBA = TIPTOEplotterBA(ion_tRecX, ion_na_GASFIR, ion_na_SFA, ion_SFA_ODE_new, delay, time, AU, lam0_pump, I_pump, lam0_probe, I_probe, FWHM_probe)
        plotterBA.matplot4()


if __name__ == "__main__":
    start_time = time.time()
    main(3)
    end_time = time.time()
    print("time: ", start_time-end_time)
    # start_time = time.time()
    # main(2)
    # end_time = time.time()
    # print("time: ", start_time-end_time)
    # start_time = time.time()
    # main(1)
    # end_time = time.time()
    # print("time: ", end_time-start_time)