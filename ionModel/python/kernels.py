# Copyright (c) 2024 Manoram Agarwal
#
# -*- coding:utf-8 -*-
# @Script: kernels.py
# @Author: Manoram Agarwal
# @Email: manoram.agarwal@mpq.mpg.de
# @Create At: 2024-07-19 15:00:20
# @Last Modified By: Manoram Agarwal
# @Last Modified At: 2025-01-24 12:29:04
# @Description: This python file contains the function that provides three different methods to compute the kernel, rates and probabilities

from scipy.integrate import simpson
import numpy as np
from field_functions import find_zero_crossings, find_extrema_positions, get_momentum_grid, meshgrid, LaserField, AtomicUnits
from field_functions import integrate_oscillating_function_jit as IOF
from field_functions import integrate_oscillating_function_numexpr as IOF_numexpr
from numba import njit, typed, types, prange
import json
# from numba.core import types
# from numba.typed import Dict
from hydrogenTransitions import transitionElement, get_coefficientstRecX, get_eigenEnergy, get_hydrogen_states, transitionElementtest, get_coefficientsNumerical, get_coefficientstRecX_delay, transitionElement_BA
import matplotlib.pyplot as plt
import csv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
########################

from line_profiler import profile
import cProfile
import time
import pstats




@njit(fastmath=False, cache=True)
def Kernel_f_term(EFp, EFm, term1, term2, Ti, params):
    t0,t1,t2=params['t0'],params['t1'],params['t2']
    t3=params['t3']
    t4=params['t4']
    tau=params['tau']
    return t0*((1+0j)*EFp*EFm)**t4*(np.pi/(Ti+1j*tau))**(1.5+t3)*tau**(2.5+t3)*(1j/(Ti+1j*tau)-4*t1*term1+4*Ti**2/(Ti+1j*tau)**2*t2*term2)
    #return t0*EFp*EFm*(np.pi/(Ti+1j*tau))**1.5*tau**2.5*(1j/(Ti+1j*tau)-4*t1*term1+4*Ti**2/(Ti+1j*tau)**2*t2*term2)# *(1j/(T+1j*tau)-4*t1*term1+4*t2*term2)*tau**2.5
    

@njit(fastmath=False, cache=True)
def Kernel_phase_term(term1, term2, DelAbar, DelA2bar, E2diff, Ti, params):
    E_g = params['E_g']
    αPol = params['αPol']
    tau=params['tau']
    e1 = params['e1']
    e2 = params['e2']
    t3= params['t3']
    
    # for QS function, nothing needs to be updated as long as 
    # 1j*(Ti*(2*E_g+DelA2bar-DelAbar**2) + 3*np.pi/4 + αPol * E2diff/2 ) -  tau*(e1*term1) is UNTOUCHED
    # term2 is zero in QS case so changes there are irrelevant
    
    return 1j*(Ti*(2*E_g+DelA2bar-DelAbar**2) + (1.5+t3)*np.pi/2  + αPol * E2diff/2 ) -  tau*(e1*term1+e2*term2*Ti/(Ti+1j*tau))

@njit(parallel=True, fastmath=False, cache=True)
def Kernel_jit_helper(tar, Tar, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a jit optimized implementation
    Args:
        tar (np.ndarray): the grid of moment of ionization
        Tar (np.ndarray): array time T before and after time t that affects that moment of ionization
        EF (np.ndarray): Electric field
        EF2 (np.ndarray): Cummulative Electric field squared
        VP (np.ndarray): Vector potential
        A (np.ndarray): cummulative of the vector potential
        A2 (np.ndarray): cummulative squared of the vector potential
        dT (float64): step size of dense arrays used for EF, EF2 etc
        n (int64): number of steps of dT needed to reach next t[i+1]
        Ti_ar (np.ndarray): indices of time array T for which the values of kernel will actually be stored. 

    Returns:
        f0 (np.ndarray, shape=(T_grid.size, t_grid.size)): 2d grid to store pre-exponential
        phase0 (np.ndarray, shape=(T_grid.size, t_grid.size)):2d grid storing complex agument of the exponential function
    """
    f0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
    phase0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
    for i in prange(Tar.size):
        Ti=Ti_ar[i]
        for j in range(tar.size):
            tj=N+nmin+j*n
            tp=tj+Ti
            tm=tj-Ti
            if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
                VPt = VP[tj]
                T= Ti*dT
                if T!=0.:
                    E2diff=EF2[tp]-EF2[tm]
                    term1 = (VP[tp] - VP[tm])**2/4
                    # xi2=0
                    DelAbar = (intA[tp] - intA[tm])/2/T 
                    term2 = (VP[tp]/2 + VP[tm]/2-DelAbar)**2
                    DelA2bar = (intA2[tp] - intA2[tm])/2/T+VPt**2-2*VPt*DelAbar
                    DelAbar = DelAbar - VPt
                else:
                    E2diff=0
                    term1 = 0
                    DelAbar = 0
                    term2 = 0
                    DelA2bar = 0
                phase0[i, j] = Kernel_phase_term(term1, term2, DelAbar, DelA2bar, E2diff, T, params)
                f0[i, j] = Kernel_f_term(EF[tp], EF[tm], term1, term2, T, params)
    return f0, phase0

@profile
def exact_SFA_jit_helper(tar, Tar, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar, p_grid, Theta_grid, window, p, theta, laser_pulses, excitedStates_boolean):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a jit optimized implementation
    Args:
        tar (np.ndarray): the grid of moment of ionization
        Tar (np.ndarray): array time T before and after time t that affects that moment of ionization
        EF (np.ndarray): Electric field
        EF2 (np.ndarray): Cummulative Electric field squared
        VP (np.ndarray): Vector potential
        A (np.ndarray): cummulative of the vector potential
        A2 (np.ndarray): cummulative squared of the vector potential
        dT (float64): step size of dense arrays used for EF, EF2 etc
        n (int64): number of steps of dT needed to reach next t[i+1]
        Ti_ar (np.ndarray): indices of time array T for which the values of kernel will actually be stored. 

    Returns:
        f0 (np.ndarray, shape=(T_grid.size, t_grid.size)): 2d grid to store pre-exponential
        phase0 (np.ndarray, shape=(T_grid.size, t_grid.size)):2d grid storing complex agument of the exponential function
    """
    E_g = params['E_g']
    excitedStates = params['excitedStates']
    coeffType = params['coeffType']
    gauge = params['gauge']
    get_p_only = params['get_p_only']
    delay = params['delay']
    pz=p*np.cos(theta)
    if excitedStates_boolean:
        # fig = make_subplots(rows=excitedStates, cols=1, shared_xaxes=True, subplot_titles=[f"After state {i}" for i in range(excitedStates)])
        EF_grid = np.arange(-N, N+1, 1.) * dT
        #EF_grid = np.arange(-int(N/2), int((N+1)/2), 1.) * dT
        if coeffType == "trecx":
            if delay != None:
                print("using trecx delay")
                coefficients = get_coefficientstRecX_delay(excitedStates, EF_grid, get_p_only, params, delay)
            else:
                coefficients = get_coefficientstRecX(excitedStates, EF_grid, get_p_only, params)
        elif coeffType == "numerical":
            coefficients = get_coefficientsNumerical(excitedStates, EF_grid, get_p_only, gauge, laser_pulses)     #16.6%
        else:
            raise ValueError("coeffType must be either 'trecx' or 'numerical'")
        eigenEnergy = get_eigenEnergy(excitedStates, get_p_only)
        config = get_hydrogen_states(excitedStates, get_p_only)
        rate = np.zeros(tar.size, dtype=np.cdouble)

        for state_idx in range(excitedStates):
            for state_range_idx in range(excitedStates):
                print(state_idx, state_range_idx)
                f0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
                phase0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
                cLeft = coefficients[state_idx, :]
                cRight = coefficients[state_range_idx, :]
                phaseleft = np.unwrap(np.angle(cLeft))
                phaseright = np.unwrap(np.angle(cRight))
                absleft = np.abs(cLeft)
                absright = np.abs(cRight)

                # if state_idx != state_range_idx:
                #     continue

                # if not (state_idx == state_range_idx and state_idx == 1):
                #     continue

                if params['only_c0_is_1_rest_normal'] and state_idx == 0:
                    print("only_c0_is_1_rest_normal is set to True")
                    absleft, absright = absleft*0+1, absright*0+1
                if params['abs_normal_phase_0'] and state_idx == 0:
                    print("abs_normal_phase_0 is set to True")
                    phaseleft, phaseright = phaseleft*0, phaseright*0

                for i in prange(Tar.size):
                    Ti=Ti_ar[i]
                    for j in range(tar.size):
                        tj=N+nmin+j*n
                        tp=tj+Ti
                        tm=tj-Ti
                        if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
                            VPt = 0 # VP[tj]
                            T= Ti*dT
                            DelA = (intA[tp] - intA[tm])-2*VPt*T
                            VP_p=VP[tp]-VPt
                            VP_m=VP[tm]-VPt
                            f_t_1= 2*np.pi*np.conjugate(transitionElementtest(config[state_idx], p, pz, VP_m, E_g))*transitionElementtest(config[state_range_idx], p, pz, VP_p, E_g)  #25.9%      #for excitedState=1 use only phase of coefficients to see stark effect
                            psquared_m = p**2 + VP_m**2 + 2*pz*VP_m +1e-12
                            pzAz_m = pz + VP_m
                            psquared_p = p**2 + VP_p**2 + 2*pz*VP_p +1e-12
                            pzAz_p = pz + VP_p
                            #f_t_1 = transitionElement_BA(config[state_idx], config[state_range_idx], psquared_m, psquared_p, pzAz_m, pzAz_p, E_g)   #np.conjugate(transitionElement_BA(config[state_idx], psquared_m, pzAz_m, phi_grid, E_g))*transitionElement_BA(config[state_range_idx], psquared_p, pzAz_p, phi_grid, E_g)
                            G1_T_p=np.trapz(f_t_1*np.exp(1j*pz*DelA)*np.sin(theta), Theta_grid)     #17%
                            G1_T=np.trapz(G1_T_p*window*p_grid**2*np.exp(1j*p_grid**2*T), p_grid)   #11.7%
                            DelA = DelA + 2 * VPt * T
                            phase0[i, j] = (intA2[tp] - intA2[tm])/2  + T*VPt**2-VPt*DelA +eigenEnergy[state_idx]*tp*dT-eigenEnergy[state_range_idx]*tm*dT -phaseleft[tm]+phaseright[tp]    #eigenEnergy[state_idx]*(T-tar[j])-eigenEnergy[state_range_idx]*(T+tar[j])
                            f0[i, j] = EF[tp]*EF[tm]*G1_T*absleft[tm]*absright[tp]
                #current_state_rate = 2*np.real(IOF(Tar, f0, (phase0)*1j))*4*np.pi       #21.5% is it the phase? or something else? whats causing the bump
                #current_state_rate = np.real(IOF(Tar, f0, (phase0)*1j))*2
                current_state_rate = np.real(np.trapz(f0*np.exp(phase0*1j), x=Tar, axis=0))*2
                rate += current_state_rate
                # rate = current_state_rate = np.real(IOF(Tar, f0, (phase0)*1j))*4*np.pi
                
                if params['plotting']:
                    config_left_str = f"n={config[state_idx][0]}, l={config[state_idx][1]}, m={config[state_idx][2]}"
                    config_right_str = f"n={config[state_range_idx][0]}, l={config[state_range_idx][1]}, m={config[state_range_idx][2]}"
                    
                    subplot_titles = (f"Rates for transition {state_idx}→{state_range_idx}", f"Coefficients (state {state_idx}, {config_left_str})")
                    fig = make_subplots(rows=1, cols=2, shared_xaxes=False, subplot_titles=subplot_titles,horizontal_spacing=0.15)
                    fig.add_trace(go.Scatter(x=tar, y=np.real(current_state_rate), mode='lines', name=f'Current transition {state_idx}→{state_range_idx} ({config_left_str}→{config_right_str})'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=tar, y=np.real(rate), mode='lines', name=f'Cumulative rate (all transitions so far)'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=EF_grid, y=np.unwrap(np.angle(cLeft)), mode='lines', name=f'|c_{state_idx}|² ({config_left_str})'), row=1, col=2)
                    if state_idx != state_range_idx:
                        fig.add_trace(go.Scatter(x=EF_grid, y=np.angle(cRight), mode='lines',name=f'|c_{state_range_idx}|² ({config_right_str})'), row=1, col=2)
                    
                    fig.update_layout(width=1200, height=400, title_text=f"Transition {state_idx}→{state_range_idx}: {config_left_str}→{config_right_str}")
                    fig.update_xaxes(title_text="Time (a.u.)", row=1, col=1, range=[-50, 50])
                    fig.update_xaxes(title_text="Time (a.u.)", row=1, col=2)
                    fig.update_yaxes(title_text="Rate", row=1, col=1)
                    fig.update_yaxes(title_text="Coefficient magnitude squared", row=1, col=2)
                    fig.show()

        return rate
    else:
        f0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
        phase0 = np.zeros((Tar.size, tar.size), dtype=np.cdouble)
        for i in prange(Tar.size):
            Ti=Ti_ar[i]
            for j in range(tar.size):
                tj=N+nmin+j*n
                tp=tj+Ti
                tm=tj-Ti
                if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
                    VPt = 0 # VP[tj]
                    T= Ti*dT
                    DelA = (intA[tp] - intA[tm])-2*VPt*T
                    VP_p=VP[tp]-VPt
                    VP_m=VP[tm]-VPt 
                    f_t_1= (pz+VP_p)/(p**2+VP_p**2+2*pz*VP_p+2*E_g)**3*(pz+VP_m)/(p**2+VP_m**2+2*pz*VP_m+2*E_g)**3
                    G1_T_p=np.trapz(f_t_1*np.exp(1j*pz*DelA)*np.sin(theta), Theta_grid)
                    G1_T=np.trapz(G1_T_p*window*p_grid**2*np.exp(1j*p_grid**2*T), p_grid)
                    DelA = DelA + 2 * VPt * T
                    phase0[i, j]  = (intA2[tp] - intA2[tm])/2 +2*E_g*T + T*VPt**2-VPt*DelA
                    f0[i, j] = EF[tp]*EF[tm]*2**9 *(2*E_g)**2.5/np.pi*G1_T #should be 7 instead of 9?
        return f0, phase0*1j

def Kernel_jit(t_grid, T_grid, laser_field, param_dict, kernel_type="GASFIR", excitedStates=False):
    """ return the kernel(t_grid,T_grid) for a given laser field computed with provided parameters using a jit optimized implementation
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        T_grid (np.ndarray): array time T before and after time t that affects that moment of ionization 
        laser_field: an object of LaserField
        param_dict:dictionnary containing the following keys:-
        e1 (float64): exponential factor responsible for decay wrt xi1**2
        e2 (float64): exponential factor responsible for decay wrt (A_bar-xi2)**2
        a0 (float64): Normalization factor
        a1 (float64): gamma, related to tuning the gaussian width
        a2 (float64): 1/(1-a2*1j*T/a1/2)  in denominators
        a3 (float64): power of the denominator
        x1 (float64): xi1**2 prefactor in numerator
        x2 (float64): xi2**2 prefactor in numerator
        E_g (float64): band gap or the Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
    Returns:
        np.ndarray, shape=(T_grid.size, t_grid.size): the kernel"""
    t=t_grid
    T=T_grid
    dt=min(np.round(np.diff(t),4))
    dT=min(np.diff(T))#/2
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=max(abs(t_min), abs(t_max))
    assert dt%dT==0
    n=int(dt//dT)
    N=int(tau_injection//dT)+1
    # nmax=int(t[-1]//dT)
    nmin=int(t[0]//dT)
    tAr=np.arange(-N, N+1, 1) * dT
    VP=laser_field.Vector_potential(tAr)
    EF=laser_field.Electric_Field(tAr)
    intA=laser_field.int_A(tAr) # np.cumsum(VP*dT)
    intA2=laser_field.int_A2(tAr) # np.cumsum(VP**2*dT)
    EF2=laser_field.int_E2(tAr) # np.cumsum(EF**2*dT)
    Ti_ar=(T//dT).astype(np.int64)
    # f0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    # phase0 = np.zeros((T.size, t.size), dtype=np.cdouble)
    params= typed.Dict.empty(key_type=types.unicode_type, value_type=types.complex128)
    # for key, value in param_dict.items():     i dont have numba so dont use it for now
    #     params[key]=value
    params = param_dict
    if kernel_type=="exact_SFA":
        div_theta=param_dict["div_theta"]
        div_p=param_dict["div_p"]
        p_grid, Theta_grid, window = get_momentum_grid(div_p, div_theta, laser_field, Ip=param_dict["E_g"])
        #print(p_grid.size, Theta_grid.size)
        p, theta = meshgrid(p_grid, Theta_grid)
        if excitedStates:
            return exact_SFA_jit_helper(t, T, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar, p_grid, Theta_grid, window, p, theta, excitedStates_boolean=excitedStates, laser_pulses=laser_field)
        else:
            f0, phase0 = exact_SFA_jit_helper(t, T, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar, p_grid, Theta_grid, window, p, theta, excitedStates_boolean=excitedStates, laser_pulses=laser_field)
    elif kernel_type=="GASFIR":
        f0, phase0= Kernel_jit_helper(t, T, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar)
    else:
        return None, None
    return f0, phase0

def IonRate(t_grid, laser_field, param_dict, dT, kernel_type="GASFIR", excitedStates=False):
    """ return the ionization rate for a define pulse computed with provided parameters 
    Args:
        t_grid (np.ndarray): the grid of moment of ionization   
        laser_field: an object of LaserField
        param_dict:dictionnary containing the following keys:-

    Returns:
        np.ndarray, shape=(t_grid.size): the ionization rates for given time grid"""
    if type(t_grid) is str:
        t_grid=json.loads(t_grid)
    t_grid=np.array(t_grid, dtype=np.float64)
    dt=min(np.diff(t_grid))
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=int(max(abs(t_min), abs(t_max)))
    T_grid=np.arange(0, tau_injection+dT, dT, dtype=np.float64)
    if (excitedStates!=0):
        rate=Kernel_jit(t_grid, T_grid, laser_field, param_dict, kernel_type=kernel_type, excitedStates=excitedStates)
    else:
        f, phase=Kernel_jit(t_grid, T_grid, laser_field, param_dict, kernel_type=kernel_type, excitedStates=excitedStates)
        rate=np.real(IOF(T_grid, f, phase))

    #rate=2*np.real(np.trapz(f*np.exp(phase), x=T_grid, axis=0))
    # the factor two is to account for the fact that the kernel 
    # is symmetric in T and we integrate from 0 to inf
    return np.array(rate)


#@njit(parallel=True,fastmath = False, cache=True)
def QSRate(field, param_dict):
    """ return the ionization rate for a define pulse computed with provided parameters  
    Args:
        field (float64/np.ndarray): the grid of electric field strengths

    Returns:
        np.ndarray, shape=(field.size): the ionization rates for given array of electric field strengths"""
    params= typed.Dict.empty(key_type=types.unicode_type, value_type=types.float64)
    param_dict["e2"]=0
    for key, value in param_dict.items():
        params[key]=value
    E_g = params['E_g']
    αPol = params['αPol']
    tau=params['tau']
    e1 = params['e1']
    
    tmp=np.zeros(field.size, dtype=np.complex128)
    field = np.abs(field)
    cond=(field>0)
    field=field[cond]
    
    T_saddle = 1j*(-e1*field*tau + np.sqrt(field**2*(αPol+e1**2*tau**2) + 2*E_g))/field
    
    E2diff=2*field**2*T_saddle
    term1 = (-2*field*T_saddle)**2/4
    DelAbar = 0
    term2 = 0
    DelA2bar = (field*T_saddle)**2/3
    
    saddle_contribution=np.sqrt(np.pi/field/np.sqrt(2 * E_g + field**2 * (e1**2 * tau**2 + αPol)))
    tmp[cond] = np.exp(Kernel_phase_term(term1, term2, DelAbar, DelA2bar, E2diff, T_saddle, params))*Kernel_f_term(field, field, term1, term2, T_saddle, params)*saddle_contribution
    rate=np.real(tmp)
    assert np.all(abs(np.imag(rate))<=1e-10)
    return rate

#@njit(parallel=True,fastmath = False, cache=True)
def analyticalRate(t_grid, laser_field,param_dict):
    """ return the ionization rate for a define pulse computed with provided parameters 
     
    Args:
        t_grid (np.ndarray): the grid of moment of ionization
        laser_field: an object of LaserField
        param_dict:dictionnary containing the following keys:-
        ## to be updated ##
    Returns:
        np.ndarray, shape=(t_grid.size): the quasi-static ionization rates for given time grid """

    
    field = np.abs(laser_field.Electric_Field(t_grid))
    return QSRate(field, param_dict)

#@njit(parallel=True,fastmath = False, cache=True)
def analyticalProb(laser_field, param_dict, dt=2):
    """ return the ionization rate for a define pulse computed with provided parameters 
     
    Args:
        laser_field: an object of LaserField
        dt (float64): time steps at which ion. rate is computed to integrate over the time grid

    Returns:
        float64: the ionization probability for given pulse"""
    t_min, t_max = laser_field.get_time_interval()
    t_grid = np.arange(t_min, t_max+dt, dt)
    tmp = analyticalRate(t_grid, laser_field, param_dict)
    return np.double(simpson(tmp, x=t_grid, axis=-1, even='simpson'))

def IonProb(laser_field, param_dict, dt=2., dT=0.25, filterTreshold=0.0, kernel_type="GASFIR", ret_Rate=False, excitedStates=0):
    """ return the ionization probability for the defined pulse computed with provided parameters 
    Note: by default this function filters out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) 
    Args:
        laser_field: an object of LaserField
        param_dict:dictionnary containing the following keys:-
        e1 (float64): exponential factor responsible for decay wrt xi1**2
        e2 (float64): exponential factor responsible for decay wrt (A_bar-xi2)**2
        a0 (float64): Normalization factor
        a1 (float64): gamma, related to tuning the gaussian width
        a2 (float64): 1/(1-a2*1j*T/a1/2)  in denominators
        a3 (float64): power of the denominator
        x1 (float64): xi1**2 prefactor in numerator
        x2 (float64): xi2**2 prefactor in numerator
        E_g (float64): band gap or the Ionization potential
        αPol (float64): Polarization computed analytically to account for stark shift
        dt (float64): time steps at which ion. rate is computed
        dT (float64): time step make it smaller if computations not converged

    Returns:
        float64: the ionization probability for given pulse"""
        
    t_min, t_max = laser_field.get_time_interval()
    tau_injection=int(max(abs(t_min), abs(t_max)))+1
    if np.any(np.array(laser_field.get_central_wavelength())<140):
        dt=0.5
        dT=0.25
    
    elif kernel_type=="exact_SFA":
        dt=np.ceil(laser_field.get_central_wavelength()/400)
    t_grid=np.arange(-tau_injection,tau_injection+dt, dt)
    
    if filterTreshold > 0:
        ### filter out the t_grid such that |E(t_grid)|>=1% of max(E(t_grid)) ###
        ElecField=lambda t: laser_field.Electric_Field(t)
        extr=find_extrema_positions(t_grid, ElecField(t_grid))
        Fextr=ElecField(extr)
        extr=extr[np.abs(Fextr)>=max(np.abs(Fextr))*filterTreshold]
        if extr[0]>0 and extr[-1]<0:
            extr=extr[::-1]
        ### smartly take the t_grid uptill the correspondin zero crossing rather than abruptly ending a a sub-cycle peak
        zeroCr=find_zero_crossings(t_grid, ElecField(t_grid))
        if zeroCr[0]>0 and zeroCr[-1]<0:
            zeroCr=zeroCr[::-1]
        t_grid=np.arange(np.floor(zeroCr[zeroCr<extr[0]][-1]), np.ceil(zeroCr[zeroCr>extr[-1]][0])+dt, dt, dtype=np.float64)
    rate=IonRate(t_grid, laser_field, param_dict, dT, kernel_type=kernel_type, excitedStates=excitedStates)
    # return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))
    if ret_Rate:
        return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson'))), (t_grid, rate)
    else:
        return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))




### if called directly, this file can be used for profiling

if __name__ == '__main__':
    start_time = time.time()
    profiler = cProfile.Profile()
    profiler.enable()





    params = {'E_g': 0.5, 'αPol': 4.51, "div_p":2**-4*8, "div_theta":1*2, 'lam0': 450, 'intensity': 8e13, 'cep': 0, 'excitedStates': 2, 'coeffType': 'trecx', 'gauge': 'length', 'get_p_only': True, 'only_c0_is_1_rest_normal': True, 'delay': -224.97}
    
    laser_pulses = LaserField(cache_results=True)

    laser_pulses.add_pulse(params['lam0'], params['intensity'], params['cep'], params['lam0']/ AtomicUnits.nm / AtomicUnits.speed_of_light)
    laser_pulses.add_pulse(250, 8e9, -np.pi/2, 250/ AtomicUnits.nm / AtomicUnits.speed_of_light, t0=-144.91)
    t_min, t_max = laser_pulses.get_time_interval()
    time_recon_1= np.arange(int(t_min), int(t_max)+1, 1.)

    rateExcited_1 = IonRate(time_recon_1, laser_pulses, params, dT=0.5/2, kernel_type='exact_SFA', excitedStates=True)
    laser_pulses.reset()


    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Print top 20 functions
    end_time = time.time()
    print(f"Function execution time: {end_time - start_time} seconds")