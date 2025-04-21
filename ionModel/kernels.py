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
from field_functions import LaserField, find_zero_crossings, find_extrema_positions, create_pulse, get_momentum_grid, meshgrid
from field_functions import integrate_oscillating_function_numexpr as IOF
from __init__ import soft_window
from numba import njit, typed, types, prange
import matplotlib.pyplot as plt
import json
# from numba.core import types
# from numba.typed import Dict


########################





@njit(fastmath=False, cache=True)
def Kernel_f_term(EFp, EFm, term1, term2, Ti, params):
    t0,t1,t2=params['t0'],params['t1'],params['t2']
    t3=params['t3']
    t4=params['t4']
    tau=params['tau']
    return t0*((1+0j)*EFp*EFm)**t4*(np.pi/(Ti+1j*tau))**(1.5+t3)*tau**(2.5+t3)*(1j/(Ti+1j*tau)-4*t1*term1+4*Ti**2/(Ti+1j*tau)**2*t2*term2)

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
    
    return 1j*(Ti*(2*E_g+DelA2bar-DelAbar**2) + (1.5+t3)*np.pi/2 + αPol * E2diff/2 ) -  tau*(e1*term1+e2*term2*Ti/(Ti+1j*tau))

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
                    # phase0[i, j] =  1j*(T*(2*E_g+DelA2bar-DelAbar**2) + 3*np.pi/4 + αPol * E2diff/2 ) -  tau*(e1*term1+term2*T/(T+1j*tau))
                    # f0[i, j] = t0*EF[tp]*EF[tm]*(np.pi/(T+1j*tau))**1.5*tau**2.5*(1j/(T+1j*tau)-4*t1*term1+4*T**2/(T+1j*tau)**2*t2*term2)# *(1j/(T+1j*tau)-4*t1*term1+4*t2*term2)*tau**2.5# 
                    phase0[i, j] = Kernel_phase_term(term1, term2, DelAbar, DelA2bar, E2diff, T, params)
                    f0[i, j] = Kernel_f_term(EF[tp], EF[tm], term1, term2, T, params)
                    #
                else:
                    t0=params['t0']
                    t3=params['t3']
                    t4=params['t4']
                    f0[i, j] =  t0*((1+0j)*EF[tp]*EF[tm])**t4*(np.pi/1j)**(1.5+t3)
                    phase0[i, j] = 1j*(np.pi/2)*(1.5+t3)
                    # f0[i, j] =  Kernel_f_term(EF[tp], EF[tm], 0, 0, T, params)
                    # phase0[i, j] = Kernel_phase_term(0, 0, 0, 0, 0, T, params)
    return f0, phase0

@njit(parallel=True, fastmath=False, cache=True)
def exact_SFA_jit_helper(tar, Tar, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar, p_grid, Theta_grid, window, p, theta):
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
    E_g = params['E_g']
    pz=p*np.cos(theta)
    for i in prange(Tar.size):
        Ti=Ti_ar[i]
        for j in range(tar.size):
            tj=N+nmin+j*n
            tp=tj+Ti
            tm=tj-Ti
            if tp>=0 and tp<EF.size and tm>=0 and tm<EF.size:
                VPt = VP[tj]
                T= Ti*dT
                DelA = (intA[tp] - intA[tm])-2*VPt*T
                VP_p=VP[tp]-VPt
                VP_m=VP[tm]-VPt
                f_t_1= (pz+VP_p)*(pz+VP_m)/(p**2+VP_p**2+2*pz*VP_p+2*E_g)**3/(p**2+VP_m**2+2*pz*VP_m+2*E_g)**3
                # Apparently it uses canonical momentum
                # This is the formula used here: \bm{d}_\mathrm{1s, H}=2^{7/2} (2 I_p)^{5/4} \frac{ \bm{p}}{(\bm{p}^2+2I_p)^3}, 
                # i need to replace |\psi_0> with \sum_n |\Psi_n> but for n>>1 there quite complex
                # later i need to replace E_g with c_n(t) \exp\{-i E_n t\} but i need to interpolate the time grid

                # Problem: i dont know how trecx stores the eigenvectors, and therefore i cant just take |n,m,l> 
                # and weight it with the coefficients from tRecX
                G1_T_p=np.trapz(f_t_1*np.exp(1j*pz*DelA)*np.sin(theta), Theta_grid)#IOF(p_grid,f_t_1,phase_t)
                # G1_T=IOF(p_grid,G1_T_p*window*p_grid**2,p_grid**2*T)
                G1_T=np.trapz(G1_T_p*window*p_grid**2*np.exp(1j*p_grid**2*T), p_grid)
                phase0[i, j]  = (intA2[tp] - intA2[tm])/2 +2*E_g*T + T*VPt**2-VPt*DelA
                f0[i, j] = EF[tp]*EF[tm]*2**9 *(2*E_g)**2.5/np.pi*G1_T
    return f0, phase0

def Kernel_jit(t_grid, T_grid, laser_field, param_dict, kernel_type="GASFIR"):
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
    # print(dt, dT)
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
    for key, value in param_dict.items():
        params[key]=value
    if kernel_type=="exact_SFA":
        div_theta=param_dict["div_theta"]
        div_p=param_dict["div_p"]
        p_grid, Theta_grid, window = get_momentum_grid(div_p, div_theta, laser_field, Ip=param_dict["E_g"])
        #print(p_grid.size, Theta_grid.size)
        p, theta = meshgrid(p_grid, Theta_grid)
        f0, phase0 = exact_SFA_jit_helper(t, T, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar, p_grid, Theta_grid, window, p, theta)
    elif kernel_type=="GASFIR":
        f0, phase0= Kernel_jit_helper(t, T, params, EF, EF2, VP, intA, intA2, dT, N, n, nmin, Ti_ar)
    else:
        return None, None
    return f0, phase0


def IonRate(t_grid, laser_field, param_dict, dT, kernel_type="GASFIR"):
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
    tau_injection=max(abs(t_min), abs(t_max))
    T_grid=np.arange(0, tau_injection+dT, dT, dtype=np.float64)
    f, phase=Kernel_jit(t_grid, T_grid, laser_field, param_dict, kernel_type=kernel_type)
    rate=2*np.real(IOF(T_grid, f, phase))
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

def IonProb(laser_field, param_dict, dt=2., dT=0.25, filterTreshold=0.0, kernel_type="GASFIR", ret_Rate=False):
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
    rate=IonRate(t_grid, laser_field, param_dict, dT, kernel_type=kernel_type)
    # return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))
    if ret_Rate:
        return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson'))), (t_grid, rate)
    else:
        return 1-np.exp(-np.double(simpson(rate, x=t_grid, axis=-1, even='simpson')))




### if called directly, this file can be used for profiling

if __name__ == '__main__':
    ## the first function call simply ensures that the function is called at least once and numba compilation is done
    laser_field = create_pulse(600, 1e14, 0, 3)
    # print(laser_field.Electric_Field(np.arange(-400, 400.125, 1)))
    print(IonProb(laser_field, {"e1":1., "e2":0., "a0":1., "a1":1., "a2":1., "a3":1., "x1":1., "x2":1., "E_g":0.5, "αPol":0, "div_p":2**-4, "div_theta":1}, dT=0.25, kernel_type="exact_SFA"))
    # IonProb(laser_field, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., E_g=0.5, αPol=0)
    # print('Profiling...')
    # print("user is adviced to change the dummy parameters (laser_field, param_dict) to see the profiling results")
    
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()
    # laser_field.reset()
    # laser_field.add_pulse(central_wavelength=400, peak_intensity=1e04, CEP=0, FWHM=50.)
    # IonProb(laser_field, Multiplier=2658.86, e0=2.5, a0=20., b0=40.0001, a1=1, b1=10., b2=3., c2=-5.33333, p1=4.04918, d1=10., E_g=0.5, αPol=0)
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats()