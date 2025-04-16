
# Copyright (c) 2024 Manoram Agarwal
#
# -*- coding:utf-8 -*-
# @Script: field_functions.py
# @Author: Manoram Agarwal
# @Email: manoram.agarwal@mpq.mpg.de
# @Create At: 2024-07-19 14:49:04
# @Description: This python file contains all functions required to define the vector potential and the electric field. Additional functions like the integration of the vector potential are also defined.


from numba import njit, prange
import numexpr as ne
import numpy as np


class AtomicUnits:
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


@njit()
def soft_window(x_grid, x_begin, x_end):
    """ Compute a soft window.

    Given a vector 'x_grid' and two numbers ('x_begin'and 'x_end'),
    the function returns a vector that contains a 'soft window',
    which gradually changes from 1 at x=x_begin to 0 at x=x_end,
    being constant outside of this range. The value of 'x_begin'
    is allowed to be larger than that of 'x_end'.
    """
    window = np.zeros(len(x_grid))
    # determine the indices where the window begins and ends
    x_min = min((x_begin, x_end))
    x_max = max((x_begin, x_end))
    u = np.nonzero(x_grid < x_min)[0]
    i1 = min((u[-1] + 1, len(x_grid))) if len(u) > 0 else 0
    u = np.nonzero(x_grid > x_max)[0]
    i2 = u[0] if len(u) > 0 else len(x_grid)
    # evaluate the window function
    if x_begin <= x_end:
        window[:i1] = 1
        if i2 > i1:
            window[i1:i2] = \
                np.cos(np.pi/2.0 * (x_grid[i1:i2]-x_min) / (x_max - x_min))**2
    else:
        window[i2:] = 1
        if i2 > i1:
            window[i1:i2] = \
                np.sin(np.pi/2.0 * (x_grid[i1:i2]-x_min) / (x_max - x_min))**2
    return window



@njit(fastmath = False,cache = True)
def cosN_vector_potential(t, A0, w0, tau, cep, N):
    return - A0 * np.cos(0.5*np.pi*t/tau)**N * np.sin(w0*t-cep)

@njit(fastmath = False,cache = True)
def cosN_electric_field(t, A0, w0, tau, cep, N):
    x = w0 * t - cep
    y = 0.5 * np.pi*t / tau
    return A0*(np.cos(y))**(N-1) * (w0*np.cos(x)*np.cos(y) - (N*np.pi*np.sin(x)*np.sin(y))/(2.*tau))

def cos8_int_A(t, A0, w0, tau, cep):
    " calculate the cumulative integral of A(t) for the cos8 envelope "
    x = w0 * t - cep
    theta = w0 * tau
    theta2 = theta**2
    result = (A0*((-20160*np.pi**8*np.cos(cep + theta)) /
        (576*np.pi**8*w0 + theta2*(-820*np.pi**6*w0 + 
        theta2*(273*np.pi**4*w0 + theta2*(-30*np.pi**2*w0 + theta2*w0)))) + 
        np.cos(x)*(35/w0 + theta*tau*
            ((-56*np.cos((np.pi*t)/tau))/(np.pi**2 - theta2) + 
            (28*np.cos((2*np.pi*t)/tau))/(-4*np.pi**2 + theta2) + 
            (8*np.cos((3*np.pi*t)/tau))/(-9*np.pi**2 + theta2) + 
            np.cos((4*np.pi*t)/tau)/(-16*np.pi**2 + theta2))) - 
        4*np.pi*tau*np.sin(x)*((14*np.sin((np.pi*t)/tau))/(np.pi**2 - theta2) + 
        (14*np.sin((2*np.pi*t)/tau))/(4*np.pi**2 - theta2) + 
        (6*np.sin((3*np.pi*t)/tau))/(9*np.pi**2 - theta2) + 
        np.sin((4*np.pi*t)/tau)/(16*np.pi**2 - theta2))))/128.
    return result

# @njit(parallel=True, fastmath = False,cache = True)
def cos8_int_A2(t, A0, w0, tau, cep):
    " calculate the cumulative integral of A^2(t) for the cos8 envelope "
    t = np.asarray(t)
    x = w0 * t - cep
    theta = w0 * tau
    thetaOverPi = theta / np.pi
    denominator = (-4 + thetaOverPi)*(-3 + thetaOverPi)*(-2 + thetaOverPi)*(-1 + thetaOverPi)* \
      (1 + thetaOverPi)*(2 + thetaOverPi)*(3 + thetaOverPi)*(4 + thetaOverPi)* \
        (-7 + 2*thetaOverPi)*(-5 + 2*thetaOverPi)*(-3 + 2*thetaOverPi)* \
         (-1 + 2*thetaOverPi)*(1 + 2*thetaOverPi)*(3 + 2*thetaOverPi)* \
          (5 + 2*thetaOverPi)*(7 + 2*thetaOverPi)*w0
    c = np.zeros((18, t.size))
    c[0, :] =(-638512875*(-2*(cep + x) + np.sin(2*(cep + theta)) + np.sin(2*x)))/2048.
    c[1, :] =(14175*(720720*np.pi + 7*np.sin(2*x - (8*np.pi*t)/tau) + 128*np.sin(2*x - (7*np.pi*t)/tau) + 1120*np.sin(2*x - (6*np.pi*t)/tau) + 6272*np.sin(2*x - (5*np.pi*t)/tau) + 25480*np.sin(2*x - (4*np.pi*t)/tau) + 81536*np.sin(2*x - (3*np.pi*t)/tau) + 224224*np.sin(2*x - (2*np.pi*t)/tau) + 640640*np.sin(2*x - (np.pi*t)/tau) - 224224*np.sin(2*(x + (np.pi*t)/tau)) - 640640*np.sin(2*x + (np.pi*t)/tau) - 81536*np.sin(2*x + (3*np.pi*t)/tau) - 25480*np.sin(2*x + (4*np.pi*t)/tau) - 6272*np.sin(2*x + (5*np.pi*t)/tau) - 1120*np.sin(2*x + (6*np.pi*t)/tau) - 128*np.sin(2*x + (7*np.pi*t)/tau) - 7*np.sin(2*x + (8*np.pi*t)/tau) + 1281280*np.sin((np.pi*t)/tau) + 448448*np.sin((2*np.pi*t)/tau) + 163072*np.sin((3*np.pi*t)/tau) + 50960*np.sin((4*np.pi*t)/tau) + 12544*np.sin((5*np.pi*t)/tau) + 2240*np.sin((6*np.pi*t)/tau) + 256*np.sin((7*np.pi*t)/tau) + 14*np.sin((8*np.pi*t)/tau)))/(16384.*np.pi)
    c[2, :] =(135*(-1849417284*t*w0 + 924708642*np.sin(2*x) + 735*np.sin(2*x - (8*np.pi*t)/tau) + 15360*np.sin(2*x - (7*np.pi*t)/tau) + 156800*np.sin(2*x - (6*np.pi*t)/tau) + 1053696*np.sin(2*x - (5*np.pi*t)/tau) + 5350800*np.sin(2*x - (4*np.pi*t)/tau) + 22830080*np.sin(2*x - (3*np.pi*t)/tau) + 94174080*np.sin(2*x - (2*np.pi*t)/tau) + 538137600*np.sin(2*x - (np.pi*t)/tau) + 94174080*np.sin(2*(x + (np.pi*t)/tau)) + 538137600*np.sin(2*x + (np.pi*t)/tau) + 22830080*np.sin(2*x + (3*np.pi*t)/tau) + 5350800*np.sin(2*x + (4*np.pi*t)/tau) + 1053696*np.sin(2*x + (5*np.pi*t)/tau) + 156800*np.sin(2*x + (6*np.pi*t)/tau) + 15360*np.sin(2*x + (7*np.pi*t)/tau) + 735*np.sin(2*x + (8*np.pi*t)/tau)))/(65536.*np.pi**2)
    c[3, :] =(-9*(388377629640*np.pi + 3733534*np.sin(2*x - (8*np.pi*t)/tau) + 68054336*np.sin(2*x - (7*np.pi*t)/tau) + 592563440*np.sin(2*x - (6*np.pi*t)/tau) + 3291310400*np.sin(2*x - (5*np.pi*t)/tau) + 13168688260*np.sin(2*x - (4*np.pi*t)/tau) + 40741460032*np.sin(2*x - (3*np.pi*t)/tau) + 101052039088*np.sin(2*x - (2*np.pi*t)/tau) + 119206767680*np.sin(2*x - (np.pi*t)/tau) - 101052039088*np.sin(2*(x + (np.pi*t)/tau)) - 119206767680*np.sin(2*x + (np.pi*t)/tau) - 40741460032*np.sin(2*x + (3*np.pi*t)/tau) - 13168688260*np.sin(2*x + (4*np.pi*t)/tau) - 3291310400*np.sin(2*x + (5*np.pi*t)/tau) - 592563440*np.sin(2*x + (6*np.pi*t)/tau) - 68054336*np.sin(2*x + (7*np.pi*t)/tau) - 3733534*np.sin(2*x + (8*np.pi*t)/tau) + 690449119360*np.sin((np.pi*t)/tau) + 241657191776*np.sin((2*np.pi*t)/tau) + 87875342464*np.sin((3*np.pi*t)/tau) + 27461044520*np.sin((4*np.pi*t)/tau) + 6759641728*np.sin((5*np.pi*t)/tau) + 1207078880*np.sin((6*np.pi*t)/tau) + 137951872*np.sin((7*np.pi*t)/tau) + 7544243*np.sin((8*np.pi*t)/tau)))/(917504.*np.pi**3)
    c[4, :] =(-3*(-545402191620*t*w0 + 272701095810*np.sin(2*x) + 800043*np.sin(2*x - (8*np.pi*t)/tau) + 16666368*np.sin(2*x - (7*np.pi*t)/tau) + 169303840*np.sin(2*x - (6*np.pi*t)/tau) + 1128449280*np.sin(2*x - (5*np.pi*t)/tau) + 5643723540*np.sin(2*x - (4*np.pi*t)/tau) + 23280834304*np.sin(2*x - (3*np.pi*t)/tau) + 86616033504*np.sin(2*x - (2*np.pi*t)/tau) + 204354458880*np.sin(2*x - (np.pi*t)/tau) + 86616033504*np.sin(2*(x + (np.pi*t)/tau)) + 204354458880*np.sin(2*x + (np.pi*t)/tau) + 23280834304*np.sin(2*x + (3*np.pi*t)/tau) + 5643723540*np.sin(2*x + (4*np.pi*t)/tau) + 1128449280*np.sin(2*x + (5*np.pi*t)/tau) + 169303840*np.sin(2*x + (6*np.pi*t)/tau) + 16666368*np.sin(2*x + (7*np.pi*t)/tau) + 800043*np.sin(2*x + (8*np.pi*t)/tau)))/(262144.*np.pi**4)
    c[5, :] =(22906892048040*np.pi + 214082960*np.sin(2*x - (8*np.pi*t)/tau) + 3868271680*np.sin(2*x - (7*np.pi*t)/tau) + 33227092080*np.sin(2*x - (6*np.pi*t)/tau) + 180387188800*np.sin(2*x - (5*np.pi*t)/tau) + 691321423520*np.sin(2*x - (4*np.pi*t)/tau) + 1939623416640*np.sin(2*x - (3*np.pi*t)/tau) + 3488715230000*np.sin(2*x - (2*np.pi*t)/tau) + 3195907274560*np.sin(2*x - (np.pi*t)/tau) - 3488715230000*np.sin(2*(x + (np.pi*t)/tau)) - 3195907274560*np.sin(2*x + (np.pi*t)/tau) - 1939623416640*np.sin(2*x + (3*np.pi*t)/tau) - 691321423520*np.sin(2*x + (4*np.pi*t)/tau) - 180387188800*np.sin(2*x + (5*np.pi*t)/tau) - 33227092080*np.sin(2*x + (6*np.pi*t)/tau) - 3868271680*np.sin(2*x + (7*np.pi*t)/tau) - 214082960*np.sin(2*x + (8*np.pi*t)/tau) + 40723363640960*np.sin((np.pi*t)/tau) + 14253177274336*np.sin((2*np.pi*t)/tau) + 5182973554304*np.sin((3*np.pi*t)/tau) + 1619679235720*np.sin((4*np.pi*t)/tau) + 398690273408*np.sin((5*np.pi*t)/tau) + 71194691680*np.sin((6*np.pi*t)/tau) + 8136536192*np.sin((7*np.pi*t)/tau) + 444966823*np.sin((8*np.pi*t)/tau))/(3.670016e6*np.pi**5)
    c[6, :] =(455*(-1228462092*t*w0 + 614231046*np.sin(2*x) + 4201*np.sin(2*x - (8*np.pi*t)/tau) + 86752*np.sin(2*x - (7*np.pi*t)/tau) + 869364*np.sin(2*x - (6*np.pi*t)/tau) + 5663648*np.sin(2*x - (5*np.pi*t)/tau) + 27131924*np.sin(2*x - (4*np.pi*t)/tau) + 101497824*np.sin(2*x - (3*np.pi*t)/tau) + 273839500*np.sin(2*x - (2*np.pi*t)/tau) + 501712288*np.sin(2*x - (np.pi*t)/tau) + 273839500*np.sin(2*(x + (np.pi*t)/tau)) + 501712288*np.sin(2*x + (np.pi*t)/tau) + 101497824*np.sin(2*x + (3*np.pi*t)/tau) + 27131924*np.sin(2*x + (4*np.pi*t)/tau) + 5663648*np.sin(2*x + (5*np.pi*t)/tau) + 869364*np.sin(2*x + (6*np.pi*t)/tau) + 86752*np.sin(2*x + (7*np.pi*t)/tau) + 4201*np.sin(2*x + (8*np.pi*t)/tau)))/(131072.*np.pi**6)
    c[7, :] =(-13*(85992346440*np.pi + 761684*np.sin(2*x - (8*np.pi*t)/tau) + 13537216*np.sin(2*x - (7*np.pi*t)/tau) + 113347080*np.sin(2*x - (6*np.pi*t)/tau) + 589758400*np.sin(2*x - (5*np.pi*t)/tau) + 2090516120*np.sin(2*x - (4*np.pi*t)/tau) + 4991861952*np.sin(2*x - (3*np.pi*t)/tau) + 7584409448*np.sin(2*x - (2*np.pi*t)/tau) + 6197920960*np.sin(2*x - (np.pi*t)/tau) - 7584409448*np.sin(2*(x + (np.pi*t)/tau)) - 6197920960*np.sin(2*x + (np.pi*t)/tau) - 4991861952*np.sin(2*x + (3*np.pi*t)/tau) - 2090516120*np.sin(2*x + (4*np.pi*t)/tau) - 589758400*np.sin(2*x + (5*np.pi*t)/tau) - 113347080*np.sin(2*x + (6*np.pi*t)/tau) - 13537216*np.sin(2*x + (7*np.pi*t)/tau) - 761684*np.sin(2*x + (8*np.pi*t)/tau) + 152875282560*np.sin((np.pi*t)/tau) + 53506348896*np.sin((2*np.pi*t)/tau) + 19456854144*np.sin((3*np.pi*t)/tau) + 6080266920*np.sin((4*np.pi*t)/tau) + 1496681088*np.sin((5*np.pi*t)/tau) + 267264480*np.sin((6*np.pi*t)/tau) + 30544512*np.sin((7*np.pi*t)/tau) + 1670403*np.sin((8*np.pi*t)/tau)))/(262144.*np.pi**7)
    c[8, :] =(-143*(-2653047540*t*w0 + 1326523770*np.sin(2*x) + 17311*np.sin(2*x - (8*np.pi*t)/tau) + 351616*np.sin(2*x - (7*np.pi*t)/tau) + 3434760*np.sin(2*x - (6*np.pi*t)/tau) + 21445760*np.sin(2*x - (5*np.pi*t)/tau) + 95023460*np.sin(2*x - (4*np.pi*t)/tau) + 302537088*np.sin(2*x - (3*np.pi*t)/tau) + 689491768*np.sin(2*x - (2*np.pi*t)/tau) + 1126894720*np.sin(2*x - (np.pi*t)/tau) + 689491768*np.sin(2*(x + (np.pi*t)/tau)) + 1126894720*np.sin(2*x + (np.pi*t)/tau) + 302537088*np.sin(2*x + (3*np.pi*t)/tau) + 95023460*np.sin(2*x + (4*np.pi*t)/tau) + 21445760*np.sin(2*x + (5*np.pi*t)/tau) + 3434760*np.sin(2*x + (6*np.pi*t)/tau) + 351616*np.sin(2*x + (7*np.pi*t)/tau) + 17311*np.sin(2*x + (8*np.pi*t)/tau)))/(262144.*np.pi**8)
    c[9, :] =(143*(37142665560*np.pi + 300160*np.sin(2*x - (8*np.pi*t)/tau) + 5190080*np.sin(2*x - (7*np.pi*t)/tau) + 41690880*np.sin(2*x - (6*np.pi*t)/tau) + 203134400*np.sin(2*x - (5*np.pi*t)/tau) + 647960320*np.sin(2*x - (4*np.pi*t)/tau) + 1378319040*np.sin(2*x - (3*np.pi*t)/tau) + 1902611200*np.sin(2*x - (2*np.pi*t)/tau) + 1462650560*np.sin(2*x - (np.pi*t)/tau) - 1902611200*np.sin(2*(x + (np.pi*t)/tau)) - 1462650560*np.sin(2*x + (np.pi*t)/tau) - 1378319040*np.sin(2*x + (3*np.pi*t)/tau) - 647960320*np.sin(2*x + (4*np.pi*t)/tau) - 203134400*np.sin(2*x + (5*np.pi*t)/tau) - 41690880*np.sin(2*x + (6*np.pi*t)/tau) - 5190080*np.sin(2*x + (7*np.pi*t)/tau) - 300160*np.sin(2*x + (8*np.pi*t)/tau) + 66031405440*np.sin((np.pi*t)/tau) + 23110991904*np.sin((2*np.pi*t)/tau) + 8403997056*np.sin((3*np.pi*t)/tau) + 2626249080*np.sin((4*np.pi*t)/tau) + 646461312*np.sin((5*np.pi*t)/tau) + 115439520*np.sin((6*np.pi*t)/tau) + 13193088*np.sin((7*np.pi*t)/tau) + 721497*np.sin((8*np.pi*t)/tau)))/(3.670016e6*np.pi**9)
    c[10, :] =(715*(-6022692*t*w0 + 3011346*np.sin(2*x) + 67*np.sin(2*x - (8*np.pi*t)/tau) + 1324*np.sin(2*x - (7*np.pi*t)/tau) + 12408*np.sin(2*x - (6*np.pi*t)/tau) + 72548*np.sin(2*x - (5*np.pi*t)/tau) + 289268*np.sin(2*x - (4*np.pi*t)/tau) + 820428*np.sin(2*x - (3*np.pi*t)/tau) + 1698760*np.sin(2*x - (2*np.pi*t)/tau) + 2611876*np.sin(2*x - (np.pi*t)/tau) + 1698760*np.sin(2*(x + (np.pi*t)/tau)) + 2611876*np.sin(2*x + (np.pi*t)/tau) + 820428*np.sin(2*x + (3*np.pi*t)/tau) + 289268*np.sin(2*x + (4*np.pi*t)/tau) + 72548*np.sin(2*x + (5*np.pi*t)/tau) + 12408*np.sin(2*x + (6*np.pi*t)/tau) + 1324*np.sin(2*x + (7*np.pi*t)/tau) + 67*np.sin(2*x + (8*np.pi*t)/tau)))/(16384.*np.pi**10)
    c[11, :] =(-13*(4637472840*np.pi + 32144*np.sin(2*x - (8*np.pi*t)/tau) + 532336*np.sin(2*x - (7*np.pi*t)/tau) + 4021920*np.sin(2*x - (6*np.pi*t)/tau) + 18012400*np.sin(2*x - (5*np.pi*t)/tau) + 52582880*np.sin(2*x - (4*np.pi*t)/tau) + 103490352*np.sin(2*x - (3*np.pi*t)/tau) + 134724128*np.sin(2*x - (2*np.pi*t)/tau) + 99909040*np.sin(2*x - (np.pi*t)/tau) - 134724128*np.sin(2*(x + (np.pi*t)/tau)) - 99909040*np.sin(2*x + (np.pi*t)/tau) - 103490352*np.sin(2*x + (3*np.pi*t)/tau) - 52582880*np.sin(2*x + (4*np.pi*t)/tau) - 18012400*np.sin(2*x + (5*np.pi*t)/tau) - 4021920*np.sin(2*x + (6*np.pi*t)/tau) - 532336*np.sin(2*x + (7*np.pi*t)/tau) - 32144*np.sin(2*x + (8*np.pi*t)/tau) + 8244396160*np.sin((np.pi*t)/tau) + 2885538656*np.sin((2*np.pi*t)/tau) + 1049286784*np.sin((3*np.pi*t)/tau) + 327902120*np.sin((4*np.pi*t)/tau) + 80714368*np.sin((5*np.pi*t)/tau) + 14413280*np.sin((6*np.pi*t)/tau) + 1647232*np.sin((7*np.pi*t)/tau) + 90083*np.sin((8*np.pi*t)/tau)))/(229376.*np.pi**11)
    c[12, :] =(-91*(-2322540*t*w0 + 1161270*np.sin(2*x) + 41*np.sin(2*x - (8*np.pi*t)/tau) + 776*np.sin(2*x - (7*np.pi*t)/tau) + 6840*np.sin(2*x - (6*np.pi*t)/tau) + 36760*np.sin(2*x - (5*np.pi*t)/tau) + 134140*np.sin(2*x - (4*np.pi*t)/tau) + 352008*np.sin(2*x - (3*np.pi*t)/tau) + 687368*np.sin(2*x - (2*np.pi*t)/tau) + 1019480*np.sin(2*x - (np.pi*t)/tau) + 687368*np.sin(2*(x + (np.pi*t)/tau)) + 1019480*np.sin(2*x + (np.pi*t)/tau) + 352008*np.sin(2*x + (3*np.pi*t)/tau) + 134140*np.sin(2*x + (4*np.pi*t)/tau) + 36760*np.sin(2*x + (5*np.pi*t)/tau) + 6840*np.sin(2*x + (6*np.pi*t)/tau) + 776*np.sin(2*x + (7*np.pi*t)/tau) + 41*np.sin(2*x + (8*np.pi*t)/tau)))/(8192.*np.pi**12)
    c[13, :] =(422702280*np.pi + 2240*np.sin(2*x - (8*np.pi*t)/tau) + 34720*np.sin(2*x - (7*np.pi*t)/tau) + 241920*np.sin(2*x - (6*np.pi*t)/tau) + 1002400*np.sin(2*x - (5*np.pi*t)/tau) + 2737280*np.sin(2*x - (4*np.pi*t)/tau) + 5110560*np.sin(2*x - (3*np.pi*t)/tau) + 6406400*np.sin(2*x - (2*np.pi*t)/tau) + 4644640*np.sin(2*x - (np.pi*t)/tau) - 6406400*np.sin(2*(x + (np.pi*t)/tau)) - 4644640*np.sin(2*x + (np.pi*t)/tau) - 5110560*np.sin(2*x + (3*np.pi*t)/tau) - 2737280*np.sin(2*x + (4*np.pi*t)/tau) - 1002400*np.sin(2*x + (5*np.pi*t)/tau) - 241920*np.sin(2*x + (6*np.pi*t)/tau) - 34720*np.sin(2*x + (7*np.pi*t)/tau) - 2240*np.sin(2*x + (8*np.pi*t)/tau) + 751470720*np.sin((np.pi*t)/tau) + 263014752*np.sin((2*np.pi*t)/tau) + 95641728*np.sin((3*np.pi*t)/tau) + 29888040*np.sin((4*np.pi*t)/tau) + 7357056*np.sin((5*np.pi*t)/tau) + 1313760*np.sin((6*np.pi*t)/tau) + 150144*np.sin((7*np.pi*t)/tau) + 8211*np.sin((8*np.pi*t)/tau))/(16384.*np.pi**13)
    c[14, :] =(5*(-262548*t*w0 + 131274*np.sin(2*x) + 7*np.sin(2*x - (8*np.pi*t)/tau) + 124*np.sin(2*x - (7*np.pi*t)/tau) + 1008*np.sin(2*x - (6*np.pi*t)/tau) + 5012*np.sin(2*x - (5*np.pi*t)/tau) + 17108*np.sin(2*x - (4*np.pi*t)/tau) + 42588*np.sin(2*x - (3*np.pi*t)/tau) + 80080*np.sin(2*x - (2*np.pi*t)/tau) + 116116*np.sin(2*x - (np.pi*t)/tau) + 80080*np.sin(2*(x + (np.pi*t)/tau)) + 116116*np.sin(2*x + (np.pi*t)/tau) + 42588*np.sin(2*x + (3*np.pi*t)/tau) + 17108*np.sin(2*x + (4*np.pi*t)/tau) + 5012*np.sin(2*x + (5*np.pi*t)/tau) + 1008*np.sin(2*x + (6*np.pi*t)/tau) + 124*np.sin(2*x + (7*np.pi*t)/tau) + 7*np.sin(2*x + (8*np.pi*t)/tau)))/(1024.*np.pi**14)
    c[15, :] =-0.00006975446428571428*(18378360*np.pi + 56*np.sin(2*x - (8*np.pi*t)/tau) + 784*np.sin(2*x - (7*np.pi*t)/tau) + 5040*np.sin(2*x - (6*np.pi*t)/tau) + 19600*np.sin(2*x - (5*np.pi*t)/tau) + 50960*np.sin(2*x - (4*np.pi*t)/tau) + 91728*np.sin(2*x - (3*np.pi*t)/tau) + 112112*np.sin(2*x - (2*np.pi*t)/tau) + 80080*np.sin(2*x - (np.pi*t)/tau) - 112112*np.sin(2*(x + (np.pi*t)/tau)) - 80080*np.sin(2*x + (np.pi*t)/tau) - 91728*np.sin(2*x + (3*np.pi*t)/tau) - 50960*np.sin(2*x + (4*np.pi*t)/tau) - 19600*np.sin(2*x + (5*np.pi*t)/tau) - 5040*np.sin(2*x + (6*np.pi*t)/tau) - 784*np.sin(2*x + (7*np.pi*t)/tau) - 56*np.sin(2*x + (8*np.pi*t)/tau) + 32672640*np.sin((np.pi*t)/tau) + 11435424*np.sin((2*np.pi*t)/tau) + 4158336*np.sin((3*np.pi*t)/tau) + 1299480*np.sin((4*np.pi*t)/tau) + 319872*np.sin((5*np.pi*t)/tau) + 57120*np.sin((6*np.pi*t)/tau) + 6528*np.sin((7*np.pi*t)/tau) + 357*np.sin((8*np.pi*t)/tau))/np.pi**15
    c[16, :] =-0.0009765625*(-25740*t*w0 + 12870*np.sin(2*x) + np.sin(2*x - (8*np.pi*t)/tau) + 16*np.sin(2*x - (7*np.pi*t)/tau) + 120*np.sin(2*x - (6*np.pi*t)/tau) + 560*np.sin(2*x - (5*np.pi*t)/tau) + 1820*np.sin(2*x - (4*np.pi*t)/tau) + 4368*np.sin(2*x - (3*np.pi*t)/tau) + 8008*np.sin(2*x - (2*np.pi*t)/tau) + 11440*np.sin(2*x - (np.pi*t)/tau) + 8008*np.sin(2*(x + (np.pi*t)/tau)) + 11440*np.sin(2*x + (np.pi*t)/tau) + 4368*np.sin(2*x + (3*np.pi*t)/tau) + 1820*np.sin(2*x + (4*np.pi*t)/tau) + 560*np.sin(2*x + (5*np.pi*t)/tau) + 120*np.sin(2*x + (6*np.pi*t)/tau) + 16*np.sin(2*x + (7*np.pi*t)/tau) + np.sin(2*x + (8*np.pi*t)/tau))/np.pi**16
    c[17, :] =(360360*np.pi + 640640*np.sin((np.pi*t)/tau) + 224224*np.sin((2*np.pi*t)/tau) + 81536*np.sin((3*np.pi*t)/tau) + 25480*np.sin((4*np.pi*t)/tau) + 6272*np.sin((5*np.pi*t)/tau) + 1120*np.sin((6*np.pi*t)/tau) + 128*np.sin((7*np.pi*t)/tau) + 7*np.sin((8*np.pi*t)/tau))/(14336.*np.pi**17)
    numerator = c[17, :]
    for i in range(16, -1, -1):
        numerator = theta * numerator + c[i, :]
    return A0**2 * numerator / denominator

# @njit(parallel=True, fastmath = False,cache = True)
def cos8_int_E2(t, A0, w0, tau, cep):
    " calculate the cumulative integral of E^2(t) for the cos8 envelope "
    t = np.asarray(t)
    x = w0 * t - cep
    theta = w0 * tau
    thetaOverPi = theta / np.pi
    denominator = -((-3 + thetaOverPi)*(-2 + thetaOverPi)*(-1 + thetaOverPi)* \
        (thetaOverPi)**2*(1 + thetaOverPi)*(2 + thetaOverPi)* \
         (3 + thetaOverPi)*(-7 + 2*thetaOverPi)*(-5 + 2*thetaOverPi)* \
         (-3 + 2*thetaOverPi)*(-1 + 2*thetaOverPi)*(1 + 2*thetaOverPi)* \
         (3 + 2*thetaOverPi)*(5 + 2*thetaOverPi)*(7 + 2*thetaOverPi))
    c = np.zeros((18, t.size))
    c[0, :] =(42567525*(-2*(cep + x) + np.sin(2*(cep + theta)) + np.sin(2*x)))/2048.
    c[1, :] =(-945*(720720*np.pi - 105*np.sin(2*x - (8*np.pi*t)/tau) - 1440*np.sin(2*x - (7*np.pi*t)/tau) - 8960*np.sin(2*x - (6*np.pi*t)/tau) - 32928*np.sin(2*x - (5*np.pi*t)/tau) - 76440*np.sin(2*x - (4*np.pi*t)/tau) - 101920*np.sin(2*x - (3*np.pi*t)/tau) + 480480*np.sin(2*x - (np.pi*t)/tau) - 480480*np.sin(2*x + (np.pi*t)/tau) + 101920*np.sin(2*x + (3*np.pi*t)/tau) + 76440*np.sin(2*x + (4*np.pi*t)/tau) + 32928*np.sin(2*x + (5*np.pi*t)/tau) + 8960*np.sin(2*x + (6*np.pi*t)/tau) + 1440*np.sin(2*x + (7*np.pi*t)/tau) + 105*np.sin(2*x + (8*np.pi*t)/tau) + 960960*np.sin((np.pi*t)/tau) - 203840*np.sin((3*np.pi*t)/tau) - 152880*np.sin((4*np.pi*t)/tau) - 65856*np.sin((5*np.pi*t)/tau) - 17920*np.sin((6*np.pi*t)/tau) - 2880*np.sin((7*np.pi*t)/tau) - 210*np.sin((8*np.pi*t)/tau)))/(16384.*np.pi)
    c[2, :] =(-9*(-1546714884*t*w0 + 1057140942*np.sin(2*x) + 11025*np.sin(2*x - (8*np.pi*t)/tau) + 180000*np.sin(2*x - (7*np.pi*t)/tau) + 1391600*np.sin(2*x - (6*np.pi*t)/tau) + 6816096*np.sin(2*x - (5*np.pi*t)/tau) + 24078600*np.sin(2*x - (4*np.pi*t)/tau) + 67776800*np.sin(2*x - (3*np.pi*t)/tau) + 176576400*np.sin(2*x - (2*np.pi*t)/tau) + 655855200*np.sin(2*x - (np.pi*t)/tau) + 176576400*np.sin(2*(x + (np.pi*t)/tau)) + 655855200*np.sin(2*x + (np.pi*t)/tau) + 67776800*np.sin(2*x + (3*np.pi*t)/tau) + 24078600*np.sin(2*x + (4*np.pi*t)/tau) + 6816096*np.sin(2*x + (5*np.pi*t)/tau) + 1391600*np.sin(2*x + (6*np.pi*t)/tau) + 180000*np.sin(2*x + (7*np.pi*t)/tau) + 11025*np.sin(2*x + (8*np.pi*t)/tau)))/(65536.*np.pi**2)
    c[3, :] =(3*(324810125640*np.pi - 56003010*np.sin(2*x - (8*np.pi*t)/tau) - 768257280*np.sin(2*x - (7*np.pi*t)/tau) - 4783725520*np.sin(2*x - (6*np.pi*t)/tau) - 17616480000*np.sin(2*x - (5*np.pi*t)/tau) - 41191566780*np.sin(2*x - (4*np.pi*t)/tau) - 57106999040*np.sin(2*x - (3*np.pi*t)/tau) - 18540522000*np.sin(2*x - (2*np.pi*t)/tau) + 33783509760*np.sin(2*x - (np.pi*t)/tau) + 18540522000*np.sin(2*(x + (np.pi*t)/tau)) - 33783509760*np.sin(2*x + (np.pi*t)/tau) + 57106999040*np.sin(2*x + (3*np.pi*t)/tau) + 41191566780*np.sin(2*x + (4*np.pi*t)/tau) + 17616480000*np.sin(2*x + (5*np.pi*t)/tau) + 4783725520*np.sin(2*x + (6*np.pi*t)/tau) + 768257280*np.sin(2*x + (7*np.pi*t)/tau) + 56003010*np.sin(2*x + (8*np.pi*t)/tau) + 406593707520*np.sin((np.pi*t)/tau) - 37081044000*np.sin((2*np.pi*t)/tau) - 122204526080*np.sin((3*np.pi*t)/tau) - 85754137560*np.sin((4*np.pi*t)/tau) - 36162319872*np.sin((5*np.pi*t)/tau) - 9743067040*np.sin((6*np.pi*t)/tau) - 1557250560*np.sin((7*np.pi*t)/tau) - 113163645*np.sin((8*np.pi*t)/tau)))/(4.58752e6*np.pi**3)
    c[4, :] =(-43182496500*t*w0 + 83370568710*np.sin(2*x) + 2400129*np.sin(2*x - (8*np.pi*t)/tau) + 39175200*np.sin(2*x - (7*np.pi*t)/tau) + 302675216*np.sin(2*x - (6*np.pi*t)/tau) + 1480157280*np.sin(2*x - (5*np.pi*t)/tau) + 5205763836*np.sin(2*x - (4*np.pi*t)/tau) + 14441012768*np.sin(2*x - (3*np.pi*t)/tau) + 35262090864*np.sin(2*x - (2*np.pi*t)/tau) + 66497869152*np.sin(2*x - (np.pi*t)/tau) + 35262090864*np.sin(2*(x + (np.pi*t)/tau)) + 66497869152*np.sin(2*x + (np.pi*t)/tau) + 14441012768*np.sin(2*x + (3*np.pi*t)/tau) + 5205763836*np.sin(2*x + (4*np.pi*t)/tau) + 1480157280*np.sin(2*x + (5*np.pi*t)/tau) + 302675216*np.sin(2*x + (6*np.pi*t)/tau) + 39175200*np.sin(2*x + (7*np.pi*t)/tau) + 2400129*np.sin(2*x + (8*np.pi*t)/tau))/(262144.*np.pi**4)
    c[5, :] =(-1813664853000*np.pi + 642248880*np.sin(2*x - (8*np.pi*t)/tau) + 8817262272*np.sin(2*x - (7*np.pi*t)/tau) + 55010474064*np.sin(2*x - (6*np.pi*t)/tau) + 203695477440*np.sin(2*x - (5*np.pi*t)/tau) + 485145294816*np.sin(2*x - (4*np.pi*t)/tau) + 734212559808*np.sin(2*x - (3*np.pi*t)/tau) + 673758028944*np.sin(2*x - (2*np.pi*t)/tau) + 340462170048*np.sin(2*x - (np.pi*t)/tau) - 673758028944*np.sin(2*(x + (np.pi*t)/tau)) - 340462170048*np.sin(2*x + (np.pi*t)/tau) - 734212559808*np.sin(2*x + (3*np.pi*t)/tau) - 485145294816*np.sin(2*x + (4*np.pi*t)/tau) - 203695477440*np.sin(2*x + (5*np.pi*t)/tau) - 55010474064*np.sin(2*x + (6*np.pi*t)/tau) - 8817262272*np.sin(2*x + (7*np.pi*t)/tau) - 642248880*np.sin(2*x + (8*np.pi*t)/tau) - 1265005822080*np.sin((np.pi*t)/tau) + 1614499574688*np.sin((2*np.pi*t)/tau) + 1833909913472*np.sin((3*np.pi*t)/tau) + 1118580230040*np.sin((4*np.pi*t)/tau) + 447979324800*np.sin((5*np.pi*t)/tau) + 117674908960*np.sin((6*np.pi*t)/tau) + 18537618816*np.sin((7*np.pi*t)/tau) + 1334900469*np.sin((8*np.pi*t)/tau))/(1.1010048e7*np.pi**5)
    c[6, :] =(-91*(167225916*t*w0 + 457055742*np.sin(2*x) + 21005*np.sin(2*x - (8*np.pi*t)/tau) + 342592*np.sin(2*x - (7*np.pi*t)/tau) + 2642348*np.sin(2*x - (6*np.pi*t)/tau) + 12866496*np.sin(2*x - (5*np.pi*t)/tau) + 44724468*np.sin(2*x - (4*np.pi*t)/tau) + 119465792*np.sin(2*x - (3*np.pi*t)/tau) + 248274004*np.sin(2*x - (2*np.pi*t)/tau) + 391514816*np.sin(2*x - (np.pi*t)/tau) + 248274004*np.sin(2*(x + (np.pi*t)/tau)) + 391514816*np.sin(2*x + (np.pi*t)/tau) + 119465792*np.sin(2*x + (3*np.pi*t)/tau) + 44724468*np.sin(2*x + (4*np.pi*t)/tau) + 12866496*np.sin(2*x + (5*np.pi*t)/tau) + 2642348*np.sin(2*x + (6*np.pi*t)/tau) + 342592*np.sin(2*x + (7*np.pi*t)/tau) + 21005*np.sin(2*x + (8*np.pi*t)/tau)))/(131072.*np.pi**6)
    c[7, :] =(-13*(35117442360*np.pi + 11425260*np.sin(2*x - (8*np.pi*t)/tau) + 157080000*np.sin(2*x - (7*np.pi*t)/tau) + 983477880*np.sin(2*x - (6*np.pi*t)/tau) + 3675672000*np.sin(2*x - (5*np.pi*t)/tau) + 8999602920*np.sin(2*x - (4*np.pi*t)/tau) + 14970352320*np.sin(2*x - (3*np.pi*t)/tau) + 16813852440*np.sin(2*x - (2*np.pi*t)/tau) + 11229408960*np.sin(2*x - (np.pi*t)/tau) - 16813852440*np.sin(2*(x + (np.pi*t)/tau)) - 11229408960*np.sin(2*x + (np.pi*t)/tau) - 14970352320*np.sin(2*x + (3*np.pi*t)/tau) - 8999602920*np.sin(2*x + (4*np.pi*t)/tau) - 3675672000*np.sin(2*x + (5*np.pi*t)/tau) - 983477880*np.sin(2*x + (6*np.pi*t)/tau) - 157080000*np.sin(2*x + (7*np.pi*t)/tau) - 11425260*np.sin(2*x + (8*np.pi*t)/tau) + 97285668480*np.sin((np.pi*t)/tau) + 70647376800*np.sin((2*np.pi*t)/tau) + 47870193280*np.sin((3*np.pi*t)/tau) + 24663289560*np.sin((4*np.pi*t)/tau) + 9142073472*np.sin((5*np.pi*t)/tau) + 2302795040*np.sin((6*np.pi*t)/tau) + 353708160*np.sin((7*np.pi*t)/tau) + 25056045*np.sin((8*np.pi*t)/tau)))/(3.93216e6*np.pi**7)
    c[8, :] =(143*(298420980*t*w0 + 296374650*np.sin(2*x) + 17311*np.sin(2*x - (8*np.pi*t)/tau) + 281952*np.sin(2*x - (7*np.pi*t)/tau) + 2167816*np.sin(2*x - (6*np.pi*t)/tau) + 10477600*np.sin(2*x - (5*np.pi*t)/tau) + 35740516*np.sin(2*x - (4*np.pi*t)/tau) + 90742624*np.sin(2*x - (3*np.pi*t)/tau) + 175350840*np.sin(2*x - (2*np.pi*t)/tau) + 259961632*np.sin(2*x - (np.pi*t)/tau) + 175350840*np.sin(2*(x + (np.pi*t)/tau)) + 259961632*np.sin(2*x + (np.pi*t)/tau) + 90742624*np.sin(2*x + (3*np.pi*t)/tau) + 35740516*np.sin(2*x + (4*np.pi*t)/tau) + 10477600*np.sin(2*x + (5*np.pi*t)/tau) + 2167816*np.sin(2*x + (6*np.pi*t)/tau) + 281952*np.sin(2*x + (7*np.pi*t)/tau) + 17311*np.sin(2*x + (8*np.pi*t)/tau)))/(262144.*np.pi**8)
    c[9, :] =(143*(12533681160*np.pi + 900480*np.sin(2*x - (8*np.pi*t)/tau) + 12409152*np.sin(2*x - (7*np.pi*t)/tau) + 78107904*np.sin(2*x - (6*np.pi*t)/tau) + 295552320*np.sin(2*x - (5*np.pi*t)/tau) + 744473856*np.sin(2*x - (4*np.pi*t)/tau) + 1297654848*np.sin(2*x - (3*np.pi*t)/tau) + 1542422784*np.sin(2*x - (2*np.pi*t)/tau) + 1081002048*np.sin(2*x - (np.pi*t)/tau) - 1542422784*np.sin(2*(x + (np.pi*t)/tau)) - 1081002048*np.sin(2*x + (np.pi*t)/tau) - 1297654848*np.sin(2*x + (3*np.pi*t)/tau) - 744473856*np.sin(2*x + (4*np.pi*t)/tau) - 295552320*np.sin(2*x + (5*np.pi*t)/tau) - 78107904*np.sin(2*x + (6*np.pi*t)/tau) - 12409152*np.sin(2*x + (7*np.pi*t)/tau) - 900480*np.sin(2*x + (8*np.pi*t)/tau) + 25029164160*np.sin((np.pi*t)/tau) + 11644624992*np.sin((2*np.pi*t)/tau) + 5982540928*np.sin((3*np.pi*t)/tau) + 2634351720*np.sin((4*np.pi*t)/tau) + 890504832*np.sin((5*np.pi*t)/tau) + 211846880*np.sin((6*np.pi*t)/tau) + 31346304*np.sin((7*np.pi*t)/tau) + 2164491*np.sin((8*np.pi*t)/tau)))/(1.1010048e7*np.pi**9)
    c[10, :] =(-143*(7190196*t*w0 + 5027802*np.sin(2*x) + 335*np.sin(2*x - (8*np.pi*t)/tau) + 5444*np.sin(2*x - (7*np.pi*t)/tau) + 41656*np.sin(2*x - (6*np.pi*t)/tau) + 199276*np.sin(2*x - (5*np.pi*t)/tau) + 665476*np.sin(2*x - (4*np.pi*t)/tau) + 1639204*np.sin(2*x - (3*np.pi*t)/tau) + 3070088*np.sin(2*x - (2*np.pi*t)/tau) + 4447532*np.sin(2*x - (np.pi*t)/tau) + 3070088*np.sin(2*(x + (np.pi*t)/tau)) + 4447532*np.sin(2*x + (np.pi*t)/tau) + 1639204*np.sin(2*x + (3*np.pi*t)/tau) + 665476*np.sin(2*x + (4*np.pi*t)/tau) + 199276*np.sin(2*x + (5*np.pi*t)/tau) + 41656*np.sin(2*x + (6*np.pi*t)/tau) + 5444*np.sin(2*x + (7*np.pi*t)/tau) + 335*np.sin(2*x + (8*np.pi*t)/tau)))/(16384.*np.pi**10)
    c[11, :] =(-13*(16609352760*np.pi + 482160*np.sin(2*x - (8*np.pi*t)/tau) + 6667920*np.sin(2*x - (7*np.pi*t)/tau) + 42265440*np.sin(2*x - (6*np.pi*t)/tau) + 161994000*np.sin(2*x - (5*np.pi*t)/tau) + 415433760*np.sin(2*x - (4*np.pi*t)/tau) + 738939600*np.sin(2*x - (3*np.pi*t)/tau) + 894912480*np.sin(2*x - (2*np.pi*t)/tau) + 635545680*np.sin(2*x - (np.pi*t)/tau) - 894912480*np.sin(2*(x + (np.pi*t)/tau)) - 635545680*np.sin(2*x + (np.pi*t)/tau) - 738939600*np.sin(2*x + (3*np.pi*t)/tau) - 415433760*np.sin(2*x + (4*np.pi*t)/tau) - 161994000*np.sin(2*x + (5*np.pi*t)/tau) - 42265440*np.sin(2*x + (6*np.pi*t)/tau) - 6667920*np.sin(2*x + (7*np.pi*t)/tau) - 482160*np.sin(2*x + (8*np.pi*t)/tau) + 30998647680*np.sin((np.pi*t)/tau) + 12393981600*np.sin((2*np.pi*t)/tau) + 5442935680*np.sin((3*np.pi*t)/tau) + 2110431960*np.sin((4*np.pi*t)/tau) + 649095552*np.sin((5*np.pi*t)/tau) + 144196640*np.sin((6*np.pi*t)/tau) + 20300160*np.sin((7*np.pi*t)/tau) + 1351245*np.sin((8*np.pi*t)/tau)))/(3.44064e6*np.pi**11)
    c[12, :] =(91*(970860*t*w0 + 569910*np.sin(2*x) + 41*np.sin(2*x - (8*np.pi*t)/tau) + 664*np.sin(2*x - (7*np.pi*t)/tau) + 5048*np.sin(2*x - (6*np.pi*t)/tau) + 23880*np.sin(2*x - (5*np.pi*t)/tau) + 78588*np.sin(2*x - (4*np.pi*t)/tau) + 190616*np.sin(2*x - (3*np.pi*t)/tau) + 352264*np.sin(2*x - (2*np.pi*t)/tau) + 505736*np.sin(2*x - (np.pi*t)/tau) + 352264*np.sin(2*(x + (np.pi*t)/tau)) + 505736*np.sin(2*x + (np.pi*t)/tau) + 190616*np.sin(2*x + (3*np.pi*t)/tau) + 78588*np.sin(2*x + (4*np.pi*t)/tau) + 23880*np.sin(2*x + (5*np.pi*t)/tau) + 5048*np.sin(2*x + (6*np.pi*t)/tau) + 664*np.sin(2*x + (7*np.pi*t)/tau) + 41*np.sin(2*x + (8*np.pi*t)/tau)))/(8192.*np.pi**12)
    c[13, :] =(530089560*np.pi + 6720*np.sin(2*x - (8*np.pi*t)/tau) + 93408*np.sin(2*x - (7*np.pi*t)/tau) + 596736*np.sin(2*x - (6*np.pi*t)/tau) + 2308320*np.sin(2*x - (5*np.pi*t)/tau) + 5975424*np.sin(2*x - (4*np.pi*t)/tau) + 10719072*np.sin(2*x - (3*np.pi*t)/tau) + 13069056*np.sin(2*x - (2*np.pi*t)/tau) + 9321312*np.sin(2*x - (np.pi*t)/tau) - 13069056*np.sin(2*(x + (np.pi*t)/tau)) - 9321312*np.sin(2*x + (np.pi*t)/tau) - 10719072*np.sin(2*x + (3*np.pi*t)/tau) - 5975424*np.sin(2*x + (4*np.pi*t)/tau) - 2308320*np.sin(2*x + (5*np.pi*t)/tau) - 596736*np.sin(2*x + (6*np.pi*t)/tau) - 93408*np.sin(2*x + (7*np.pi*t)/tau) - 6720*np.sin(2*x + (8*np.pi*t)/tau) + 962881920*np.sin((np.pi*t)/tau) + 358534176*np.sin((2*np.pi*t)/tau) + 143421824*np.sin((3*np.pi*t)/tau) + 50526840*np.sin((4*np.pi*t)/tau) + 14243712*np.sin((5*np.pi*t)/tau) + 2937760*np.sin((6*np.pi*t)/tau) + 388992*np.sin((7*np.pi*t)/tau) + 24633*np.sin((8*np.pi*t)/tau))/(49152.*np.pi**13)
    c[14, :] =-0.0009765625*(873444*t*w0 + 464178*np.sin(2*x) + 35*np.sin(2*x - (8*np.pi*t)/tau) + 564*np.sin(2*x - (7*np.pi*t)/tau) + 4256*np.sin(2*x - (6*np.pi*t)/tau) + 19964*np.sin(2*x - (5*np.pi*t)/tau) + 65156*np.sin(2*x - (4*np.pi*t)/tau) + 156884*np.sin(2*x - (3*np.pi*t)/tau) + 288288*np.sin(2*x - (2*np.pi*t)/tau) + 412412*np.sin(2*x - (np.pi*t)/tau) + 288288*np.sin(2*(x + (np.pi*t)/tau)) + 412412*np.sin(2*x + (np.pi*t)/tau) + 156884*np.sin(2*x + (3*np.pi*t)/tau) + 65156*np.sin(2*x + (4*np.pi*t)/tau) + 19964*np.sin(2*x + (5*np.pi*t)/tau) + 4256*np.sin(2*x + (6*np.pi*t)/tau) + 564*np.sin(2*x + (7*np.pi*t)/tau) + 35*np.sin(2*x + (8*np.pi*t)/tau))/np.pi**14
    c[15, :] =-4.6502976190476195e-6*(183423240*np.pi + 840*np.sin(2*x - (8*np.pi*t)/tau) + 11760*np.sin(2*x - (7*np.pi*t)/tau) + 75600*np.sin(2*x - (6*np.pi*t)/tau) + 294000*np.sin(2*x - (5*np.pi*t)/tau) + 764400*np.sin(2*x - (4*np.pi*t)/tau) + 1375920*np.sin(2*x - (3*np.pi*t)/tau) + 1681680*np.sin(2*x - (2*np.pi*t)/tau) + 1201200*np.sin(2*x - (np.pi*t)/tau) - 1681680*np.sin(2*(x + (np.pi*t)/tau)) - 1201200*np.sin(2*x + (np.pi*t)/tau) - 1375920*np.sin(2*x + (3*np.pi*t)/tau) - 764400*np.sin(2*x + (4*np.pi*t)/tau) - 294000*np.sin(2*x + (5*np.pi*t)/tau) - 75600*np.sin(2*x + (6*np.pi*t)/tau) - 11760*np.sin(2*x + (7*np.pi*t)/tau) - 840*np.sin(2*x + (8*np.pi*t)/tau) + 328648320*np.sin((np.pi*t)/tau) + 117717600*np.sin((2*np.pi*t)/tau) + 44437120*np.sin((3*np.pi*t)/tau) + 14600040*np.sin((4*np.pi*t)/tau) + 3819648*np.sin((5*np.pi*t)/tau) + 731360*np.sin((6*np.pi*t)/tau) + 90240*np.sin((7*np.pi*t)/tau) + 5355*np.sin((8*np.pi*t)/tau))/np.pi**15
    c[16, :] =(25740*t*w0 + 12870*np.sin(2*x) + np.sin(2*x - (8*np.pi*t)/tau) + 16*np.sin(2*x - (7*np.pi*t)/tau) + 120*np.sin(2*x - (6*np.pi*t)/tau) + 560*np.sin(2*x - (5*np.pi*t)/tau) + 1820*np.sin(2*x - (4*np.pi*t)/tau) + 4368*np.sin(2*x - (3*np.pi*t)/tau) + 8008*np.sin(2*x - (2*np.pi*t)/tau) + 11440*np.sin(2*x - (np.pi*t)/tau) + 8008*np.sin(2*(x + (np.pi*t)/tau)) + 11440*np.sin(2*x + (np.pi*t)/tau) + 4368*np.sin(2*x + (3*np.pi*t)/tau) + 1820*np.sin(2*x + (4*np.pi*t)/tau) + 560*np.sin(2*x + (5*np.pi*t)/tau) + 120*np.sin(2*x + (6*np.pi*t)/tau) + 16*np.sin(2*x + (7*np.pi*t)/tau) + np.sin(2*x + (8*np.pi*t)/tau))/(1024.*np.pi**16)
    c[17, :] =(360360*np.pi + 640640*np.sin((np.pi*t)/tau) + 224224*np.sin((2*np.pi*t)/tau) + 81536*np.sin((3*np.pi*t)/tau) + 25480*np.sin((4*np.pi*t)/tau) + 6272*np.sin((5*np.pi*t)/tau) + 1120*np.sin((6*np.pi*t)/tau) + 128*np.sin((7*np.pi*t)/tau) + 7*np.sin((8*np.pi*t)/tau))/(14336.*np.pi**17)
    numerator = c[17, :]
    for i in range(16, -1, -1):
        numerator = theta * numerator + c[i, :]
    return -w0 * A0**2 * numerator / denominator


@njit(parallel=True, fastmath = False,cache = True)
def find_zero_crossings(X, Y):
    """ Find all the zero crossings: y(x) = 0

Parameters
----------
X : a 1D float array of x-values sorted in ascending order;
    the array may not have identical elements;
Y : a float array of the same shape as X;

Returns
-------
out : an array of x-values where the linearly interpolated function y(x)
has zero values (an empty list if there are no zero crossings).
"""
    Z = Y[:-1] * Y[1:]
    out = []
    for i in np.nonzero(Z <= 0)[0]:
        if Z[i] == 0:
            if Y[i] == 0:
                out.append(X[i])
        else:
            # there is a zero crossing between X[i] and X[i+1]
            out.append((X[i]*Y[i+1] - X[i+1]*Y[i]) / (Y[i+1] - Y[i]))
    return np.array(out)

@njit(parallel=True, fastmath = False,cache = True)
def find_extrema_positions(X, Y):
    """ Find all the extrema of the given function

    Parameters
    ----------
    X : a 1D float array of x-values sorted in ascending order;
        the array may not have identical elements;
    Y : a float array of the same shape as X;

    Returns
    -------
    out : an array of x-values where the linearly interpolated y'(x)
    has zero values (an empty list if there are no such x-values).
    """
    dY_dX = (Y[1:] - Y[:-1]) / (X[1:] - X[:-1])
    return find_zero_crossings(0.5 * (X[1:] + X[:-1]), dY_dX)

@njit(parallel=True, fastmath = False,cache = True)
def integrate_oscillating_function_jit(X, f, phi, phase_step_threshold=1e-3):
    r""" The function evaluates \int dx f(x) exp[i phi(x)] using an algorithm
    suitable for integrating quickly oscillating functions.

    Parameters
    ----------
    X: a vector of sorted x-values;
    f: either a vector or a 2D matrix where each column contains
        the values of a function f(x);
    phi: either a vector or a 2D matrix where each column contains
        the values of a complex-valued phase phi(x);
    phase_step_threshold (float): a small positive number; the formula that
        approximates the integration of an oscillating function over a
        small interval contains phi(x+dx)-phi(x) in the denominator;
        this parameter prevents divisions by very small numbers.

    Returns
    -------
    result: a row vector where elements correspond to the columns in
      'f' and 'phi'.
    """
    
    # check that the input data is OK
    assert(X.shape[0] == f.shape[0])
    assert(X.shape[0] == phi.shape[0])
    phi=-1j*phi
    # evaluate the integral(s)
    dx = X[1:] - X[:-1]
    result=np.zeros((f.shape[1]),dtype=np.complex128)
    for i in range(f.shape[1]):
        f1 = f[:-1, i]
        f2 = f[1:, i]
        phi1=phi[1:, i]
        phi2=phi[:-1, i]
        df = f2 - f1
        dphi = phi1-phi2
        Z=np.where(np.abs(dphi).real < phase_step_threshold,  (0.5 * (f1+f2) + 0.125*1j * dphi * df) *np.exp(0.5*1j*(phi1 + phi2)), 1 / dphi**2 * (np.exp(1j * phi1) * (df - 1j*f2*dphi)-(df - 1j*f1*dphi) * np.exp(1j * phi2)))
        result[i]=np.sum(Z*dx)
    return result


def integrate_oscillating_function_numexpr(X, f, phi, phase_step_threshold=1e-3):
    r""" The function evaluates \int dx f(x) exp[i phi(x)] using an algorithm
    suitable for integrating quickly oscillating functions.

    Parameters
    ----------
    X: a vector of sorted x-values;
    f: either a vector or a matrix where each column contains
        the values of a function f(x);
    phi: either a vector or a matrix where each column contains
        the values of a real-valued phase phi(x);
    phase_step_threshold (float): a small positive number; the formula that
        approximates the integration of an oscillating function over a
        small interval contains phi(x+dx)-phi(x) in the denominator;
        this parameter prevents divisions by very small numbers.

    Returns
    -------
    result: a row vector where elements correspond to the columns in
      'f' and 'phi'.
    """

    # check that the input data is OK
    assert(X.shape[0] == f.shape[0])
    assert(X.shape[0] == phi.shape[0])
    assert(np.all(np.imag(phi)) == 0)
    # evaluate the integral(s)
    dx = X[1:] - X[:-1]
    f1 = f[:-1, ...]
    f2 = f[1:, ...]
    phi1=phi[1:, ...]
    phi2=phi[:-1, ...]
    df = ne.evaluate("f2 - f1")
    f_sum=ne.evaluate("f2+f1")
    dphi = ne.evaluate("phi1-phi2")
    #phi_sum= ne.evaluate("0.5*1j*(phi1 + phi2)")
    s = np.ones((f.ndim), dtype=int)
    s[0] = dx.size
    Z = dx.reshape(s)
    Z=ne.evaluate("Z*exp(0.5*1j*(phi1 + phi2))")
    del phi1, phi2
    #absdphi=abs(dphi)
    Z=ne.evaluate("where(abs(dphi).real < phase_step_threshold, Z * (0.5 * f_sum + 0.125*1j * dphi * df), Z / dphi**2 * (exp(0.5*1j * dphi) * (df - 1j*f2*dphi)-(df - 1j*f1*dphi) / exp(0.5*1j * dphi)))")
    return ne.evaluate("sum(Z, axis=0)")


### the following are a set of function for an experimental high order oscillating function integral scheme ###
## as of right now, it is not used in the code and remains unoptimized ##
@njit(fastmath = False)
def calculate_c(dx, f, phi):
    a0, a1, a2, a3 = (f[0], f[1], f[2], f[3])
    b0, b1=(phi[0], phi[1])
    c0 = 1 / (2 * dx**5) * (-2 * (6 * a3 + 3 * a2 * dx + (a1 + a3 * b1) * dx**2) + (12 * a3 + 6 * a2 * dx + 2 * (a1 - 5 * a3 * b1) * dx**2 - 6 * (2 * a3 * b0 + a2 * b1) * dx**3 - 2 * (3 * a2 * b0 + b1 * (a1 - 2 * a3 * b1)) * dx**4 + 2 * b1 * (a0 + 6 * a3 * b0 + 2 * a2 * b1) * dx**5 + (6 * a0 * b0 + 9 * a3 * b0**2 + 4 * b1 * (3 * a2 * b0 + a1 * b1)) * dx**6 + (9 * a2 * b0**2 + 4 * b1 * (3 * a1 * b0 + a0 * b1)) * dx**7 + 3 * b0 * (3 * a1 * b0 + 4 * a0 * b1) * dx**8 + 9 * a0 * b0**2 * dx**9) * np.exp(dx**2 * (b1 + b0 * dx)))
    c1 = 1 / dx**4 * (dx * (8 * a2 + 3 * a1 * dx) + 3 * a3 * (5 + b1 * dx**2) - (15 * a3 + 8 * a2 * dx + 3 * (a1 - 4 * a3 * b1) * dx**2 - (15 * a3 * b0 + 8 * a2 * b1) * dx**3 + (-9 * a2 * b0 + 4 * b1 * (-a1 + a3 * b1)) * dx**4 + (-3 * a1 * b0 + 4 * b1 * (3 * a3 * b0 + a2 * b1)) * dx**5 + (3 * a0 * b0 + 9 * a3 * b0**2 + 4 * b1 * (3 * a2 * b0 + a1 * b1)) * dx**6 + (9 * a2 * b0**2 + 4 * b1 * (3 * a1 * b0 + a0 * b1)) * dx**7 + 3 * b0 * (3 * a1 * b0 + 4 * a0 * b1) * dx**8 + 9 * a0 * b0**2 * dx**9) * np.exp(dx**2 * (b1 + b0 * dx)))
    c2 = 1 / (2 * dx**3) * (-20 * a3 - 12 * a2 * dx - 6 * (a1 + a3 * b1) * dx**2 + (6 * dx * (2 * a2 + a1 * dx) + a3 * (20 + 4 * b1**2 * dx**4 + 9 * b0 * dx**3 * (-2 + b0 * dx**3) + 2 * b1 * dx**2 * (-7 + 6 * b0 * dx**3)) + dx**3 * (-6 * a1 * dx * (b1 + b0 * dx) + dx**2 * (a2 + a1 * dx) * (2 * b1 + 3 * b0 * dx)**2 - 2 * a2 * (5 * b1 + 6 * b0 * dx) + a0 * (2 + 4 * b1**2 * dx**4 + 9 * b0**2 * dx**6 + 2 * b1 * dx**2 * (-1 + 6 * b0 * dx**3)))) * np.exp(dx**2 * (b1 + b0 * dx)))
    c3 = a1 + a3 * b1
    c4 = a2
    c5 = a3

    return {'c0': c0, 'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5, 'b2': phi[2], 'b3': phi[3]}

#@jit(fastmath = False)
def calculate_expression(dx, b2, c0, c1, c2, c3, c4, c5):
    term1 = 120 * c0 + b2 * (-24 * c1 + b2 * (6 * c2 + b2 * (-2 * c3 + b2 * (c4 - b2 * c5))))
    term3 = 120 * c0 + b2 * (-24 * c1 + b2 * (6 * c2 - 2 * b2 * c3 + b2**2 * c4))
    term4 = -60 * c0 + b2 * (12 * c1 + b2 * (-3 * c2 + b2 * c3))
    term5 = 20 * c0 + b2 * (-4 * c1 + b2 * c2)
    term6 = -5 * c0 + b2 * c1

    d=b2*dx

    return (-term1*np.expm1(d) + np.exp(d) * d * np.polynomial.polynomial.polyval(d, [c0,  term6, term5, term4, term3][::-1], tensor=False))/ b2**6

#@jit(fastmath = False)
def calculate_expression_appx(dx, b2, c0, c1, c2, c3, c4, c5):
    term1 = c5 #* dx
    term2 = 1/2 * (c4 + b2 * c5) #* dx**2
    term3 = 1/6 * (2 * c3 + b2 * (2 * c4 + b2 * c5)) #* dx**3
    term4 = 1/8 * (2 * c2 + b2 * (2 * c3 + b2 * c4)) #* dx**4
    term5 = 1/10 * (2 * c1 + b2 * (2 * c2 + b2 * c3)) #* dx**5
    term6 = 1/12 * (2 * c0 + b2 * (2 * c1 + b2 * c2)) #* dx**6
    term7 = 1/14 * b2 * (2 * c0 + b2 * c1) #* dx**7
    term8 = 1/16 * b2**2 * c0 #* dx**8
    result=np.polynomial.polynomial.polyval(dx, [term8, term7, term6, term5, term4, term3, term2, term1][::-1], tensor=False)
    return result*dx

#@jit(fastmath = False)
def calculate_expression_main(dx, b2, b3, c0, c1, c2, c3, c4, c5):
    s=(np.abs(b2)<=1e-10)
    # print(s.shape)
    result=np.zeros(b2.shape)
    if np.any(s):
        result=calculate_expression_appx(dx, b2, c0, c1, c2, c3, c4, c5)
    s=np.logical_not(s)
    if np.any(s):
        dx, b2, c0, c1, c2, c3, c4, c5=(np.tile(dx, b2.shape[1])[s], b2[s], c0[s], c1[s], c2[s], c3[s], c4[s], c5[s])
        # print(dx.shape, s)
        result[s]=calculate_expression(dx, b2, c0, c1, c2, c3, c4, c5)
        
    return result*np.exp(b3)

def integrate_oscillating_function_splined(X, f, phi, phase_step_threshold=1e-3):
    r""" The function evaluates \int dx f(x) exp[i phi(x)] using an algorithm
    suitable for integrating quickly oscillating functions.

    Parameters
    ----------
    X: a vector of sorted x-values;
    f: either a vector or a matrix where each column contains
        the values of a function f(x);
    phi: either a vector or a matrix where each column contains
        the values of a real-valued phase phi(x);
    phase_step_threshold (float): a small positive number; the formula that
        approximates the integration of an oscillating function over a
        small interval contains phi(x+dx)-phi(x) in the denominator;
        this parameter prevents divisions by very small numbers.

    Returns
    -------
    result: a row vector where elements correspond to the columns in
      'f' and 'phi'.
    """
    # check that the input data is OK
    assert(X.shape[0] == f.shape[0])
    assert(X.shape[0] == phi.shape[0])
    # assert(np.all(np.imag(phi)) == 0)
    # evaluate the integral(s)
    dx= X[1:] - X[:-1]
    #dx = np.append(X[1:] - X[:-1],0.)
    dx=dx[:,None]  #np.tile(dx, (f.shape[1],1)).T
    fR=CubicSpline(X, f.real, bc_type='natural').c
    fI=CubicSpline(X, f.imag, bc_type='natural').c
    f=np.vectorize(complex)(fR,fI)
    phiR=CubicSpline(X, phi.real, bc_type='natural').c
    phiI=CubicSpline(X, phi.imag, bc_type='natural').c
    phi=np.vectorize(complex)(phiR,phiI)
    # print(np.polyval(f, dx)*np.exp(np.polyval(phi, dx)))
    #print(np.sum(np.polyval(f, dx)*np.exp(np.polyval(phi, dx)), axis=0))
    #print(f.shape, phi.shape, dx.shape)
    cDict=calculate_c(dx, f, phi)
    expr=calculate_expression_main(dx, **cDict)
    plt.plot(X[1:], np.cumsum(expr[:,0], axis=0))
    plt.show()
    plt.close()
    Z=np.sum(expr, axis=0)
    return Z
    Z=np.sum(np.where(abs(cDict['b2'])<=1e-3, calculate_expression_appx(dx, **cDict), calculate_expression(dx, **cDict)), axis=0)
    return Z

#####################


class LaserField:

    def __init__(self, cache_results=True):
        self.reset(cache_results)
    
    def get_number_of_pulses(self):
        return self.__number_of_pulses
    
    def get_central_wavelength(self):
        au = AtomicUnits()
        lam= [(2*np.pi*au.speed_of_light/i*au.nm) for i in self.__omega0_list]
        return lam

    def add_pulse(self, central_wavelength, peak_intensity, CEP, FWHM, envelope_N=8, t0=0.0):
        au = AtomicUnits()
        self.__number_of_pulses += 1
        w0 = 2*np.pi*au.speed_of_light / (central_wavelength / au.nm)
        self.__omega0_list.append(w0)
        self.__A0_list.append(np.sqrt(peak_intensity/1e15 * 0.02849451308 / w0**2))
        self.__CEP_list.append(CEP)
        self.__FWHM_list.append(FWHM)
        self.__t0_list.append(t0)
        self.__envelope_N_list.append(envelope_N)
        self.__tau_list.append(np.pi*FWHM/(4*np.arccos(2**(-1/(2*envelope_N)))))
    
    def get_ponderomotive_energy(self):
        if self.__number_of_pulses <= 0:
            return 0.0
        Up=0.0
        Up = max(self.Vector_potential(self.get_tgrid()))**2/4
        return Up
    def get_initial_KE_energy(self, Ip):
        if self.__number_of_pulses <= 0:
            return 0.0
        initKE=[]
        for i in range(self.__number_of_pulses):
            nPhoton=int(Ip / (self.__omega0_list[i] ))+1
            initKE.append(nPhoton* self.__omega0_list[i] - Ip)
        return np.mean(initKE)
    def get_total_energy(self, Ip):
        return self.get_initial_KE_energy(Ip) + self.get_ponderomotive_energy()
    def get_time_interval(self):
        if self.__number_of_pulses <= 0:
            return((None, None))
        t_min = self.__t0_list[0] - self.__tau_list[0]
        t_max = self.__t0_list[0] + self.__tau_list[0]
        for i in range(1, self.__number_of_pulses):
            t_min = min(t_min, self.__t0_list[i] - self.__tau_list[i])
            t_max = max(t_max, self.__t0_list[i] + self.__tau_list[i])
        return (t_min, t_max)
    
    def get_tgrid(self, npoint=1000):
        t_min, t_max = self.get_time_interval()
        if t_min is None or t_max is None:
            return None
        return np.linspace(t_min, t_max, npoint)

    def reset(self, cache_results=True):
        self.__number_of_pulses = 0
        self.__omega0_list = []
        self.__A0_list = []
        self.__CEP_list = []
        self.__FWHM_list = []
        self.__t0_list = []
        self.__envelope_N_list = []
        self.__tau_list = []
        self.__cache_results = cache_results
        self.__cached_t_for_A = None
        self.__cached_A = None
        self.__cached_t_for_E = None
        self.__cached_E = None
        self.__cached_t_for_int_A = None
        self.__cached_int_A = None
        self.__cached_t_for_int_A2 = None
        self.__cached_int_A2 = None
        self.__cached_t_for_int_E2 = None
        self.__cached_int_E2 = None


    def Vector_potential(self, t):
        t = np.asarray(t)
        if self.__cache_results == True:
            if self.__cached_t_for_A is None or not(np.array_equal(t, self.__cached_t_for_A)):
                self.__cached_t_for_A = t.copy()
            elif not(self.__cached_A is None):
                return self.__cached_A
        # print("LaserField.Vector_potential: calculating from scratch:", ) # DEBUGGING
        Field = np.zeros(t.size)
        if self.__number_of_pulses <= 0:
            return Field
        for i in range(self.__number_of_pulses):
            w0 = self.__omega0_list[i]
            A0 = self.__A0_list[i]
            if A0 == 0:
                continue
            tau = self.__tau_list[i]
            N = self.__envelope_N_list[i]
            tt = t - self.__t0_list[i]
            cep = self.__CEP_list[i]
            Field += np.where(np.abs(tt) < tau, cosN_vector_potential(tt, A0, w0, tau, cep, N), 0)
        if self.__cache_results == True:
            self.__cached_A = Field.copy()
        return Field

    def Electric_Field(self, t):
        t = np.asarray(t)
        if self.__cache_results == True:
            if self.__cached_t_for_E is None or not(np.array_equal(t, self.__cached_t_for_E)):
                self.__cached_t_for_E = t.copy()
            elif not(self.__cached_E is None):
                return self.__cached_E
        Field = np.zeros(t.size)
        if self.__number_of_pulses <= 0:
            return Field
        for i in range(self.__number_of_pulses):
            w0 = self.__omega0_list[i]
            A0 = self.__A0_list[i]
            if A0 == 0:
                continue
            tau = self.__tau_list[i]
            N = self.__envelope_N_list[i]
            tt = t - self.__t0_list[i]
            cep = self.__CEP_list[i]
            Field += np.where(np.abs(tt) < tau, cosN_electric_field(tt, A0, w0, tau, cep, N), 0)
        if self.__cache_results == True:
            self.__cached_E = Field.copy()
        return Field

    def int_A(self, t):
        " calculate the cumulative integral of A(t) "
        t = np.asarray(t)
        if self.__cache_results == True:
            if self.__cached_t_for_int_A is None or not(np.array_equal(t, self.__cached_t_for_int_A)):
                self.__cached_t_for_int_A = t.copy()
            elif not(self.__cached_int_A is None):
                return self.__cached_int_A
        result = np.zeros(t.size)
        # print("LaserField.int_A: calculating from scratch") # DEBUGGING
        if self.__number_of_pulses <= 0:
            return result
        for i in range(self.__number_of_pulses):
            w0 = self.__omega0_list[i]
            A0 = self.__A0_list[i]
            if A0 == 0:
                continue
            tau = self.__tau_list[i]
            assert(self.__envelope_N_list[i] == 8)
            tt = t - self.__t0_list[i]
            cep = self.__CEP_list[i]
            result += np.where(np.abs(tt) < tau, cos8_int_A(tt, A0, w0, tau, cep), 0)
            result[tt >= tau] += cos8_int_A(tau, A0, w0, tau, cep)
        if self.__cache_results == True:
            self.__cached_int_A = result.copy()
        return result
    
    def int_A2(self, t):
        " calculate the cumulative integral of A^2(t) "
        t = np.asarray(t)
        if self.__cache_results == True:
            if self.__cached_t_for_int_A2 is None or not(np.array_equal(t, self.__cached_t_for_int_A2)):
                self.__cached_t_for_int_A2 = t.copy()
            elif not(self.__cached_int_A2 is None):
                return self.__cached_int_A2
        result = np.zeros(t.size)
        # print("LaserField.int_A2: calculating from scratch") # DEBUGGING
        if self.__number_of_pulses <= 0:
            return result
        for i in range(self.__number_of_pulses):
            w0 = self.__omega0_list[i]
            A0 = self.__A0_list[i]
            if A0 == 0:
                continue
            tau = self.__tau_list[i]
            assert(self.__envelope_N_list[i] == 8)
            tt = t - self.__t0_list[i]
            cep = self.__CEP_list[i]
            result += np.where(np.abs(tt) < tau, cos8_int_A2(tt, A0, w0, tau, cep), 0)
            result[tt >= tau] += cos8_int_A2(tau, A0, w0, tau, cep)
        if self.__cache_results == True:
            self.__cached_int_A2 = result.copy()
        return result
    
    def int_E2(self, t) :
        " calculate the cumulative integral of E^2(t) "
        t = np.asarray(t)
        if self.__cache_results == True:
            if self.__cached_t_for_int_E2 is None or not(np.array_equal(t, self.__cached_t_for_int_E2)):
                self.__cached_t_for_int_E2 = t.copy()
            elif not(self.__cached_int_E2 is None):
                return self.__cached_int_E2
        result = np.zeros(t.size)
        # print("LaserField.int_E2: calculating from scratch") # DEBUGGING
        if self.__number_of_pulses <= 0:
            return result
        for i in range(self.__number_of_pulses):
            w0 = self.__omega0_list[i]
            A0 = self.__A0_list[i]
            if A0 == 0:
                continue
            tau = self.__tau_list[i]
            assert(self.__envelope_N_list[i] == 8)
            tt = t - self.__t0_list[i]
            cep = self.__CEP_list[i]
            result += np.where(np.abs(tt) < tau, cos8_int_E2(tt, A0, w0, tau, cep), 0)
            result[tt >= tau] += cos8_int_E2(tau, A0, w0, tau, cep)
        if self.__cache_results == True:
            self.__cached_int_E2 = result.copy()
        return result
    def w0(self):
        return self.__omega0_list
    
@njit()
def meshgrid(x, y):
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in range(y.size):
        for k in range(x.size):
            xx[k, j] = x[k]  # change to x[k] if indexing xy
            yy[k, j] = y[j]  # change to y[j] if indexing xy
    return xx, yy

#@njit()
def get_momentum_grid(div_p, div_theta, laser_field, Ip):
    tot_energy=laser_field.get_total_energy(Ip)
    tau_injection=np.diff(laser_field.get_time_interval())[0]
    p_phys_max = np.sqrt(10*tot_energy)

    dcostheta=1/(2**2*p_phys_max)/div_theta
    density_theta=int(np.log2(2/dcostheta-1))+1
    length_cos=min(max(15,int(2**(density_theta))+1),27)
    Theta_grid=np.linspace(0, np.pi, length_cos)
    intA=laser_field.int_A(laser_field.get_tgrid())
    Int_max=max(max(intA)-min(intA),1)
    dp=min(np.pi/(2*tau_injection*4), np.pi/abs(4*Int_max))/div_p
    p_grid=np.unique(np.concatenate((np.arange(0, p_phys_max+dp, dp)
                                            , np.arange(p_phys_max+dp, 2*p_phys_max, dp*4))))
    window = soft_window(p_grid, p_phys_max, 2*p_phys_max)
    return p_grid, Theta_grid, window

def fwhmOC_to_fwhmAU(fwhmOC, wavel):
    return fwhmOC*wavel/299792458/2.418884328*10**8

def create_pulse(wavel, intens, cep, fwhmCyc):
    tmpObj = LaserField()
    fwhmAU = fwhmOC_to_fwhmAU(fwhmCyc, wavel)
    tmpObj.add_pulse(central_wavelength=wavel, peak_intensity=intens, CEP=cep, FWHM=fwhmAU)
    return tmpObj
