"""Field functions for laser pulse calculations.

This module provides classes and functions to define and manipulate laser pulses.
It includes functionality for calculating vector potentials, electric fields,
and various integrals needed for ionization calculations.

The main class is LaserField which represents a laser pulse with configurable
parameters like wavelength, intensity, CEP, and envelope shape.

Typical usage example:
    laser = LaserField()
    laser.add_pulse(800, 1e14, 0, 30)  # 800nm, 1e14 W/cm2, CEP=0, 30fs FWHM
    t = np.linspace(-100, 100, 1000)
    E = laser.Electric_Field(t)
"""

from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
from numba import njit
from lmfit import minimize, Parameters
import numexpr as ne


class AtomicUnits:
    """Constants in atomic units.
    
    This class provides conversion factors between SI units and atomic units,
    as well as fundamental constants in atomic units.
    """
    
    meter: float = 5.2917720859e-11  # atomic unit of length in meters
    nm: float = 5.2917721e-2  # atomic unit of length in nanometres
    second: float = 2.418884328e-17  # atomic unit of time in seconds
    fs: float = 2.418884328e-2  # atomic unit of time in femtoseconds
    Joule: float = 4.359743935e-18  # atomic unit of energy in Joules
    eV: float = 27.21138383  # atomic unit of energy in electronvolts
    Volts_per_meter: float = 5.142206313e+11  # atomic unit of electric field in V/m
    Volts_per_Angstrom: float = 51.42206313  # atomic unit of electric field in V/Angström
    speed_of_light: float = 137.035999  # vacuum speed of light in atomic units
    Coulomb: float = 1.60217646e-19  # atomic unit of electric charge in Coulombs
    PW_per_cm2_au: float = 0.02849451308  # PW/cm^2 in atomic units



@njit(fastmath=True, cache=True)
def cosN_vector_potential(
    t: Union[float, npt.NDArray[np.float64]],
    A0: float,
    w0: float,
    tau: float,
    cep: float,
    N: int
) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate vector potential for a cosN envelope pulse.

    Args:
        t: Time point(s)
        A0: Peak vector potential amplitude
        w0: Central frequency
        tau: Pulse duration
        cep: Carrier-envelope phase
        N: Power of cosine envelope

    Returns:
        Vector potential value(s)
    """
    return -A0 * np.cos(0.5*np.pi*t/tau)**N * np.sin(w0*t-cep)


@njit(fastmath=True, cache=True)
def cosN_electric_field(
    t: Union[float, npt.NDArray[np.float64]],
    A0: float,
    w0: float,
    tau: float,
    cep: float,
    N: int
) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate electric field for a cosN envelope pulse.

    Args:
        t: Time point(s)
        A0: Peak vector potential amplitude
        w0: Central frequency
        tau: Pulse duration
        cep: Carrier-envelope phase
        N: Power of cosine envelope

    Returns:
        Electric field value(s)
    """
    x = w0 * t - cep
    y = 0.5 * np.pi*t / tau
    return A0*(np.cos(y))**(N-1) * (
        w0*np.cos(x)*np.cos(y) - (N*np.pi*np.sin(x)*np.sin(y))/(2.*tau)
    )


@njit(fastmath = True, cache = True)
def cos8_int_A(t: Union[float, npt.NDArray[np.float64]], A0: float, w0: float, tau: float, cep: float):
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

@njit(fastmath=True, cache=True)
def cos8_int_A2(t: Union[float, npt.NDArray[np.float64]], A0: float, w0: float, tau: float, cep: float) -> Union[float, npt.NDArray[np.float64]]:
    """Calculate the cumulative integral of A^2(t) for the cos8 envelope.
    
    Optimized version for Numba compilation.
    
    Args:
        t: Time point(s)
        A0: Peak vector potential amplitude
        w0: Central frequency
        tau: Pulse duration
        cep: Carrier-envelope phase
        
    Returns:
        Integrated value of A^2
    """
    t = np.asarray(t)
    x = w0 * t - cep
    theta = w0 * tau
    thetaOverPi = theta / np.pi    
    
    # Pre-calculate common terms    
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

@njit(fastmath = True, cache = True)
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



#####################

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

def fwhmOC_to_fwhmAU(fwhmOC, wavel):
    return fwhmOC*wavel/299792458/2.418884328*10**8

def create_pulse(wavel, intens, cep, fwhmCyc):
    """
    Create a LaserField object with specified parameters.
    Args:
        wavel (float): Central wavelength in nm.
        intens (float): Peak intensity in W/cm^2.
        cep (float): Carrier envelope phase in radians.
        fwhmCyc (float): FWHM of the pulse in optical cycles.
    Returns:
        LaserField: An instance of the LaserField class with the specified parameters.
    """
    tmpObj = LaserField()
    fwhmAU = fwhmOC_to_fwhmAU(fwhmCyc, wavel)
    # print("FWHM in AU: ", fwhmAU) # DEBUGGING
    tmpObj.add_pulse(central_wavelength=wavel, peak_intensity=intens, CEP=cep, FWHM=fwhmAU)
    return tmpObj

def ret_pulse_from_pandas_table(data):
    """
    Create a list of LaserField objects from a pandas dataframe."""
    pulses=[]
    if "FWHM_OC" in data:
        for dat in data.iterrows():
            dat=dat[1]
            pulses.append(create_pulse(wavel=dat.wavel, intens=dat.intens, cep=dat.cep, fwhmCyc=dat.FWHM_OC))   
        return pulses
    elif "fwhmau" in data:
        for dat in data.iterrows():
            dat=dat[1]
            pulse=LaserField()
            pulse.add_pulse(central_wavelength=dat.wavel, peak_intensity=dat.intens, CEP=dat.cep, FWHM=dat.fwhmau)
            pulses.append(pulse)   
        return pulses
    else:
        raise ValueError("The dataframe does not contain the required columns. Try modifying dataFrame or ret_pulse_from_pandas_table function to correctly inherit pulse duration")

#@njit(fastmath=True, cache=True)
def _compute_vector_potential_core(t: np.ndarray, w0: float, A0: float, cep: float, tau: float, N: int, t0: float) -> np.ndarray:
    """Numba-optimized core computation for vector potential."""
    tt = t - t0
    mask = np.abs(tt) < tau
    result = np.zeros_like(t, dtype=np.float64)
    
    for i in range(len(t)):
        if mask[i]:
            result[i] = cosN_vector_potential(tt[i], A0, w0, tau, cep, N)
    
    return result

#@njit(fastmath=True, cache=True)
def _compute_electric_field_core(t: np.ndarray, w0: float, A0: float, cep: float, tau: float, N: int, t0: float) -> np.ndarray:
    """Numba-optimized core computation for electric field."""
    tt = t - t0
    mask = np.abs(tt) < tau
    result = np.zeros_like(t, dtype=np.float64)
    
    for i in range(len(t)):
        if mask[i]:
            result[i] = cosN_electric_field(tt[i], A0, w0, tau, cep, N)
    
    return result

#@njit(fastmath=True, cache=True)
def _compute_int_A_core(t: np.ndarray, w0: float, A0: float, cep: float, tau: float, N: int, t0: float) -> np.ndarray:
    """Numba-optimized core computation for integrated vector potential."""
    tt = t - t0
    result = np.zeros_like(t, dtype=np.float64)
    
    for i in range(len(t)):
        if abs(tt[i]) < tau:
            result[i] = cos8_int_A(tt[i], A0, w0, tau, cep)
        elif tt[i] >= tau:
            result[i] = cos8_int_A(tau, A0, w0, tau, cep)
    
    return result

#@njit(fastmath=True, cache=True)
def _compute_int_A2_core(t: np.ndarray, w0: float, A0: float, cep: float, tau: float, N: int, t0: float) -> np.ndarray:
    """Numba-optimized core computation for integrated squared vector potential."""
    tt = t - t0
    result = np.zeros_like(t, dtype=np.float64)
    
    for i in range(len(t)):
        if abs(tt[i]) < tau:
            result[i] = cos8_int_A2(tt[i], A0, w0, tau, cep)[0]
        elif tt[i] >= tau:
            result[i] = cos8_int_A2(tau, A0, w0, tau, cep)[0]
    
    return result

@njit(fastmath=True, cache=True)
def _compute_int_E2_core(t: np.ndarray, w0: float, A0: float, cep: float, tau: float, N: int, t0: float) -> np.ndarray:
    """Numba-optimized core computation for integrated squared electric field."""
    tt = t - t0
    result = np.zeros_like(t, dtype=np.float64)
    
    for i in range(len(t)):
        if abs(tt[i]) < tau:
            result[i] = cos8_int_E2(tt[i], A0, w0, tau, cep)[0]
        elif tt[i] >= tau:
            result[i] = cos8_int_E2(tau, A0, w0, tau, cep)[0]
    
    return result



class LaserField:
    """Class representing a laser pulse with configurable parameters.
    
    This class provides methods to define and calculate various properties
    of laser pulses, including vector potentials, electric fields, and
    their integrals.

    Attributes:
        cache_results: Whether to cache computed results
        _pulses: List of pulse parameters
        _cache: Dictionary storing cached results
    """

    def __init__(self, cache_results: bool = True):
        """Initialize LaserField.

        Args:
            cache_results: Whether to cache computed results
        """
        self._pulses = []
        self._cache = {}
        self.cache_results = cache_results
        # Initialize cache variables
        self._cached_t_for_A = None
        self._cached_A = None
        self._cached_t_for_E = None
        self._cached_E = None
        self._cached_t_for_int_A = None
        self._cached_int_A = None
        self._cached_t_for_int_A2 = None
        self._cached_int_A2 = None
        self._cached_t_for_int_E2 = None
        self._cached_int_E2 = None

    def add_pulse(
        self,
        central_wavelength: float,
        peak_intensity: float,
        CEP: float,
        FWHM: float,
        envelope_N: int = 8,
        t0: float = 0.0
    ) -> None:
        """Add a new pulse to the field.

        Args:
            central_wavelength: Central wavelength in nm
            peak_intensity: Peak intensity in W/cm^2
            CEP: Carrier-envelope phase in radians
            FWHM: Full width at half maximum in fs
            envelope_N: Power of cosine envelope (default: 8)
            t0: Time offset (default: 0.0)
        """
        au = AtomicUnits()
        w0 = 2*np.pi*au.speed_of_light / (central_wavelength / au.nm) # angular frequency   
        A0 = np.sqrt(peak_intensity/1e15 * 0.02849451308 / w0**2) # amplitude
        tau = np.pi*FWHM/(4*np.arccos(2**(-1/(2*envelope_N)))) # duration
        self._pulses.append((w0, A0, CEP, tau, envelope_N, t0))
    
    # def add_pulse_from_field_trace(self, time_stamps: np.ndarray, field_trace: np.ndarray, t0: float = 0.0) -> None:
    #     """Add a pulse from x and y data.

    #     Args:
    #         time_stamps: Array of  time stamps at which field was sampled (in atomic units)
    #         field_trace: Array of measured field values (in atomic units)
    #         t0: Time offset (default: 0.0) : useful in case aligning with other pulses
    #     """
    #     if len(time_stamps) != len(field_trace):
    #         raise ValueError("Time stamps and field trace must have the same length.")
        
        
    def get_central_wavelength(self) -> float:
        """Get the central wavelength of the pulse.

        Returns:
            Central wavelength in nm
        """
        au = AtomicUnits()
        return [2*np.pi*au.speed_of_light/(pulse[0])*au.nm for pulse in self._pulses]


    def get_ponderomotive_energy(self) -> float:
        """Calculate ponderomotive energy.

        Returns:
            Ponderomotive energy in atomic units
        """
        if not self._pulses:
            return 0.0
        Up=0.0
        Up = max(self.Vector_potential(self.get_tgrid()))**2/4
        return Up

    def get_initial_KE_energy(self, Ip: float) -> float:
        """Calculate initial kinetic energy.

        Args:
            Ip: Ionization potential in atomic units

        Returns:
            Initial kinetic energy in atomic units
        """
        if not self._pulses:
            return 0.0
        initKE=[]
        for i in range(len(self._pulses)):
            nPhoton=int(Ip / (self._pulses[i][0] ))+1
            initKE.append(nPhoton* self._pulses[i][0] - Ip)
        return np.mean(initKE)

    def get_total_energy(self, Ip: float) -> float:
        """Calculate total energy.

        Args:
            Ip: Ionization potential in atomic units

        Returns:
            Total energy in atomic units
        """
        return self.get_initial_KE_energy(Ip) + self.get_ponderomotive_energy()

    def get_time_interval(self) -> Tuple[float, float]:
        """Get time interval containing the pulse.

        Returns:
            Tuple of (start_time, end_time) in atomic units
        """
        if not self._pulses:
            return((None, None))
        t_min = self._pulses[0][5] - self._pulses[0][3]
        t_max = self._pulses[0][5] + self._pulses[0][3]
        for i in range(1, len(self._pulses)):
            t_min = min(t_min, self._pulses[i][5] - self._pulses[i][3])
            t_max = max(t_max, self._pulses[i][5] + self._pulses[i][3])
        return (t_min, t_max)
    
    def get_tgrid(self, npoint=1000):
        t_min, t_max = self.get_time_interval()
        if t_min is None or t_max is None:
            return None
        return np.linspace(t_min, t_max, npoint)
    
    def get_tGrid_dt(self, dt=1.):
        t_min, t_max = self.get_time_interval()
        if t_min is None or t_max is None:
            return None
        return np.arange(t_min, t_max+dt, dt)
    
    def reset(self, cache_results=True):
        self._pulses = []
        self._cache = {}
        self.cache_results = cache_results
        self._cached_t_for_A = None
        self._cached_A = None
        self._cached_t_for_E = None
        self._cached_E = None
        self._cached_t_for_int_A = None
        self._cached_int_A = None
        self._cached_t_for_int_A2 = None
        self._cached_int_A2 = None
        self._cached_t_for_int_E2 = None
        self._cached_int_E2 = None

    def Vector_potential(self, t):
        t = np.asarray(t)
        if self.cache_results == True:
            if self._cached_t_for_A is None or not(np.array_equal(t, self._cached_t_for_A)):
                self._cached_t_for_A = t.copy()
            elif not(self._cached_A is None):
                return self._cached_A
        
        Field = np.zeros(t.size)
        if not self._pulses:
            return Field
            
        for w0, A0, cep, tau, N, t0 in self._pulses:
            if A0 == 0:
                continue
            Field += _compute_vector_potential_core(t, w0, A0, cep, tau, N, t0)
            
        if self.cache_results == True:
            self._cached_A = Field.copy()
        return Field

    def Electric_Field(self, t):
        t = np.asarray(t)
        if self.cache_results == True:
            if self._cached_t_for_E is None or not(np.array_equal(t, self._cached_t_for_E)):
                self._cached_t_for_E = t.copy()
            elif not(self._cached_E is None):
                return self._cached_E
                
        Field = np.zeros(t.size)
        if not self._pulses:
            return Field
            
        for w0, A0, cep, tau, N, t0 in self._pulses:
            if A0 == 0:
                continue
            Field += _compute_electric_field_core(t, w0, A0, cep, tau, N, t0)
            
        if self.cache_results == True:
            self._cached_E = Field.copy()
        return Field

    def int_A(self, t):
        t = np.asarray(t)
        if self.cache_results == True:
            if self._cached_t_for_int_A is None or not(np.array_equal(t, self._cached_t_for_int_A)):
                self._cached_t_for_int_A = t.copy()
            elif not(self._cached_int_A is None):
                return self._cached_int_A
                
        result = np.zeros(t.size)
        if not self._pulses:
            return result
            
        for w0, A0, cep, tau, N, t0 in self._pulses:
            if A0 == 0:
                continue
            assert(N == 8)
            result += _compute_int_A_core(t, w0, A0, cep, tau, N, t0)
            
        if self.cache_results == True:
            self._cached_int_A = result.copy()
        return result

    def int_A2(self, t):
        t = np.asarray(t)
        if self.cache_results == True:
            if self._cached_t_for_int_A2 is None or not(np.array_equal(t, self._cached_t_for_int_A2)):
                self._cached_t_for_int_A2 = t.copy()
            elif not(self._cached_int_A2 is None):
                return self._cached_int_A2
                
        result = np.zeros(t.size)
        if not self._pulses:
            return result
        elif len(self._pulses) == 1:
            w0, A0, cep, tau, N, t0 = self._pulses[0]
            assert(N == 8)
            return _compute_int_A2_core(t, w0, A0, cep, tau, N, t0)
        tmp_t = self.get_tGrid_dt(dt=0.125/16.)
        tmp_A = self.Vector_potential(tmp_t) 
        from scipy.integrate import cumulative_simpson
        tmp_A2 = cumulative_simpson(y=tmp_A**2, x=tmp_t, initial=0) # Numerical integration
        # interpolate the result to the original time grid
        from scipy.interpolate import interp1d
        interp_func = interp1d(tmp_t, tmp_A2, bounds_error=False, fill_value=(tmp_A2[0], tmp_A2[-1]))
        result = interp_func(t)
        

        # for w0, A0, cep, tau, N, t0 in self._pulses:
        #     if A0 == 0:
        #         continue
        #     assert(N == 8)
        #     result += _compute_int_A2_core(t, w0, A0, cep, tau, N, t0)
        


            
        if self.cache_results == True:
            self._cached_int_A2 = result.copy()
        return result

    def int_E2(self, t):
        t = np.asarray(t)
        if self.cache_results == True:
            if self._cached_t_for_int_E2 is None or not(np.array_equal(t, self._cached_t_for_int_E2)):
                self._cached_t_for_int_E2 = t.copy()
            elif not(self._cached_int_E2 is None):
                return self._cached_int_E2
                
        result = np.zeros(t.size)
        if not self._pulses:
            return result
        elif len(self._pulses) == 1:
            w0, A0, cep, tau, N, t0 = self._pulses[0]
            assert(N == 8)
            return _compute_int_E2_core(t, w0, A0, cep, tau, N, t0)
        tmp_t = self.get_tGrid_dt(dt=0.125/4.)
        tmp_E = self.Electric_Field(tmp_t)
        from scipy.integrate import cumulative_simpson
        tmp_E2 = cumulative_simpson(y=tmp_E**2, x=tmp_t, initial=0)
        # interpolate the result to the original time grid
        from scipy.interpolate import interp1d
        interp_func = interp1d(tmp_t, tmp_E2, bounds_error=False, fill_value=(tmp_E2[0], tmp_E2[-1]))
        result = interp_func(t)
            
        # for w0, A0, cep, tau, N, t0 in self._pulses:
        #     if A0 == 0:
        #         continue
        #     assert(N == 8)
        #     result += _compute_int_E2_core(t, w0, A0, cep, tau, N, t0)
            
        if self.cache_results == True:
            self._cached_int_E2 = result.copy()
        return result

    def w0(self):
        return [w0 for w0, _, _, _, _, _ in self._pulses]
        
      
class PulseFitter:
    """
    A class to fit experimental electric field data with multiple cos8 pulses and return the corresponding laser field object.
    """
    def __init__(self, x_data, y_data, num_pulses=25, phase_shift=0):
        """
        Initialize the pulse fitter.
        
        Args:
            x_data (np.ndarray): Time data points (in atomic units)
            y_data (np.ndarray): Electric field data points (in atomic units)
            num_pulses (int): Number of cos8 pulses to use for fitting
            phase_shift (float): Phase shift to apply to the data so that one gets desired cep, for example: cosine pulse with single strongest peak at t=0
        """
        self.num_pulses = num_pulses
        self.normFactor = 1
        self.params = None
        self.xdata = x_data
        self.ydata = y_data
        self.ydata_norm = self.ydata / max(abs(self.ydata))  # Normalize the data
        self.fit_result = None
        self.fitted_signal = None
        self.phase_shift = phase_shift
        self.fit_pulse = LaserField(cache_results=False)
    def _setup_parameters(self, minI=1e12, maxI=1e16, minl=250, maxl=2000, minFWHM=1, maxFWHM=6, minshift=-100, maxshift=100):
        """Set up initial parameters for the fit."""
        params = Parameters()
        params.add('c1', value=2, min=1, max=20)
        
        for n in range(self.num_pulses):
            # Initial wavelength distributed across 100-1300nm range
            params.add(f'I{n}', value=1e15, min=minI, max=maxI)
            params.add(f'l{n}', value=100+1200/self.num_pulses*n, min=minl, max=maxl)
            params.add(f'FWHM{n}', value=2, min=minFWHM, max=maxFWHM)
            params.add(f'shift{n}', value=self.phase_shift, min=minshift, max=maxshift)
            params.add(f'cep{n}', value=0, min=-np.pi, max=np.pi)
            
        self.params = params
    
    
    def _residual(self, params, x, data, uncertainty):
        """Calculate residual for fitting."""
        pulse = LaserField(cache_results=False)
        pulse.reset()
        for n in range(self.num_pulses):
            pulse.add_pulse(
                central_wavelength=params[f'l{n}'].value, 
                peak_intensity=params[f'I{n}'].value, 
                CEP=params[f'cep{n}'].value, 
                FWHM=params[f'FWHM{n}'].value, 
                t0=params[f'shift{n}'].value
            )
        return (data - params['c1']* pulse.Electric_Field(x)) / uncertainty
    
    def fit(self, minI=1e12, maxI=1e16, minl=250, maxl=2000, minFWHM=1, maxFWHM=6, minshift=-100, maxshift=100):
        """Perform the fitting procedure.
        Args:
            minI (float): Minimum intensity for fitting
            maxI (float): Maximum intensity for fitting
            minl (float): Minimum wavelength for fitting
            maxl (float): Maximum wavelength for fitting
            minFWHM (float): Minimum FWHM for fitting
            maxFWHM (float): Maximum FWHM for fitting
            minshift (float): Minimum shift for fitting
            maxshift (float): Maximum shift for fitting
            """
        print('Fitting...')
        if self.params is None:
            self._setup_parameters(minI, maxI, minl, maxl, minFWHM, maxFWHM, minshift, maxshift)

        # Perform the fit
        self.fit_result = minimize(self._residual, self.params, 
                                 args=(self.xdata, self.ydata_norm, 0.1))
        
        # Calculate fitted signal
        self.fitted_signal = 0
        self.fit_pulse.reset()
        for n in range(self.num_pulses):
            self.fit_pulse.add_pulse(
                central_wavelength=self.fit_result.params[f'l{n}'].value, 
                peak_intensity=self.fit_result.params[f'I{n}'].value, 
                CEP=self.fit_result.params[f'cep{n}'].value, 
                FWHM=self.fit_result.params[f'FWHM{n}'].value, 
                t0=self.fit_result.params[f'shift{n}'].value
            )
        self.normFactor=max((self.ydata)**2) / max((self.fit_pulse.Electric_Field(self.xdata))**2)
        print('Fitting done')

    
    def get_pulse(self, target_intensity=None, plot_results=False):
        """Return the time grid and the electric field of the fitted pulse."""
        if self.fit_result is None:
            raise ValueError("Must run fit() before getting electric field")
        elif target_intensity is None:
            self.fit_pulse.reset()
            for n in range(self.num_pulses):
                self.fit_pulse.add_pulse(
                    central_wavelength=self.fit_result.params[f'l{n}'].value, 
                    peak_intensity=self.normFactor*self.fit_result.params[f'I{n}'].value, 
                    CEP=self.fit_result.params[f'cep{n}'].value, 
                    FWHM=self.fit_result.params[f'FWHM{n}'].value, 
                    t0=self.fit_result.params[f'shift{n}'].value
                )
        elif target_intensity is not None:
            self.ydata = self.ydata_norm*(target_intensity*AtomicUnits.PW_per_cm2_au/1e15)**0.5
            self.normFactor = target_intensity*AtomicUnits.PW_per_cm2_au/1e15 / max(abs(self.fit_pulse.Electric_Field(self.xdata))**2)
            self.fit_pulse.reset()
            for n in range(self.num_pulses):
                self.fit_pulse.add_pulse(
                    central_wavelength=self.fit_result.params[f'l{n}'].value, 
                    peak_intensity=self.normFactor * self.fit_result.params[f'I{n}'].value, 
                    CEP=self.fit_result.params[f'cep{n}'].value, 
                    FWHM=self.fit_result.params[f'FWHM{n}'].value, 
                    t0=self.fit_result.params[f'shift{n}'].value
                )
        if plot_results:
            self.plot_results()
        return self.fit_pulse
    
    def plot_results(self):
        """Plot the fitting results."""
        import matplotlib.pyplot as plt
        if self.fit_result is None:
            raise ValueError("Must run fit() before plotting results")
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.xdata, self.ydata, label='Experimental data')
        plt.plot(self.xdata, self.fit_pulse.Electric_Field(self.xdata), 
                alpha=0.8, label='Fitted')
        plt.legend(loc='upper right', title=f'N={self.num_pulses}')
        plt.xlabel('Time (atomic units)')
        plt.ylabel('Electric Field (normalized)')
        plt.title('Pulse Fitting Results')
        plt.show()
        print("if not happy with the fit, try the number of pulses and if that is insufficient, try to change the min and max values of the parameters in fit()")
  

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
    length_cos=max(15,int(2**(density_theta))+1)
    Theta_grid=np.linspace(0, np.pi, length_cos)
    intA=laser_field.int_A(laser_field.get_tgrid())
    Int_max=max(max(intA)-min(intA),1)
    dp=min(10*np.pi/(tau_injection), np.pi/abs(4*Int_max))/div_p # np.pi/(2*tau_injection*4)
    p_grid=np.unique(np.concatenate((np.arange(0, p_phys_max+dp, dp)
                                            , np.arange(p_phys_max+dp, 2*p_phys_max, dp*8))))
    window = soft_window(p_grid, p_phys_max, 2*p_phys_max)
    return p_grid, Theta_grid, window

def fwhmOC_to_fwhmAU(fwhmOC, wavel):
    return fwhmOC*wavel/299792458/2.418884328*10**8

def create_pulse(wavel, intens, cep, fwhmCyc):
    tmpObj = LaserField()
    fwhmAU = fwhmOC_to_fwhmAU(fwhmCyc, wavel)
    # print("FWHM in AU: ", fwhmAU) # DEBUGGING
    tmpObj.add_pulse(central_wavelength=wavel, peak_intensity=intens, CEP=cep, FWHM=fwhmAU)
    return tmpObj

@njit(parallel=True, fastmath = False,cache = False)
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