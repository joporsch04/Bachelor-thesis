import kernels as kern
import numpy as np
from field_functions import LaserField
from lmfit import Model
from field_functions import create_pulse

IonProb=np.vectorize(kern.IonProb)
IonRate=np.vectorize(kern.IonRate, otypes=[np.ndarray])
def model_unified(params, data):
    from field_functions import LaserField
    try:
        params=params.valuesdict()
    except:
        print("params has no valuesdict method")
        print("assuming it is already a dictionary")
    model=0#np.zeros(data.Y.shape)
    if type(data.Function_to_fit) is not str:    
        Function_to_fit=data.Function_to_fit.to_numpy()
        model=np.zeros(Function_to_fit.shape)
    
    if np.any(Function_to_fit) == "Log_QSRate":
        model = np.log(kern.QSRate(data.field.to_numpy(), param_dict=params))
    elif np.any(Function_to_fit) == "QSRate":
        model = kern.QSRate(data.field.to_numpy(), param_dict=params)
    else:
        pulses=data.pulses
        if np.any(Function_to_fit) == "Kernel":
            model = kern.Kernel_jit(data.t, data.T, pulses, param_dict=params)
        if np.any(Function_to_fit) == "IonRate":
            model = IonRate(data.t, pulses, param_dict=params, dT=0.25)
        if np.any(Function_to_fit) == "IonProb":
            # print(e0, a0, a1, b0, b1, b2, c2, p1, d1, Ip, αPol)
            model = IonProb(pulses, param_dict=params, dt=2, dT=1, filterTreshold=0.0)
    #print(model)
    return np.nan_to_num(model)
def model_calc_diabatic_Probabilty(central_wavelength, peak_intensity, CEP, fwhmCyc, t0, t1, t2, t3, t4, e1, e2, tau, αPol, E_g):
    pulse=create_pulse(central_wavelength, peak_intensity, CEP, fwhmCyc)
    params = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4, "e1": e1, "e2": e2, "tau": tau, "αPol": αPol, "E_g": E_g}
    return IonProb(pulse, param_dict=params, dt=2, dT=1, filterTreshold=0.0)
Pmodel= Model(model_calc_diabatic_Probabilty, independent_vars=['central_wavelength', 'peak_intensity', 'CEP', 'fwhmCyc'], param_names=['t0', 't1', 't2', 't3', 't4', 'e1', 'e2', 'tau', 'αPol', 'E_g'])

def add_pulses(dat):
    tmpObj = LaserField()
    tmpObj.add_pulse(central_wavelength=dat.wavel, peak_intensity=dat.intens, CEP=dat.cep, FWHM=dat.fwhmau)
    return tmpObj

def create_pulse(wavel, intens, cep, fwhmCyc):
    tmpObj = LaserField()
    fwhmAU = fwhmOC_to_fwhmAU(fwhmCyc, wavel)
    tmpObj.add_pulse(central_wavelength=wavel, peak_intensity=intens, CEP=cep, FWHM=fwhmAU)
    return tmpObj

def fwhmOC_to_fwhmAU(fwhmOC, wavel):
    return fwhmOC*wavel/299792458/2.418884328*10**8

def residuals(params, data_Nadiabatic=None, uncertainty_adiabatic=None, data_QS=None, uncertainty_QS=None):
    """this function will calculate the residuals automatically providing the right input data for a given function
    prams is a special object that allows manipulation of fit parameters, including fixing them and providing a mathematical expression based on another new parameter

    Args:
        params (lmfit.Parameters()): a dictionary like object containing all information about the fit parameters like bounds, initial guess, fixed mathematical expressions etc. 
        x (an array of dictionaryies of lenght(data)): contains all input data relevant for obtaining data given parameters
        data (a 1d array of numbers with corresponding x): the data can combine numbers from different funcitons as long as the corresponding x is the same
        uncertainty (array, len(data)): to be used to weight the dataset according to needs
    """
    res_QS=[0.]
    res_Nad=[0.]
    # if data_Nadiabatic is not None:
    #     res_Nad= (model_unified(params, data_Nadiabatic)-data_Nadiabatic.Y.to_numpy())/uncertainty_adiabatic
    # if data_QS is not None:
    #     res_QS= (model_unified(params, data_QS)-data_QS.Y.to_numpy())/uncertainty_QS

    if data_Nadiabatic is not None:
        res_Nad= (model_unified(params, data_Nadiabatic)-data_Nadiabatic.Y.to_numpy())/uncertainty_adiabatic#/data_Nadiabatic.Y.to_numpy()
    if data_QS is not None:
        res_QS= (model_unified(params, data_QS)-data_QS.Y.to_numpy())/uncertainty_QS#/data_QS.Y.to_numpy()
    return np.concatenate((res_QS,res_Nad))