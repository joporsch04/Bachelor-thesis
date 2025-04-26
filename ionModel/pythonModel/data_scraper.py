import numpy as np
import pandas as pd
import kernels as kern
from tqdm import tqdm

IpEV_dict = {
    "H": 0.5*27.2114, "He": 24.587, "He_Hacc5": 24.59, "Ne": 21.564, "Kr": 14.0, "Xe": 12.13,
    "Ar": 15.7596, "H2_Hacc5": 15.439, "CO_Hacc5": 14.3, "N2_Hacc5": 14.531, "Rb": 4.1771,
    "Cs": 3.8939, "Fr": 4.0727, "K": 4.3407, 'SiO2': 8.96
}

αPol_dict = {
    "H": 4.51, "He": 1.38, "Ne": 2.66, "Kr": 16.8, "Xe": 27.3, "Ar": 11.1,
    "He_Hacc5": 1.38, "H2_Hacc5": 0, "CO_Hacc5": 13.08, "N2_Hacc5": 11.5, "Rb": 320,
    "Cs": 401, "Fr": 318, "K": 290, "SiO2": 6.16
}
import glob
from scipy.integrate import simpson
from field_functions import LaserField
# tmp=pd.read_table("/home/manoram/Downloads/outspec_Hfin.dat", comment='#', sep=" +", engine='python', names=["file_name", "Y"], usecols=[0,1], skiprows=1)
tmp=pd.read_table("GASFIR/Data/Unprocessed/tRecX_data.dat", comment='#', sep=" +", engine='python', names=["file_name", "Y"], usecols=[0,1], skiprows=1)
tRecX_data=pd.DataFrame({"wavel":tmp.file_name.apply(lambda x: float(x.split("lambda_")[1].split("_nm/")[0])), 
                         "intens":tmp.file_name.apply(lambda x: float(x.split("I_")[1].split("_Wpcm2/")[0])), 
                         "FWHM_OC":tmp.file_name.apply(lambda x: float(x.split("FWHM_")[1].split("OC/")[0])),
                         "cep":tmp.file_name.apply(lambda x: np.pi/180*float(x.split("CEP_")[1].split("deg/")[0])),
                         "Y":tmp.Y})
tRecX_data["fwhmau"]=tRecX_data["FWHM_OC"]*tRecX_data.wavel/299792458/2.418884328*10**8
tRecX_data["Function_to_fit"]="IonProb"
tRecX_data["file_name"]=tmp.file_name
tRecX_data.to_csv("GASFIR/Data/Unprocessed/tRecX_data.csv", index=False)
pulses=[]
SFA_data_ar=[]
pbar=tqdm(total=len(tRecX_data))
for dat in tRecX_data.iterrows():
    dat=dat[1]
    pulse=LaserField()
    pulse.add_pulse(central_wavelength=dat.wavel, peak_intensity=dat.intens, CEP=dat.cep, FWHM=dat.fwhmau)
    pulses.append(pulse)   
    prob, data= kern.IonProb(pulse, {"E_g":0.5, "div_p":2**-4, "div_theta":1}, dT=0.25, kernel_type="exact_SFA", ret_Rate=True)
    tmpframe=pd.DataFrame({"wavel":dat.wavel, 
                         "intens":dat.intens, 
                         "cep": dat.cep,
                         "fwhmau": dat.fwhmau,
                         "FWHM_OC":dat.FWHM_OC,
                         "Y":prob,
                         "t":[np.NaN],
                         "rate":[np.NaN]})
    tmpframe["t"]=tmpframe["t"].astype(object)
    tmpframe["rate"]=tmpframe["rate"].astype(object)
    tmpframe.at[0, "t"]= data[0]
    tmpframe.at[0, "rate"]= data[1]
    pbar.update(1)
    SFA_data_ar.append(tmpframe)
    SFA_data=pd.concat(SFA_data_ar)
    SFA_data.to_csv("GASFIR/Data/Unprocessed/SFA_data_new.csv", index=False)
SFA_data=pd.concat(SFA_data_ar)
SFA_data.to_csv("GASFIR/Data/Unprocessed/SFA_data_new.csv", index=False)
SFA_data["pulses"]=pulses
SFA_data.to_pickle("GASFIR/Data/Unprocessed/SFA_data_new.pkl")    
tRecX_data["pulses"]=pulses
tRecX_data.to_pickle("GASFIR/Data/Unprocessed/tRecX_data.pkl")

SFA_data=[]
for file in glob.glob("/home/manoram/Desktop/Hydrogen_paper_data/SFA/7_Oct_23/Exdx/tRecX/*/*/*/*/*/*/*/G.csv"):

    wavel=float(file.split("lambda_")[1].split("_nm/")[0])
    intens=float(file.split("I_")[1].split("_Wpcm2/")[0])
    cep=np.pi/180*float(file.split("CEP_")[1].split("deg/")[0])
    FWHM_OC=float(file.split("FWHM_")[1].split("OC/")[0])
    fwhmau=FWHM_OC*wavel/299792458/2.418884328*10**8
    data=pd.read_csv(file, usecols=["t", "G_gamma"])
    prob=1-np.exp(-simpson(y=data.G_gamma,x=data.t))
    tmpframe=pd.DataFrame({"wavel":wavel, 
                         "intens":intens, 
                         "cep": cep,
                         "fwhmau": fwhmau,
                         "FWHM_OC":FWHM_OC,
                         "Y":prob,
                         "t": data.t.tolist(), 
                         "rate": data.G_gamma.tolist()})
    tmpframe=pd.DataFrame({"wavel":wavel, 
                         "intens":intens, 
                         "cep": cep,
                         "fwhmau": fwhmau,
                         "FWHM_OC":FWHM_OC,
                         "Y":prob,
                         "t":[np.NaN],
                         "rate":[np.NaN]})
    tmpframe["t"]=tmpframe["t"].astype(object)
    tmpframe["rate"]=tmpframe["rate"].astype(object)
    tmpframe.at[0, "t"]= data.t.tolist()
    tmpframe.at[0, "rate"]= data.G_gamma.tolist()
    # print(tmpframe)
    SFA_data.append(tmpframe)
SFA_data=pd.concat(SFA_data)
SFA_data.to_csv("GASFIR/Data/Unprocessed/SFA_data.csv", index=False)
pulses=[]
for dat in SFA_data.iterrows():
    dat=dat[1]
    pulse=LaserField()
    pulse.add_pulse(central_wavelength=dat.wavel, peak_intensity=dat.intens, CEP=dat.cep, FWHM=dat.fwhmau)
    pulses.append(pulse)   
SFA_data["pulses"]=pulses
SFA_data.to_pickle("GASFIR/Data/Unprocessed/SFA_data.pkl")