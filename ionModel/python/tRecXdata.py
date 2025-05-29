import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go

class tRecXdata:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.time0 = self.extractTime0()
        self.field0 = self.extractField0()
        self.coefficients = self.extractCoefficients()
        self.laser_params = self.extractLaserParams()

    def extractTime0(self):
        file_path = os.path.join(self.folder_path, "Laser")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Laser file not found in {self.folder_path}")
        with open(file_path, 'r') as file:
            first_colomn = []
            for row, line in enumerate(file):
                if row < 6:
                    continue
                line = line.strip()
                colomns = line.split()
                if len(colomns) >= 5:
                    first_colomn.append(float(colomns[0]))
        return first_colomn

    def extractField0(self):
        file_path = os.path.join(self.folder_path, "Laser")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Laser file not found in {self.folder_path}")
        with open(file_path, 'r') as file:
            fifth_colomn = []
            for row, line in enumerate(file):
                if row < 6:
                    continue
                line = line.strip()
                colomns = line.split()
                if len(colomns) >= 5:
                    fifth_colomn.append(float(colomns[5]))
        return fifth_colomn

    def extractCoefficients(self):
        file_path = os.path.join(self.folder_path, "expec")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"expec file not found in {self.folder_path}")
        
        with open(file_path, 'r') as file:
            header_row = None
            for row_idx, line in enumerate(file):
                line = line.strip()
                if line.startswith('#      Time') or 'Time' in line and 'CPU' in line:
                    header_row = row_idx
                    break
        
        if header_row is None:
            raise ValueError("Could not find the data header row in the expec file")

        df=pd.read_csv(file_path, sep='\s+', header=header_row)
        df.columns = df.columns[1:].tolist() + [""]
        df = df.iloc[:, :-1]
        return df

    def plotCoefficients(self, state_indices, plot_type="occ"):
        fig = go.Figure()

        if plot_type == "occ":
            for idx in state_indices:
                col_name = f"<Occ{{H0:{idx}}}>"
                if col_name in self.coefficients.columns:
                    fig.add_trace(go.Scatter(x=self.coefficients["Time"], y=self.coefficients[col_name], mode='lines', name=col_name))

        elif plot_type == "real":
            for idx in state_indices:
                col_name = f"Re{{<H0:{idx}|psi>}}"
                if col_name in self.coefficients.columns:
                    fig.add_trace(go.Scatter(x=self.coefficients["Time"], y=self.coefficients[col_name], mode='lines', name=col_name))
        
        elif plot_type == "imag":
            for idx in state_indices:
                col_name = f"Imag{{<H0:{idx}|psi>}}"
                if col_name in self.coefficients.columns:
                    fig.add_trace(go.Scatter(x=self.coefficients["Time"], y=self.coefficients[col_name], mode='lines', name=col_name))
        
        elif plot_type == "mag":
            for idx in state_indices:
                real_col = f"Re{{<H0:{idx}|psi>}}"
                imag_col = f"Imag{{<H0:{idx}|psi>}}"
                if real_col in self.coefficients.columns and imag_col in self.coefficients.columns:
                    magnitude = self.coefficients[real_col]**2 + self.coefficients[imag_col]**2
                    fig.add_trace(go.Scatter(x=self.coefficients["Time"], y=magnitude, mode='lines', name=f"|<H0:{idx}|psi>|²"))
        
        else:
            raise ValueError("plot_type must be one of: 'occupation', 'real', 'imag', 'mag'")
        
        laser_text = "<br>".join([
            "<b>Laser Parameters:</b>",
            f"Shape: {self.laser_params.get('shape', 'N/A')}",
            f"Intensity: {self.laser_params.get('intensity_W_cm2', 'N/A')} W/cm²",
            f"FWHM: {self.laser_params.get('FWHM', 'N/A')}",
            f"Wavelength: {self.laser_params.get('wavelength_nm', 'N/A')} nm",
            f"CEP: {self.laser_params.get('CEP', 'N/A')}"
        ])
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=f"{plot_type.capitalize()} coefficients",
            legend_title="States",
            xaxis=dict(range=[-100, 100]),
            annotations=[
                dict(
                    x=0.02,
                    y=0.98,
                    xref="paper",
                    yref="paper",
                    text=laser_text,
                    showarrow=False,
                    font=dict(size=10),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="gray",
                    borderwidth=1,
                    align="left",
                    valign="top"
                )
            ]
        )
        fig.show()

    def extractLaserParams(self):
        """Extract laser parameters from inpc file"""
        file_path = os.path.join(self.folder_path, "inpc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"inpc file not found in {self.folder_path}")
        
        laser_params = {}
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
            for i, line in enumerate(lines):
                if line.strip().startswith('Laser:') and 'shape' in line:
                    if i + 1 < len(lines):
                        values_line = lines[i + 1].strip()
                        values = values_line.split()
                        
                        if len(values) >= 8:
                            fwhm_value = values[2].rstrip(',')
                            if len(values) >= 9 and values[3].rstrip(',') == 'OptCyc':
                                fwhm_full = f"{fwhm_value} {values[3].rstrip(',')}"
                                wavelength_idx = 4
                            else:
                                fwhm_full = fwhm_value
                                wavelength_idx = 3
                            
                            laser_params = {
                                'shape': values[0].rstrip(','),
                                'intensity_W_cm2': f"{float(values[1].rstrip(',')):.0e}",
                                'FWHM': fwhm_full,
                                'wavelength_nm': float(values[wavelength_idx].rstrip(',')),
                                'polar_angle': float(values[wavelength_idx + 1].rstrip(',')),
                                'azimuth_angle': float(values[wavelength_idx + 2].rstrip(',')),
                                'CEP': float(values[wavelength_idx + 3].rstrip(',')),
                                'peak_time': values[wavelength_idx + 4].rstrip(',') if wavelength_idx + 4 < len(values) else ''
                            }
                        break
        
        if not laser_params:
            raise ValueError("Could not find laser parameters in inpc file")
        
        return laser_params

if __name__ == "__main__":
    datamixed = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0040")
    datalength = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0042")
    datavelocity = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0044")
    data = tRecXdata("/home/user/BachelorThesis/trecxcoefftests/tiptoe_dense/0034")

    #datamixed.plotCoefficients([1,3], "occ")
    #datalength.plotCoefficients([3], "imag")
    #datavelocity.plotCoefficients([1,3], "occ")
    #datalength.plotCoefficients([1,3], "occ")
    print(datalength.laser_params)