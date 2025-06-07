import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

class plotter:
    def __init__(self, params, time_data, rate_SFA, rateExcited_1, rateExcited_2, rateExcited_3, useTex=False):
        self.params = params
        self.time_data = time_data
        self.rate_SFA = rate_SFA
        self.rateExcited_1 = rateExcited_1
        self.rateExcited_2 = rateExcited_2
        self.rateExcited_3 = rateExcited_3
        if useTex:
            plt.rcParams['text.usetex'] = useTex
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Latin Modern Roman', 'Computer Modern']
            plt.rcParams['font.size'] = 12
    
    def matplot4(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
        
        title = f"Laser parameters: lam0: {self.params['lam0']}nm, Intensity: {self.params['intensity']:.2e}, cep: {self.params['cep']}"
        fig.suptitle(title, fontsize=16)

        # Plot 1: Rate SFA vs Rate Excited (3 states)
        ax1.plot(self.time_data, np.real(self.rate_SFA), label='Rate SFA', color='green', linewidth=1)
        ax1.plot(self.time_data, np.real(self.rateExcited_1), label='Rate Excited (3 states)', color='red', linewidth=1)
        ax1.set_xlabel('Time (a.u.)')
        ax1.set_ylabel('Ionization Rate')
        ax1.set_title('SFA vs excited rate (3 states)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([-100, 100])
        
        # Plot 2: Rate SFA vs Rate Excited (1 state)
        ax2.plot(self.time_data, np.real(self.rate_SFA), label='Rate SFA', color='green', linewidth=1)
        ax2.plot(self.time_data, np.real(self.rateExcited_2), label='Rate Excited (1 state, abs(c0)=1)', color='blue', linewidth=1)
        ax2.set_xlabel('Time (a.u.)')
        ax2.set_ylabel('Ionization Rate')
        ax2.set_title('SFA vs excited rate (1 state, c_n=0, abs(c_0)=1)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([-100, 100])
        
        # Plot 3: Rate SFA vs Rate Excited c0=1
        ax3.plot(self.time_data, np.real(self.rate_SFA), label='Rate SFA', color='green', linewidth=1)
        ax3.plot(self.time_data, np.real(self.rateExcited_3), label='Rate Excited (abs(c0)=1)', color='orange', linewidth=1)
        ax3.set_xlabel('Time (a.u.)')
        ax3.set_ylabel('Ionization Rate')
        ax3.set_title('SFA vs excited rate (3 states, only abs(c_0)=1)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([-100, 100])
        
        # Plot 4: All rates comparison
        ax4.plot(self.time_data, np.real(self.rate_SFA), label='Rate SFA', color='green', linewidth=1)
        ax4.plot(self.time_data, np.real(self.rateExcited_1), label='Rate Excited (3 states)', color='red', linewidth=1)
        ax4.plot(self.time_data, np.real(self.rateExcited_2), label='Rate Excited (1 state)', color='blue', linewidth=1)
        ax4.plot(self.time_data, np.real(self.rateExcited_3), label='Rate Excited (abs(c0)=1)', color='orange', linewidth=1)
        ax4.set_xlabel('Time (a.u.)')
        ax4.set_ylabel('Ionization Rate')
        ax4.set_title('All Rates Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([-100, 100])
        
        plt.tight_layout()

        pdf_filename = f"/home/user/BachelorThesis/Bachelor-thesis/ionModel/python/plots/rate4_{self.params['lam0']}_{self.params['intensity']:.2e}_onlystark.pdf"
        if not os.path.exists(pdf_filename):
            with PdfPages(pdf_filename) as pdf:
                pdf.savefig(fig)
        else:
            print(f"File {pdf_filename} already exists - skipping save")
        plt.show()
        plt.close()