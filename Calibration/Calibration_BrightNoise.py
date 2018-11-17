##################################################################
#
# Trap calibration with stage oscillation (by Jongmin Sung)
# Ref: Tolic-Norrelykke et al. (2006)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import os
import shutil


### User input ##################################

# First you need to change directory (cd) to where the file is located

# Update the file name


files = ['Dark_Power000',
         'Bright_Power010',
         'Bright_Power020',
         'Bright_Power030',
         'Bright_Power040',
         'Bright_Power050',
         'Bright_Power060',
         'Bright_Power070',
         'Bright_Power080',
         'Bright_Power090',
         'Bright_Power099',
         'Bright_Power100']        


f_sample = 20000                              # Sampling frequency (Hz)
dt = 1/f_sample                               # Time interval during sampling (s)
t_total = 100                                  # Total time (s)
N_total = int(f_sample * t_total)             # Total number of data

# I use 1 sec window for PSD and do averaging of them
t_window = 0.1                                 # Time for one window in sec
N_window = int(f_sample * t_window)           # Num of data in a window
df = 1/t_window                               # Freq interval for a window
N_avg = int(t_total / t_window)               # Num of windows for averaging


###############################################

class Data:
    def __init__(self, fname, power):
        self.fname = fname
        self.power = power
        
    def read(self):
        # File information below
        tdms_file = TdmsFile(self.fname+'.tdms') # Reads a tdms file.
        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object

        self.ch = np.zeros((len(channels), N_total)) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels): 
            self.ch[i,] = channel.data[range(N_total)]
            
        self.x = self.ch[0] - np.mean(self.ch[0])
        self.y = self.ch[1] - np.mean(self.ch[1])
        self.s = self.ch[2]         
      
    def analyze(self):
        x = self.x.reshape((N_avg, N_window))
        y = self.y.reshape((N_avg, N_window))
        s = self.s.reshape((N_avg, N_window))

        PSD_X = np.zeros((N_avg, int(N_window/2)-1))
        PSD_Y = np.zeros((N_avg, int(N_window/2)-1))
        PSD_S = np.zeros((N_avg, int(N_window/2)-1))        
        PSD_XY = np.zeros((N_avg, int(N_window/2)-1))


        for j in range(N_avg): # per window
            PSD_X0 = np.abs(fft(x[j]))**2/t_window
            PSD_Y0 = np.abs(fft(y[j]))**2/t_window
            PSD_S0 = np.abs(fft(s[j]))**2/t_window
            PSD_XY0 = fft(x[j])*np.conj(fft(y[j]))/t_window
            PSD_XY0 = PSD_XY0/(PSD_X0*PSD_Y0)**0.5
                 
            PSD_X[j] = PSD_X0[1:int(N_window/2)]
            PSD_Y[j] = PSD_Y0[1:int(N_window/2)]
            PSD_S[j] = PSD_S0[1:int(N_window/2)]
            PSD_XY[j] = PSD_XY0[1:int(N_window/2)]

        self.PSD_X = np.mean(PSD_X, axis=0)       
        self.PSD_Y = np.mean(PSD_Y, axis=0) 
        self.PSD_S = np.mean(PSD_S, axis=0) 
        self.PSD_XY = np.mean(PSD_XY, axis=0) 
        self.f = df * np.arange(1, N_window/2)   
   
        
    def plot(self): # PSD
        # PSD fitting (log-log)
        # PSD (lin)
        t = dt * np.arange(N_window)     
       
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  

        sp = fig.add_subplot(221)
        sp.loglog(self.f, self.PSD_X, 'k', lw=1)    
#        sp.set_ylim([1e-12, 5e-9])
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('PSD_X (V^2/s)')    

        sp = fig.add_subplot(222)
        sp.loglog(self.f, self.PSD_Y, 'k', lw=1)    
#        sp.set_ylim([1e-12, 5e-9])
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('PSD_Y (V^2/s)')    

        sp = fig.add_subplot(223)
        sp.plot(self.f, self.PSD_XY, 'k', lw=1)    
#        sp.set_ylim([1e-12, 5e-9])
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('PSD_XY')    

        sp = fig.add_subplot(224)
        sp.loglog(self.f, self.PSD_S, 'k', lw=1)    
#        sp.set_ylim([1e-12, 5e-9])
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('PSD_S (V^2/s)')   
        sp.set_title('Trap power = %d %%' %(self.power)) 
                       
        fig.savefig(self.fname)
        plt.close(fig)
                             

def main():
    for fname in files:
        print(fname)
        power = int(fname[-3:])
        data = Data(fname, power)
        data.read()
        data.analyze()
        data.plot()




if __name__ == "__main__":
    main()



