##################################################################
#
# Trap calibration with stage oscillation (by Jongmin Sung)
# Ref: Tolic-Norrelykke et al. (2006)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm
import os
import shutil


### User input ##################################

# First you need to change directory (cd) to where the file is located

# Update the file name

X1 = 'X_White_fs20kHz_flp5kHz'
X2 = 'X_White_fs20kHz_flp10kHz'
X3 = 'X_White_fs20kHz_flp20kHz'
X4 = 'X_White_fs20kHz_flp40kHz'
X5 = 'X_White_fs20kHz_flp80kHz'

Y1 = 'Y_White_fs20kHz_flp5kHz'
Y2 = 'Y_White_fs20kHz_flp10kHz'
Y3 = 'Y_White_fs20kHz_flp20kHz'
Y4 = 'Y_White_fs20kHz_flp40kHz'
Y5 = 'Y_White_fs20kHz_flp80kHz'

X = [X1, X2, X3, X4, X5]
Y = [Y1, Y2, Y3, Y4, Y5]

f_sample = 20000                              # Sampling frequency (Hz)
f_lowpass = [5000*(2**i) for i in range(5)]  # Low pass filter (Hz)
dt = 1/f_sample                               # Time interval during sampling (s)
t_total = 10                                  # Total time (s)
N_total = int(f_sample * t_total)             # Total number of data

# I use 1 sec window for PSD and do averaging of them
t_window = 0.1                                 # Time for one window in sec
N_window = int(f_sample * t_window)           # Num of data in a window
df = 1/t_window                               # Freq interval for a window
N_avg = int(t_total / t_window)               # Num of windows for averaging
start = 10

# Constants
pi = 3.141592

ch_name = ['QPD_x', 'QPD_y', 'QPD_z', 'PZT_x', 'PZT_y', 'PZT_z']

###############################################

def running_mean(x, N=1000): # Smoothening by running averaging
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def G(f, G0, fc):
    Gain = np.zeros(len(f))
    for k, fk in enumerate(f):
        Gain[k] = np.sum([G0/(1+((fk+i*f_sample)/fc)**8) for i in range(-20,20)])
    return Gain

class Data:
    def __init__(self, j, fname):
        self.fname = fname
        self.f_lp = f_lowpass[j]
        if fname[0] == 'X':
            self.axis = 'X'
        else:
            self.axis = 'Y'

        
    def read(self):
        # File information below
        tdms_file = TdmsFile(self.fname+'.tdms') # Reads a tdms file.
        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object

        self.ch = np.zeros((len(channels), N_total)) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels): 
            self.ch[i,] = channel.data[range(N_total)]
            
        if self.axis == 'X':
            self.x = self.ch[0]
        else:
            self.x = self.ch[1]

        self.x = self.x - np.mean(self.x)
      
    def analyze(self):

        # Get PSD
        self.PSD_mean = np.zeros((1, int(N_window/2)-1)) 
        self.PSD_sem = np.zeros((1, int(N_window/2)-1))
        
        x = self.x.reshape((N_avg, N_window))
        psd = np.zeros((N_avg, int(N_window/2)-1))
            
        for j in range(N_avg): # per window
            psd0 = np.abs(dt*np.fft.fft(x[j]))**2/t_window
            psd[j] = psd0[1:int(N_window/2)]
            
        PSD_mean = np.mean(psd, axis=0)
        self.PSD_mean = PSD_mean[start:]
        PSD_sd = np.std(psd, axis=0)
        self.PSD_sd = PSD_sd[start:]     

        # Fit PSD > Determine D_volt, fc
        self.f = df * np.arange(start+1, N_window/2)   
        p, cov = curve_fit(G, self.f, self.PSD_mean, p0=[max(PSD_mean), 1000]) #4995, 4961
#        p, cov = curve_fit(G, self.f, self.PSD_mean, p0=[max(PSD_mean), self.f_lp], sigma=1/self.PSD_mean) # 4966, 4920
#        p, cov = curve_fit(G, self.f, self.PSD_mean, p0=[max(PSD_mean), self.f_lp], sigma=1/self.PSD_sd) #4955, 4922
        self.G0_fit = p[0]*N_avg/(N_avg-2)
        self.fc_fit = p[1] 
        self.PSD_fit = G(self.f, self.G0_fit, self.fc_fit)
   
         
    def plot(self): # PSD
        # PSD fitting (log-log)
        # PSD (lin)
        t = dt * np.arange(N_window)     
       
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  

        sp = fig.add_subplot(221)
        x = np.linspace(min(self.x), max(self.x), 100)
        sp.hist(self.x, color='k', histtype='step', normed=True)
        sp.plot(x, norm.pdf(x, loc=0, scale=np.std(self.x)), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('X (V)')

        sp = fig.add_subplot(222)
        sp.loglog(self.f, self.PSD_mean, 'ko', ms=1)
        sp.loglog(self.f, self.PSD_fit, 'r')
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('PSD (V^2/s)')    
        sp.set_title('fc = %d, fc_fit = %d (error = %.2f %%)' %(self.f_lp, self.fc_fit, 100*(self.fc_fit-self.f_lp)/self.f_lp))
           

        sp = fig.add_subplot(223)
        self.PSD_ratio = self.PSD_mean/self.PSD_fit
        sp.plot(self.f, self.PSD_ratio, 'ko', ms=1)
        sp.axhline(y=1, color='r', linewidth=1)   
        sp.set_xlabel('Frequency (Hz)')                      
        sp.set_ylabel('Data/Fit')    
        
        sp = fig.add_subplot(224)
        x = np.linspace(min(self.PSD_ratio), max(self.PSD_ratio), 100)
        sp.hist(self.PSD_ratio, color='k', histtype='step', normed=True)
        sp.plot(x, norm.pdf(x, loc=1, scale=np.std(self.PSD_ratio)), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Data/Fit')



        fig.savefig(self.fname)
        plt.close(fig)
                             

def main():
    for ch in [X, Y]:
        for j, fname in enumerate(ch):
            print(fname)
            data = Data(j, fname)
            data.read()
            data.analyze()
            data.plot()




if __name__ == "__main__":
    main()



