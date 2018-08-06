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
from scipy.optimize import curve_fit
from scipy.stats import norm
import os
import shutil

### User input ##################################

# First you need to change directory (cd) to where the file is located

# Update the file name
fname = 'Y_50Hz_100nm_2018_07_06_17_26_19' 

beta_x = 209.5 # This is actually the fitting result, not the user input
beta_y = 120.5 # This is actually the fitting result, not the user input

f_sample = 20000                # Sampling frequency
f_lowpass = 10000               # Anti-aliasing low pass filter (f_Nyq = f_sample/2)
dt = 1/f_sample                 # Time interval during sampling
t_total = 100                   # Total time in sec
N_total = int(f_sample * t_total)    # Total number of data

# I use 1 sec window for PSD and do averaging of them
t_window = 1                    # Time for one window in sec
N_window = int(f_sample * t_window)    # Num of data in a window
df = 1/t_window                  # Freq interval for a window
N_avg = int(t_total / t_window)       # Num of windows for averaging

# PZT 
f_drive = 50                   # Oscillation frequency
A_drive = 100                  # Oscillation amplitude
N_drive = int(f_sample/f_drive) # Number of data in one oscillation
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor

# Constants
pi = 3.141592
kT = 4.1    # Thermal energy (pN * nm)
R = 500     # Bead radius, expected according to the spec (nm)   
rho = 1e-21 # Density of water (and bead) (pN s^2/nm^4)
nu = 1e12   # Kinematic viscosity of water (nm^2/s)
gamma_0 = 6.*pi*rho*nu*R # Zero frequency Stokes drag coefficent (pN s/nm)
D_0 = kT/gamma_0    # Diffusion constant, expected
mHI = 2*pi*rho*R**3 # Hydrodynamic mass of bead:
fv = nu/(pi*R**2)   # Hydrodynamic frequency:
fm = gamma_0/(2.*pi*mHI) # Frequency for inertial relaxation:

ch_name = ['QPD_x', 'QPD_y', 'QPD_z', 'PZT_x', 'PZT_y', 'PZT_z']

###############################################

def running_mean(x, N=1000): # Smoothening by running averaging
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def sine(t, A, t0, b): # Sine function
    return A * np.sin(2*pi*f_drive*(t-t0)) + b

# Power spectrum excluding hydrodynamics:
def P_0(f, D, fc):
    return D/(2*pi**2)/(fc**2 + f**2)

# Power spectrum as sum of two Lorentzian. I use this one becasue it fits much better
# I use higher freq fc for the trap stiffness. Lower fc might be due to low freq noise. 
def P_1(f, D1, fc1, D2, fc2):
    return D1/(2.*pi**2)/(fc1**2 + f**2) + D2/(2.*pi**2)/(fc2**2 + f**2)

class Data(object):
    def __init__(self):
        pass
        

    def read(self):
        # File information below
        tdms_file = TdmsFile(fname+'.tdms') # Reads a tdms file.
        root_object = tdms_file.object() # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object
        self.channel_name = [str(channels[i].channel) for i in range(0, len(channels))]

        self.ch = np.zeros((len(channels), N_total)) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels): 
            self.ch[i,] = channel.data[range(N_total)]
            
        # Convert PZT unit from V to nm    
        for i in range(3):
            pzt = self.ch[i+3,]
            pzt = PZT_nm2V[i] * (pzt - np.mean(pzt))  
            self.ch[i+3,] = pzt 

        print('Channel number = %d' % len(channels))
        print('Channel names = %s' % self.channel_name) 
        print('Sampling frequency = %d [Hz]' % (f_sample))
        print('Lowpass filter frequency = %d [Hz]' % (f_lowpass))
        print('Total time = %d [s]' % (t_total))
        print('window time = %d [s]' % (t_window))
        print('Oscillation frequency = %d [Hz]' % (f_drive))
        print('Oscillation amplitude = %d [nm]\n' % (A_drive))

        # Make a directory to save the results
        self.data_path = os.getcwd()
        self.dir = self.data_path+'\\%s' %(fname)

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
            os.makedirs(self.dir)
        else:
            os.makedirs(self.dir)
      
    def analyze(self):

        # Find the oscillation axis
        if max(self.ch[3,]) > max(self.ch[4,]): 
            self.os_axis = 0 # X-axis
        else:
            self.os_axis = 1 # Y-axis

        # PZT fit > determine A, f_drive, Axis        
        t = dt * np.arange(N_window)
        p_pzt, cov_pzt = curve_fit(sine, t, self.ch[self.os_axis+3,][:len(t)],  p0=[A_drive, f_drive, 0])
        self.pzt_A = p_pzt[0]
        self.pzt_t0 = p_pzt[1]
        self.pzt_b = p_pzt[2]

        # Get PSD
        self.PSD_mean = np.zeros((1, int(N_window/2)-1)) 
        self.PSD_sem = np.zeros((1, int(N_window/2)-1))
        
        x = self.ch[self.os_axis].reshape((N_avg, N_window))
        psd = np.zeros((N_avg, int(N_window/2)-1))
            
        for j in range(N_avg): # per window
            psd0 = np.abs(dt*np.fft.fft(x[j]))**2/t_window
            psd[j] = psd0[1:int(N_window/2)]
            
        self.PSD_mean = np.mean(psd, axis=0)
        self.PSD_sem = np.std(psd, axis=0)/(N_avg)**0.5     

        # Fit PSD > Determine D_volt, fc
        f = df * np.arange(1, N_window/2)   
        fc_guess = f[np.argmin(abs(self.PSD_mean - self.PSD_mean[0]/2))]
        D_guess = 2 * pi**2 * fc_guess**2 * self.PSD_mean[0]    

        # LSQ: Sum of two Lorentzian
        p_psd, cov_psd = curve_fit(P_1, f[f!=f_drive], self.PSD_mean[f!=f_drive], [2*D_guess, fc_guess/2, D_guess/2, 2*fc_guess])        
        self.D1 = abs(p_psd[0])
        self.fc1 = abs(p_psd[1])
        self.W1 = sum(P_0(f, self.D1, self.fc1))
        self.D2 = abs(p_psd[2])
        self.fc2 = abs(p_psd[3])      
        self.W2 = sum(P_0(f, self.D2, self.fc2))  
        
        print('fc1 = %d [Hz]' % (self.fc1))                              
        print('D1 = %f [nm^2/s]' % (self.D1))
        print('fc2 = %d [Hz]' % (self.fc2))                              
        print('D2 = %f [nm^2/s]\n' % (self.D2))

        # Determine beta, kappa_fit, gamma_fit, R_fit
        self.W_th = 0.5*self.pzt_A**2 / (1 + (self.fc2 / f_drive)**2)
        self.W_ex = df * (self.PSD_mean[f==f_drive] - P_1(f_drive, self.D1, self.fc1, self.D2, self.fc2))
        self.beta = (self.W_th / self.W_ex)**0.5
        if np.isnan(self.beta):
            self.beta = 1
        self.kappa = 2*pi*self.fc2*kT/self.beta**2/self.D2
        self.gamma = kT/self.beta**2/self.D2
        self.R = self.gamma / (6*pi*rho*nu)

        print('A_fit = %f [nm])' % (abs(self.pzt_A)))
        print('beta = %f [nm/V]' %(self.beta))    
        print('kappa = %f [pN/nm]' %(self.kappa))
        print('fc = %d [Hz]' % (self.fc2))                              
        print('D = %f [nm^2/s]' % (self.D2))
        print('gamma = %f [pN s/nm]' % (self.gamma))
        print('R = %d [nm] (dev = %d %%)' %(self.R, 100*(self.R-R)/R))          


    def plot_fig1(self): # Time series 
        t = dt * np.arange(N_window)     
        t_m = running_mean(t, int(N_drive/10))
        
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        for i in range(6):
            sp = fig.add_subplot(6, 1, i+1)
            ch_m = running_mean(self.ch[i][:len(t)], int(N_drive/10))
            sp.plot(t_m, ch_m, 'k-')
            sp.set_xlim([min(t), max(t)])               
            sp.set_ylabel(ch_name[i])       
        sp.set_xlabel('Time (s)')  

        fig.savefig(self.dir + '\\Fig1_Trace.png')
        plt.close(fig)
                                  
    def plot_fig2(self): # XY
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        sp = fig.add_subplot(121)
        sp.plot(self.ch[0][:N_window], self.ch[1][:N_window], 'k.')
        sp.axis('equal')
        sp.set_aspect('equal')
        sp.set_xlabel('X (V)')
        sp.set_ylabel('Y (V)')   
        sp.set_title('2D plot (V)')

        sp = fig.add_subplot(122)
        sp.plot(beta_x*self.ch[0][:N_window], beta_y*self.ch[1][:N_window], 'k.')
        sp.axis('equal')
        sp.set_aspect('equal')
        sp.set_xlabel('X (nm)')
        sp.set_ylabel('Y (nm)')   
        sp.set_title('2D plot (nm)')    
        
        fig.savefig(self.dir + '\\Fig2_QPD2D.png')
        plt.close(fig)

    def plot_fig3(self): # PZT
        t = dt * np.arange(N_drive*10)
        t_m = t[::10]
        pzt_m = self.ch[self.os_axis+3][:len(t):10]
        
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        sp = fig.add_subplot(211)
        sp.plot(t, sine(t, self.pzt_A, self.pzt_t0, self.pzt_b), 'k')        
        sp.plot(t_m, pzt_m, 'ro')
        sp.set_xlim([min(t), max(t)])
        sp.set_ylabel('PZT (nm)')
        sp.set_title('PZT oscillation: Data (red) vs Fit (black)')
        
        sp = fig.add_subplot(212)
        sp.plot(t, self.ch[self.os_axis+3][:len(t)]-sine(t, self.pzt_A, self.pzt_t0, self.pzt_b), 'k')        
        sp.set_xlabel('Time (s)')
        sp.set_ylabel('Residual (nm)')
        sp.set_xlim([min(t), max(t)])

        fig.savefig(self.dir + '\\Fig3_PZT.png')
        plt.close(fig)        
                
    def plot_fig4(self): # PSD 
        f = df * np.arange(1, N_window/2)   # Freq series of a window
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  

        sp = fig.add_subplot(131)              
        sp.loglog(f, self.PSD_mean, 'k.')
        sp.loglog(f, P_1(f, self.D1, self.fc1, self.D2, self.fc2), 'r-')
        sp.set_xlim([min(f), max(f)])         
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('Power (V^2/Hz)')        
        sp.set_title('PSD (black) and Fit (red)')

        sp = fig.add_subplot(132)              
        self.r1 = self.PSD_mean / P_1(f, self.D1, self.fc1, self.D2, self.fc2)          
        sp.plot(f[f!=f_drive], self.r1[f!=f_drive] , 'k.')
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlim([min(f), max(f)])         
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('Ratio (exp / fit)')
        sp.set_title('Data (black) vs Theory (red)')  
        
        sp = fig.add_subplot(133)
        r = self.r1[f!=f_drive]
        sp.hist(r, bins='auto', normed=True, histtype='step', color='k')   
        x = np.arange(min(r), max(r), 0.01)
        sp.plot(x, norm.pdf(x, 1, 1/(N_avg)**0.5), 'r')  
        sp.set_xlabel('Ratio (exp / fit)')
        sp.set_ylabel('Probability density')  
        sp.set_title('Residual histogram (black) vs Theory (red)')   
                                                                                                            
        fig.savefig(self.dir + '\\Fig4_PSD.png')
        plt.close(fig)


    def plot(self):
        self.plot_fig1()
        self.plot_fig2()
        self.plot_fig3()
        self.plot_fig4()      
                

def main():
    data = Data()
    data.read()
    data.analyze()
    data.plot()


if __name__ == "__main__":
    main()



