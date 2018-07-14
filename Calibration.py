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


### Parameters ##################################

#fname = "X_50Hz_100nm_2018_07_06_17_23_48.tdms"
fname = 'Y_50Hz_100nm_2018_07_06_17_26_19.tdms'
#fname = 'White_Noise_X_2018_07_13_18_02_33.tdms'
#fname = 'White_Noise_Y_2018_07_13_18_05_55.tdms'

beta_x = 209.5
beta_y = 120.5

f_sample = 10000                    # Sampling rate
f_lowpass = 5000
dt = 1/f_sample                     # Time interval during sampling
t_total = 100                       # Total time in sec
N_total = int(f_sample * t_total)    # Total number of data

t_block = 1                    # Time for one block in sec
N_block = int(f_sample * t_block)    # Num of data in a block
df = 1/t_block                  # Freq interval for a block
N_avg = int(t_total / t_block)       # Num of blocks for averaging

# PZT 
f_drive = 50
A_drive = 100
N_drive = int(f_sample/f_drive)
PZT_nm2V = [5000, 5000, 3000]

# Constants
pi = 3.141592
kT = 4.1 # pN nm
R = 500 # Bead radius (nm)
rho = 1e-21 # Density of water (and bead) (pN s^2/nm^4)
nu = 1e12 # Kinematic viscosity of water (nm^2/s)
gamma_0 = 6.*pi*rho*nu*R # Zero frequency Stokes drag coefficent (pN s/nm)
D_0 = kT/gamma_0
mHI = 2*pi*rho*R**3 # Hydrodynamic mass of bead:
fv = nu/(pi*R**2) # Hydrodynamic frequency:
fm = gamma_0/(2.*pi*mHI) # Frequency for inertial relaxation:

###############################################

def running_mean(x, N=1000):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def sine(t, A, t0, b):
    return A * np.sin(2*pi*f_drive*(t-t0)) + b

# Power spectrum excluding hydrodynamics:
def P_0(f, D, fc):
    return D/(2*pi**2)/(fc**2 + f**2)

def P_1(f, D1, fc1, D2, fc2):
    return D1/(2.*pi**2)/(fc1**2 + f**2) + D2/(2.*pi**2)/(fc2**2 + f**2)

class Data(object):
    def __init__(self):
        pass
        

    def read(self):
        # File information below
        tdms_file = TdmsFile(fname) # Reads a tdms file.
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
            pzt = self.ch[i+8,]
            pzt = PZT_nm2V[i] * (pzt - np.mean(pzt))  
            self.ch[i+8,] = pzt 

        print('Channel number = %d' % len(channels))
        print('Channel names = %s' % self.channel_name) 
        print('Sampling frequency = %d [Hz]' % (f_sample))
        print('Lowpass filter frequency = %d [Hz]' % (f_lowpass))
        print('Total time = %d [s]' % (t_total))
        print('Block time = %d [s]' % (t_block))
        print('Oscillation frequency = %d [Hz]' % (f_drive))
        print('Oscillation amplitude = %d [nm]\n' % (A_drive))
        
      
    def analyze(self):
        # PZT fit > determine A, f_drive, Axis
        if max(self.ch[8,]) > max(self.ch[9,]):
            self.os_axis = 0
        else:
            self.os_axis = 1
        
        t = dt * np.arange(N_block)
        p_pzt, cov_pzt = curve_fit(sine, t, self.ch[self.os_axis+8,][:len(t)],  p0=[A_drive, 0, 0])
        self.pzt_A = p_pzt[0]
        self.pzt_t0 = p_pzt[1]
        self.pzt_b = p_pzt[2]

        # Get PSD
        self.PSD_mean = np.zeros((1, int(N_block/2)-1)) 
        self.PSD_sem = np.zeros((1, int(N_block/2)-1))
        
        x = self.ch[self.os_axis].reshape((N_avg, N_block))
        psd = np.zeros((N_avg, int(N_block/2)-1))
            
        for j in range(N_avg): # per block
            psd0 = np.abs(dt*np.fft.fft(x[j]))**2/t_block
            psd[j] = psd0[1:int(N_block/2)]
            
        self.PSD_mean = np.mean(psd, axis=0)
        self.PSD_sem = np.std(psd, axis=0)/(N_avg)**0.5     

        # Fit PSD > Determine D_volt, fc
        f = df * np.arange(1, N_block/2)   
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

        # Determine beta, kappa_fit, gamma_fit, R_fit
        self.W_th = 0.5*self.pzt_A**2 / (1 + (self.fc2 / f_drive)**2)
        self.W_ex = df * (self.PSD_mean[f==f_drive] - P_1(f_drive, self.D1, self.fc1, self.D2, self.fc2))
        self.beta = (self.W_th / self.W_ex)**0.5
        self.kappa = 2*pi*self.fc2*kT/self.beta**2/self.D2
        self.gamma = kT/self.beta**2/self.D2
        self.R = self.gamma / (6*pi*rho*nu)

        print('A_fit = %f [nm])' % (abs(self.pzt_A)))
        print('beta = %f [nm/V]' %(self.beta))    
        print('kappa = %f [pN/nm]' %(self.kappa))
        print('R = %d [nm] (dev = %d %%)' %(self.R, 100*(self.R-R)/R))  
        print('fc = %d [Hz]' % (self.fc2))                              
        print('D = %f [nm^2/s]' % (self.D2))
        print('gamma = %f [pN s/nm]' % (self.gamma))


    def plot_fig1(self):
        n = [0, 1, 2, 8, 9, 10]
        y = ['QPD_x (V)', 'QPD_y (V)', 'QPD_s (V)', 'PZT_x (nm)', 'PZT_y (nm)', 'PZT_z (nm)']
        t = dt * np.arange(N_block)     # Time series in a block
        t_m = running_mean(t, int(N_drive/10))
        
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        for i in range(6):
            sp = fig.add_subplot(6, 1, i+1)
            ch_m = running_mean(self.ch[n[i]][:len(t)], int(N_drive/10))
            sp.plot(t_m, ch_m, 'k-')
            sp.set_xlim([min(t), max(t)])               
            sp.set_ylabel(y[i])   
            sp.grid(True)      
        sp.set_xlabel('Time (s)')     
        fig.savefig('Trace.png')
        plt.close(fig)
                                  
    def plot_fig2(self):
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        sp = fig.add_subplot(121)
        sp.plot(self.ch[0][:N_block], self.ch[1][:N_block], 'k.')
        sp.set_aspect('equal')
        sp.set_xlabel('X (V)')
        sp.set_ylabel('Y (V)')   
        sp.grid(True) 

        sp = fig.add_subplot(122)
        sp.plot(beta_x*self.ch[0][:N_block], beta_y*self.ch[1][:N_block], 'k.')
        sp.set_aspect('equal')
        sp.set_xlabel('X (nm)')
        sp.set_ylabel('Y (nm)')   
        sp.grid(True)         
        
        fig.savefig('QPD_2D.png')
        plt.close(fig)

    def plot_fig3(self):
        t = dt * np.arange(N_drive*5)
        t_m = t[::10]
        pzt_m = self.ch[self.os_axis+8][:len(t):10]
        
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  
        sp = fig.add_subplot(211)
        sp.plot(t, sine(t, self.pzt_A, self.pzt_t0, self.pzt_b), 'k')        
        sp.plot(t_m, pzt_m, 'ro')
        sp.set_ylabel('PZT fit (nm)')
        
        sp = fig.add_subplot(212)
        sp.plot(t, self.ch[self.os_axis+8][:len(t)]-sine(t, self.pzt_A, self.pzt_t0, self.pzt_b))        
        sp.set_ylabel('Residual (nm)')
        fig.savefig('PZT.png')
        plt.close(fig)        
                
    def plot_fig4(self):
        f = df * np.arange(1, N_block/2)   # Freq series of a block
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  

        sp = fig.add_subplot(131)              
        sp.loglog(f, self.PSD_mean, 'k.')
        sp.loglog(f, P_1(f, self.D1, self.fc1, self.D2, self.fc2), 'r-')
        sp.set_xlim([min(f), max(f)])         
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('Power (V^2/Hz)')
        sp.grid(True)          

        sp = fig.add_subplot(132)              
        self.r1 = self.PSD_mean / P_1(f, self.D1, self.fc1, self.D2, self.fc2)          
        sp.plot(f[f!=f_drive], self.r1[f!=f_drive] , 'k.')
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlim([min(f), max(f)])         
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('Ratio [Exp/Fit]')
        sp.grid(True)    
        
        sp = fig.add_subplot(133)
        sp.hist(self.r1[f!=f_drive])                             
                                                                                          
        fig.savefig('PSD.png')
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



