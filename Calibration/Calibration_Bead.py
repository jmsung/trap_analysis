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
import matplotlib.gridspec as gridspec
from nptdms import TdmsFile
from scipy.optimize import curve_fit, fmin, minimize
from scipy.stats import norm, expon, gamma, pearsonr
import os
import shutil

### User input ##################################

# First you need to change directory (cd) to where the file is located

# Update the file name
files = ['X_fd50Hz_Ad50nm_Power020',
         'X_fd50Hz_Ad50nm_Power040',
         'X_fd50Hz_Ad50nm_Power060',         
         'X_fd50Hz_Ad50nm_Power080',         
         'X_fd50Hz_Ad50nm_Power100',         
         'Y_fd50Hz_Ad50nm_Power020',         
         'Y_fd50Hz_Ad50nm_Power040',
         'Y_fd50Hz_Ad50nm_Power060',         
         'Y_fd50Hz_Ad50nm_Power080',         
         'Y_fd50Hz_Ad50nm_Power100']         
  
#files = ['X_H1000nm_fd50Hz_Ad50nm_Power100',                 
#         'Y_H1000nm_fd50Hz_Ad50nm_Power100']    
                       
ch_name = ['QPD_x', 'QPD_y', 'QPD_z', 'PZT_x', 'PZT_y', 'PZT_z']

f_sample = 20000                # Sampling frequency
f_lowpass = 20000               # Anti-aliasing low pass filter (f_Nyq = f_sample/2)
dt = 1/f_sample                 # Time interval during sampling
t_total = 10                   # Total time in sec
N_total = int(f_sample * t_total)    # Total number of data
time = dt*np.arange(N_total)

# Windowing
t_window = 0.1                  # Time for one window in sec
N_window = int(f_sample * t_window)    # Num of data in a window
df = 1/t_window                  # Freq interval for a window
N_avg = int(t_total / t_window)       # Num of windows for averaging

# PZT 
f_drive = 50                   # Oscillation frequency
A_drive = 50                  # Oscillation amplitude
N_drive = int(f_sample/f_drive) # Number of data in one oscillation
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor

# Constants
pi = 3.141592
kT = 4.1    # Thermal energy (pN * nm)
R = 430     # Bead radius, expected according to the spec (nm)  
H = 1000 * 0.81
L = H + R
rho = 1e-21 # Density of water (and bead) (pN s^2/nm^4)
nu = 1e12   # Kinematic viscosity of water (nm^2/s)
gamma_0 = 6.*pi*rho*nu*R # Zero frequency Stokes drag coefficent (pN s/nm)
m0 = 4.*pi*rho*R**3/3. # Mass of bead:
mHI = 2*pi*rho*R**3 # Hydrodynamic mass of bead:
fv = nu/(pi*R**2) # Hydrodynamic frequency:
fm = gamma_0/(2.*pi*mHI) # Frequency for inertial relaxation (hydrodynamic):
fm0 = gamma_0/(2.*pi*m0) # Frequency for inertial relaxation (inertial):

###############################################

def running_mean(x, N=1000): # Smoothening by running averaging
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def sine(t, A, t0, b): # Sine function
    return A * np.sin(2*pi*f_drive*(t-t0)) + b

def LP(f, a_diode, f_diode):
    return a_diode**2 + (1-a_diode**2)/(1+(f/f_diode)**2) 

def P_0(f, D1, fc1, D2, fc2):
    f = abs(f)
    return D1/(2*pi**2)/(fc1**2 + f**2) + D2/(2.*pi**2)/(fc2**2 + f**2)

def PSD(f, D1, fc1, D2, fc2, a_diode, f_diode):
    return np.sum([LP(f+i*f_sample, a_diode, f_diode)*P_0(f+i*f_sample, D1, fc1, D2, fc2) for i in range(-1,2)], axis=0)

def P_H(f, D1, fc1, D2, fc2):
    f = abs(f)
    P_noise = D1/(2*pi**2)/(fc1**2 + f**2)
    Re_g = 1 + (f/fv)**0.5 - 3*R/(16*L) + 3*R/(4*L)*np.exp(-2*L/R*(f/fv)**0.5)*np.cos(2*L/R*(f/fv)**0.5)
    Im_g = -(f/fv)**0.5 + 3*R/(4*L)*np.exp(-2*L/R*(f/fv)**0.5)*np.sin(2*L/R*(f/fv)**0.5)
    P_h = (D2/(2*pi**2)*Re_g/((fc2 + f*Im_g - f**2/fm)**2 + (f*Re_g)**2))
    return  P_noise + P_h 

def PSD_H(f, D1, fc1, D2, fc2, a_diode, f_diode):
    return np.sum([LP(f+i*f_sample, a_diode, f_diode)*P_H(f+i*f_sample, D1, fc1, D2, fc2) for i in range(-1,2)], axis=0)

def ratio_stokes(f):
    return 1 + (1-1j)*(f/fv)**0.5 - 1j*2/9*f/fv
    
def W_H(A, fc):
    ra=ratio_stokes(f_drive)
    return A**2*abs(ra)**2./(4 * (fc/f_drive + np.imag(ra) - f_drive/fm0)**2 + np.real(ra)**2)

def W_th(A, fc):
    return A**2 / (1 + (fc / f_drive)**2) / 4     

def LL(p, Pf, f):
    return (sum(np.log(PSD(f, p[0], p[1], p[2], p[3], p[4], p[5]))
               +Pf/PSD(f, p[0], p[1], p[2], p[3], p[4], p[5])))

def LL_H(p, Pf, f):
    return (sum(np.log(PSD_H(f, p[0], p[1], p[2], p[3], p[4], p[5]))
               +Pf/PSD_H(f, p[0], p[1], p[2], p[3], p[4], p[5])))

class Data:
    def __init__(self, fname, axis, power):
        self.fname = fname
        self.axis = axis
        self.power = power

        # Make a directory to save the results
        self.data_path = os.getcwd()
        self.dir = os.path.join(self.data_path, fname)

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
            os.makedirs(self.dir)
        else:
            os.makedirs(self.dir)
                              
    def read(self):
        # File information below
        tdms_file = TdmsFile(self.fname+'.tdms') # Reads a tdms file.
        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object

        self.ch = np.zeros((len(channels), N_total)) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels): 
            self.ch[i,] = channel.data[range(N_total)]

        self.QPDx = self.ch[0] - np.mean(self.ch[0])
        self.QPDy = self.ch[1] - np.mean(self.ch[1])
        self.QPDs = self.ch[2]      
        self.PZTx = (self.ch[3] - np.mean(self.ch[3])) * PZT_nm2V[0]  
        self.PZTy = (self.ch[4] - np.mean(self.ch[4])) * PZT_nm2V[1]      
        
        self.QPDx = self.QPDx/self.QPDs         
        self.QPDy = self.QPDy/self.QPDs  

                                                                                                                                            
    def analyze(self):

        # Windowing  
        x = self.QPDx.reshape((N_avg, N_window))
        y = self.QPDy.reshape((N_avg, N_window))

        self.PSD_X0 = np.zeros((N_avg, int(N_window/2)-1))
        self.PSD_Y0 = np.zeros((N_avg, int(N_window/2)-1))      
        PSD_XY = np.zeros((N_avg, int(N_window/2)-1))

        for j in range(N_avg): # per window
            PSD_X0 = np.abs(dt*fft(x[j]))**2/t_window
            PSD_Y0 = np.abs(dt*fft(y[j]))**2/t_window
            PSD_XY0 = (dt*fft(x[j])*np.conj(dt*fft(y[j]))).real/t_window
            PSD_XY0 = PSD_XY0/(PSD_X0*PSD_Y0)**0.5
                 
            self.PSD_X0[j] = PSD_X0[1:int(N_window/2)]
            self.PSD_Y0[j] = PSD_Y0[1:int(N_window/2)]
            PSD_XY[j] = PSD_XY0[1:int(N_window/2)]

        self.PSD_X = np.mean(self.PSD_X0, axis=0)       
        self.PSD_Y = np.mean(self.PSD_Y0, axis=0) 
        self.PSD_XS = np.std(self.PSD_X0, axis=0)
        self.PSD_YS = np.std(self.PSD_Y0, axis=0)        
        self.PSD_XY = np.mean(PSD_XY, axis=0) 
        self.f = df * np.arange(1, N_window/2)    

        # PSD_X ################################################################
        f = self.f[self.f!=f_drive]  
        Pf = self.PSD_X[self.f!=f_drive]

        # LSQ fitting  
        fc1 = self.power*2.1 + 19
        D1 = 8e-2  
        fc2 = self.power*46 - 367
        D2 = 7e-1                   
        a_diode = 0.8
        fc_diode = 1e+4
        p0 = [D1, fc1, D2, fc2, a_diode, fc_diode]
        lb = (D1/5, fc1/5, D2/5, fc2/5, 0, fc_diode/5)
        ub = (D1*5, fc1*5, D2*5, fc2*5, 1, fc_diode*5)
        p, cov = curve_fit(PSD, f, Pf, p0, sigma=self.PSD_XS[self.f!=f_drive], bounds = (lb, ub))

        # MLE fitting 
        p0 = [p[0], p[1], p[2], p[3], p[4], p[5]]
        bnds = ((p[0]/5, p[0]*5), (p[1]/5, p[1]*5), (p[2]/5, p[2]*5), (p[3]/5, p[3]*5), (0, 1), (p[5]/5, p[5]*5))
        result = minimize(LL, x0=p0, args=(Pf,f), bounds = bnds)  
        p = result.x
         
        self.D_X = p[2]
        self.fc_X = p[3] 

        dx = self.D_X/1000
        self.dD_X = (abs(LL([p[0], p[1], p[2]+dx, p[3], p[4], p[5]], Pf, f) 
                       + LL([p[0], p[1], p[2]-dx, p[3], p[4], p[5]], Pf, f) 
                     - 2*LL([p[0], p[1], p[2],    p[3], p[4], p[5]], Pf, f))/dx**2)**-0.5

        dx = self.fc_X/1000
        self.dfc_X = (abs(LL([p[0], p[1], p[2], p[3]+dx, p[4], p[5]], Pf, f) 
                        + LL([p[0], p[1], p[2], p[3]-dx, p[4], p[5]], Pf, f) 
                      - 2*LL([p[0], p[1], p[2], p[3],    p[4], p[5]], Pf, f))/dx**2)**-0.5

        self.PSD_X_fit = PSD(self.f, p[0], p[1], p[2], p[3], p[4], p[5])
        self.residual_X = self.PSD_X[self.f!=f_drive]/self.PSD_X_fit[self.f!=f_drive]

        self.residual_X0 = np.zeros((N_avg, sum(self.f!=f_drive)))
        for j in range(N_avg): # per window
            self.residual_X0[j] = self.PSD_X0[j][self.f!=f_drive]/self.PSD_X_fit[self.f!=f_drive]
  
        # PSD_Y ################################################################                   
        Pf = self.PSD_Y[self.f!=f_drive]
        
        # LSQ fitting         
        fc1 = self.power*2.6 + 12
        D1 = 5e-3
        fc2 = self.power*26.4 + 88
        D2 = 1.5               
        a_diode = 0.8
        fc_diode = 1.2e+4
        p0 = [D1, fc1, D2, fc2, a_diode, fc_diode]
        lb = (D1/5, fc1/5, D2/5, fc2/5, 0, fc_diode/5)
        ub = (D1*5, fc1*5, D2*5, fc2*5, 1, fc_diode*5) 
        p, cov = curve_fit(PSD, f, Pf, p0, sigma=self.PSD_YS[self.f!=f_drive], bounds = (lb, ub))

        # MLE fitting 
        p0 = [p[0], p[1], p[2], p[3], p[4], p[5]]
        bnds = ((p[0]/5, p[0]*5), (p[1]/5, p[1]*5), (p[2]/5, p[2]*5), (p[3]/5, p[3]*5), (0, 1), (p[5]/5, p[5]*5))
        result = minimize(LL, x0=p0, args=(Pf,f), bounds = bnds)  
        p = result.x

        self.D_Y = p[2]
        self.fc_Y = p[3]   

        dx = self.D_Y/1000
        self.dD_Y = (abs(LL([p[0], p[1], p[2]+dx, p[3], p[4], p[5]], Pf, f) 
                       + LL([p[0], p[1], p[2]-dx, p[3], p[4], p[5]], Pf, f) 
                     - 2*LL([p[0], p[1], p[2],    p[3], p[4], p[5]], Pf, f))/dx**2)**-0.5

        dx = self.fc_Y/1000
        self.dfc_Y = (abs(LL([p[0], p[1], p[2], p[3]+dx, p[4], p[5]], Pf, f) 
                        + LL([p[0], p[1], p[2], p[3]-dx, p[4], p[5]], Pf, f) 
                      - 2*LL([p[0], p[1], p[2], p[3],    p[4], p[5]], Pf, f))/dx**2)**-0.5
                             
        self.PSD_Y_fit = PSD(self.f, p[0], p[1], p[2], p[3], p[4], p[5])
        self.residual_Y = self.PSD_Y[self.f!=f_drive]/self.PSD_Y_fit[self.f!=f_drive]
        
        self.residual_Y0 = np.zeros((N_avg, sum(self.f!=f_drive)))
        for j in range(N_avg): # per window
            self.residual_Y0[j] = self.PSD_Y0[j][self.f!=f_drive]/self.PSD_Y_fit[self.f!=f_drive]
            
        # Determine beta, kappa_fit, gamma_fit, R_fit
        if self.axis == 'X': 
            p, cov = curve_fit(sine, time, self.PZTx,  p0=[A_drive, f_drive, 0])
            self.PZT_A = abs(p[0])   
            self.fc = self.fc_X
            self.dfc = self.dfc_X
            self.D_V = self.D_X    
            self.dD_V = self.dD_X       
            self.W_ex = df * (self.PSD_X[self.f==f_drive] - self.PSD_X_fit[self.f==f_drive])             
        else:
            p, cov = curve_fit(sine, time, self.PZTy,  p0=[A_drive, f_drive, 0])
            self.PZT_A = abs(p[0])            
            self.fc = self.fc_Y  
            self.dfc = self.dfc_Y                        
            self.D_V = self.D_Y
            self.dD_V = self.dD_Y            
            self.W_ex = df * (self.PSD_Y[self.f==f_drive] - self.PSD_Y_fit[self.f==f_drive])          
 
        self.W_th = W_th(self.PZT_A, self.fc)      
        self.beta = (self.W_th / self.W_ex)**0.5 
        self.D = self.D_V * self.beta**2    
        self.kappa = 2*pi*self.fc*kT/self.D
        self.gamma = kT/self.D
        self.ratio_stoke = self.gamma / gamma_0

        self.dW_th = abs(W_th(self.PZT_A, self.fc+self.dfc) - W_th(self.PZT_A, self.fc-self.dfc))/2
        self.dbeta = abs(((self.W_th+self.dW_th)/self.W_ex)**0.5 - ((self.W_th-self.dW_th)/self.W_ex)**0.5)/2         
        self.dD = abs((self.D_V+self.dD_V)*(self.beta+self.dbeta)**2 - (self.D_V-self.dD_V) * (self.beta-self.dbeta)**2)/2    
        self.dkappa = 2*pi*abs((self.fc+self.dfc)*kT/(self.D-self.dD) - (self.fc-self.dfc)*kT/(self.D+self.dD))/2
        self.dgamma = abs(kT/(self.D-self.dD) - kT/(self.D+self.dD))/2

#        print('A_fit = %.1f [nm])' % (abs(self.PZT_A)))
        print('beta = %.1f +/- %.1f [nm/V]' %(self.beta, self.dbeta))  
#        print('fc = %.1f +/- %.1f [Hz]' % (self.fc, self.dfc))
        print('kappa = %.3f +/- %.3f [pN/nm] \n' %(self.kappa, self.dkappa))
#        print('Stoke ratio = %.1f \n' %(self.ratio_stoke)) # Due to viscosity and surface hydrodynamic effect  
             
    def plot_fig1(self):
        fig = plt.figure(1, figsize = (20, 10), dpi=300) 
        grid = gridspec.GridSpec(ncols=2, nrows=2)
        
        # X (V)
        sp = fig.add_subplot(grid[0,0])
        sp.hist(self.QPDx, bins=100, color='k', histtype='step', normed=True)
        x = np.linspace(min(self.QPDx), max(self.QPDx), 100)
        sp.plot(x, norm.pdf(x, loc=0, scale=np.std(self.QPDx)), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('X (V)')
        sp.set_title('Width = %f (V)' %(np.std(self.QPDx))) 

        # Y (V)
        sp = fig.add_subplot(grid[1,0])
        y = np.linspace(min(self.QPDy), max(self.QPDy), 100)
        sp.hist(self.QPDy, bins=100, color='k', histtype='step', normed=True)
        sp.plot(y, norm.pdf(y, loc=0, scale=np.std(self.QPDy)), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Y (V)')   
        sp.set_title('Width = %f (V)' %(np.std(self.QPDy)))             
                        
        # X-Y
        sp = fig.add_subplot(grid[:,1])
        heatmap, xedges, yedges = np.histogram2d(self.QPDx, self.QPDy, bins=100)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        sp.imshow(heatmap.T, extent=extent, origin='lower')
#        plt.axis('equal')
        sp.set_aspect('equal')
        sp.set_xlabel('X (V)')
        sp.set_ylabel('Y (V)')  
        sp.set_title('Correlation = %f' %(pearsonr(self.QPDx, self.QPDy)[0]))

        fig.savefig(os.path.join(self.dir, 'Fig1_XY.png'))
        plt.close(fig)    


    def plot_fig2(self):
        fig = plt.figure(2, figsize = (20, 10), dpi=300)  

        # PSD_X
        sp = fig.add_subplot(221)
        sp.loglog(self.f, self.PSD_X, 'k', ms=1)
        sp.loglog(self.f, self.PSD_X_fit, 'r', lw=2)       
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('PSD_X (V^2 Hz)')
        if self.axis == 'X': 
            sp.set_title('beta = %d +/- %d [nm/V], k = %.3f +/- %.3f [pN/nm], fc = %d +/- %d [Hz]' 
                        %(self.beta, self.dbeta, self.kappa, self.dkappa, self.fc_X, self.dfc_X))
        else:
            sp.set_title('fc = %d +/- %d (Hz)' %(self.fc_X, self.dfc_X))            

        # Residual_X
        sp = fig.add_subplot(222)
        sp.plot(self.f[self.f!=f_drive], self.residual_X, 'k', ms=1)
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('PSD_X (Exp/Fit)')

        sp = fig.add_subplot(223)        
        sp.hist(self.residual_X, bins=20, color='k', histtype='step', normed=True)
        x = np.linspace(min(self.residual_X), max(self.residual_X), 100)
        sp.plot(x, norm.pdf(x, loc=1, scale=1/(N_avg)**0.5), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('PSD_X (Exp/Fit)')

        sp = fig.add_subplot(224)     
        x = np.linspace(0, max(self.residual_X0.flatten()), 100) 
        sp.hist(self.residual_X0.flatten(), bins=20, color='k', histtype='step', normed=True)
        sp.plot(x, expon.pdf(x), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('PSD_X0 (Exp/Fit)')
        
        fig.savefig(os.path.join(self.dir, 'Fig2_PSD_X.png'))
        plt.close(fig)                


    def plot_fig3(self):
        fig = plt.figure(3, figsize = (20, 10), dpi=300)  

        # PSD_Y
        sp = fig.add_subplot(221)
        sp.loglog(self.f, self.PSD_Y, 'k', ms=1)
        sp.loglog(self.f, self.PSD_Y_fit, 'r', lw=2)
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('PSD_Y')

        if self.axis == 'Y': 
            sp.set_title('beta = %d +/- %d [nm/V], k = %.3f +/- %.3f [pN/nm], fc = %d +/- %d [Hz]' 
                        %(self.beta, self.dbeta, self.kappa, self.dkappa, self.fc_Y, self.dfc_Y))
        else:
            sp.set_title('fc = %d +/- %d (Hz)' %(self.fc_Y, self.dfc_Y)) 

        # Residual_Y
        sp = fig.add_subplot(222)
        sp.plot(self.f[self.f!=f_drive], self.residual_Y, 'k', ms=1)
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('PSD_Y (Exp/Fit)')             


        sp = fig.add_subplot(223)     
        sp.hist(self.residual_Y, bins=20, color='k', histtype='step', normed=True)
        y = np.linspace(min(self.residual_Y), max(self.residual_Y), 100)
        sp.plot(y, norm.pdf(y, loc=1, scale=1/(N_avg)**0.5), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('PSD_Y (Exp/Fit)')

        sp = fig.add_subplot(224)     
        y = np.linspace(0, max(self.residual_Y0.flatten()), 100)
        sp.hist(self.residual_Y0.flatten(), bins=20, color='k', histtype='step', normed=True)
        sp.plot(y, expon.pdf(y), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('PSD_Y0 (Exp/Fit)')
                     
#        # PSD_XY
#        sp = fig.add_subplot(336)
#        sp.plot(self.f, self.PSD_XY, 'k', lw=1)
#        sp.set_xlabel('Frequency (Hz)')
#        sp.set_ylabel('PSD_XY')   
         
        fig.savefig(os.path.join(self.dir, 'Fig3_PSD_Y.png'))
#        fig.savefig(self.fname)
        plt.close(fig)                

    def plot(self):
        self.plot_fig1()                                                
        self.plot_fig2()                                                                                                                                                  
        self.plot_fig3()  


def main():
    for fname in files:
        print(fname)
        axis = fname[:1]
        power = int(fname[-3:])
        data = Data(fname, axis, power)
        data.read()
        data.analyze()
        data.plot()


if __name__ == "__main__":
    main()



