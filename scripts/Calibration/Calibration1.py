##################################################################
#
# Trap calibration with stage oscillation (by Jongmin Sung)
# Ref: Tolic-Norrelykke et al. (2006)
#
# Consider QPD low-pass filter and Hydrodynamic interaction
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nptdms import TdmsFile
from scipy.optimize import curve_fit, minimize
from scipy.stats import norm, expon, pearsonr
import os
import shutil
import sys

### Constants ##################################
      
pi = 3.141592
kBT = 4.1    # Thermal energy (pN * nm)
rho = 1e-21 # Density of water (and bead) (pN s^2/nm^4)
nu = 1e12   # Kinematic viscosity of water (nm^2/s)

# QPD        
CH_NAME = ['QPD_x', 'QPD_y', 'QPD_z', 'PZT_x', 'PZT_y', 'PZT_z']

# PZT
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor

###############################################

def sine(t, f, A, t0, b): # Sine function
    return A * np.sin(2*pi*f*(t-t0)) + b

def QPD_lowpass(f, a_QPD, f_QPD):
    return a_QPD**2 + (1-a_QPD**2)/(1+(f/f_QPD)**2) 

def BW_lowpass(f, f_lp):
    return 1/(1 + (f/f_lp)**8)
    
#def P(f, D1, fc1, D2, fc2):
#    return D1/(2*pi**2)/(fc1**2 + f**2) + D2/(2.*pi**2)/(fc2**2 + f**2)

#def PSD(f, D1, fc1, D2, fc2, a_diode, f_diode):
#    return np.sum([QPD(f+i*f_sample, a_diode, f_diode)*LP(f+i*f_sample, f_lp)*P(f+i*f_sample, D1, fc1, D2, fc2) for i in range(-10,11)], axis=0)

def P_H(f, D1, fc1, D2, fc2, R, L, fv, fm):
    f = abs(f)
    P_noise = D1/(2*pi**2)/(fc1**2 + f**2)
    Re_g = 1 + (f/fv)**0.5 - 3*R/(16*L) + 3*R/(4*L)*np.exp(-2*L/R*(f/fv)**0.5)*np.cos(2*L/R*(f/fv)**0.5)
    Im_g = -(f/fv)**0.5 + 3*R/(4*L)*np.exp(-2*L/R*(f/fv)**0.5)*np.sin(2*L/R*(f/fv)**0.5)
    P_h = (D2/(2*pi**2)*Re_g/((fc2 + f*Im_g - f**2/fm)**2 + (f*Re_g)**2))
    return  P_noise + P_h 

def PSD_H(f, D1, fc1, D2, fc2, a_QPD, f_QPD, R, L, f_sample, f_lp, fv, fm):
    return np.sum([QPD_lowpass(f+i*f_sample, a_QPD, f_QPD)*BW_lowpass(f+i*f_sample, f_lp)*P_H(f+i*f_sample, D1, fc1, D2, fc2, R, L, fv, fm) for i in range(-10,11)], axis=0)

def ratio_stokes(f, fv):
    return 1 + (1-1j)*(f/fv)**0.5 - 1j*2/9*f/fv
    
def W_H(A, f_drive, fc, fv, fm0):
    ra=ratio_stokes(f_drive, fv)
    return A**2*abs(ra)**2./(2 * (fc/f_drive + np.imag(ra) - f_drive/fm0)**2 + np.real(ra)**2)

#def W_th(A, f_drive, fc):
#    return A**2 / (1 + (fc / f_drive)**2) / 2     

#def LL(p, Pf, f):
#    return (sum(np.log(PSD(f, p[0], p[1], p[2], p[3], p[4], p[5])) + Pf/PSD(f, p[0], p[1], p[2], p[3], p[4], p[5])))

def LL_H(p, Pf, f, R, L, f_sample, f_lp, fv, fm):
    return (sum(np.log(PSD_H(f, p[0], p[1], p[2], p[3], p[4], p[5], R, L, f_sample, f_lp, fv, fm)) + Pf/PSD_H(f, p[0], p[1], p[2], p[3], p[4], p[5], R, L, f_sample, f_lp, fv, fm)))

class Data:
    def __init__(self, fname, f_sample, f_lp, R, power, axis, fd, Ad, height):
        self.fname = fname
        self.f_sample = f_sample
        self.f_lp = f_lp
        self.R = R
        self.power = power        
        self.axis = axis
        self.fd = fd
        self.Ad = Ad
        self.height = height
        self.L = height + self.R

        self.gamma_0 = 6.*pi*rho*nu*R # Zero frequency Stokes drag coefficent (pN s/nm)
        self.m0 = 4.*pi*rho*R**3/3. # Mass of bead:
        self.mHI = 2*pi*rho*R**3 # Hydrodynamic mass of bead:
        self.fv = nu/(pi*R**2) # Hydrodynamic frequency:
        self.fm = self.gamma_0/(2.*pi*self.mHI) # Frequency for inertial relaxation (hydrodynamic):
        self.fm0 = self.gamma_0/(2.*pi*self.m0) # Frequency for inertial relaxation (inertial):

        # Total time
        self.dt = 1/f_sample                 # Time interval during sampling
        self.t_total = 100                   # Total time in sec
        self.N_total = int(f_sample * self.t_total)    # Total number of data
        self.time = self.dt*np.arange(self.N_total)

        # Windowing 
        self.t_window = 1                  # Time for one window in sec
        self.N_window = int(f_sample * self.t_window)    # Num of data in a window
        self.df = 1/self.t_window                  # Freq interval for a window
        self.N_avg = int(self.t_total / self.t_window)       # Num of windows for averaging

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

        self.ch = np.zeros((len(channels), self.N_total)) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels): 
            self.ch[i,] = channel.data[range(self.N_total)]

        self.QPDx = self.ch[0] - np.mean(self.ch[0])
        self.QPDy = self.ch[1] - np.mean(self.ch[1])
        self.QPDs = self.ch[2]      
        self.PZTx = (self.ch[3] - np.mean(self.ch[3])) * PZT_nm2V[0]  
        self.PZTy = (self.ch[4] - np.mean(self.ch[4])) * PZT_nm2V[1]      
        
        self.QPDx = self.QPDx/self.QPDs         
        self.QPDy = self.QPDy/self.QPDs  

                                                                                                                                            
    def analyze(self):

        # Write the calibration result to a file
        info = open(os.path.join(self.dir, 'Calibration.txt'), 'w')
        info.write('Filename = %s \n\n' %(self.fname))

        # Windowing  
        x = self.QPDx.reshape((self.N_avg, self.N_window))
        y = self.QPDy.reshape((self.N_avg, self.N_window))

        self.PSD_X0 = np.zeros((self.N_avg, int(self.N_window/2)-1))
        self.PSD_Y0 = np.zeros((self.N_avg, int(self.N_window/2)-1))      
        PSD_XY = np.zeros((self.N_avg, int(self.N_window/2)-1))

        for j in range(self.N_avg): # per window
            PSD_X0 = np.abs(self.dt*fft(x[j]))**2/self.t_window
            PSD_Y0 = np.abs(self.dt*fft(y[j]))**2/self.t_window
            PSD_XY0 = (self.dt*fft(x[j])*np.conj(self.dt*fft(y[j]))).real/self.t_window
            PSD_XY0 = PSD_XY0/(PSD_X0*PSD_Y0)**0.5
                 
            self.PSD_X0[j] = PSD_X0[1:int(self.N_window/2)]
            self.PSD_Y0[j] = PSD_Y0[1:int(self.N_window/2)]
            PSD_XY[j] = PSD_XY0[1:int(self.N_window/2)]

        self.PSD_X = np.mean(self.PSD_X0, axis=0)       
        self.PSD_Y = np.mean(self.PSD_Y0, axis=0) 
        self.PSD_XS = np.std(self.PSD_X0, axis=0)
        self.PSD_YS = np.std(self.PSD_Y0, axis=0)        
        self.PSD_XY = np.mean(PSD_XY, axis=0) 
        self.f = self.df * np.arange(1, self.N_window/2)    

        # PSD_X ################################################################
        f = self.f[self.f!=self.fd]  
        Pf = self.PSD_X[self.f!=self.fd]

        # MLE fitting 
        D1_X = 0.1          
        fc1_X = self.power*0.5
        D2_X = 0.5  
        fc2_X = self.power*5    
        a_QPD = 0.5
        f_QPD = 5000

        p0 = [D1_X, fc1_X, D2_X, fc2_X, a_QPD, f_QPD]
        bnds = ((D1_X/10, D1_X*10), (fc1_X/10, fc1_X*10), (D2_X/10, D2_X*10), (fc2_X/10, fc2_X*10), (0, 1), (f_QPD/10, f_QPD*10))

        result = minimize(LL_H, x0=p0, args=(Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm), bounds = bnds)  
        p = result.x

        info.write('D1_X = %f \n' %(p[0]))
        info.write('fc1_X = %f \n' %(p[1]))
        info.write('D2_X = %f \n' %(p[2]))
        info.write('fc2_X = %f \n' %(p[3]))
        info.write('a_QPD_X = %f \n' %(p[4]))
        info.write('f_QPD_X = %f \n\n' %(p[5]))
         
        self.D_X = p[2]
        self.fc_X = p[3] 

        dx = self.D_X/1000
        self.dD_X = (abs(LL_H([p[0], p[1], p[2]+dx, p[3], p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm) 
                       + LL_H([p[0], p[1], p[2]-dx, p[3], p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm) 
                     - 2*LL_H([p[0], p[1], p[2],    p[3], p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm))/dx**2)**-0.5

        dx = self.fc_X/1000
        self.dfc_X = (abs(LL_H([p[0], p[1], p[2], p[3]+dx, p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm) 
                        + LL_H([p[0], p[1], p[2], p[3]-dx, p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm) 
                      - 2*LL_H([p[0], p[1], p[2], p[3]   , p[4], p[5]], Pf, f, self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm))/dx**2)**-0.5

        self.PSD_X_fit = PSD_H(self.f, p[0], p[1], p[2], p[3], p[4], p[5], self.R, self.L, self. f_sample, self.f_lp, self.fv, self.fm)        
        self.residual_X = self.PSD_X[self.f!=self.fd]/self.PSD_X_fit[self.f!=self.fd]

        self.residual_X0 = np.zeros((self.N_avg, sum(self.f!=self.fd)))
        for j in range(self.N_avg): # per window
            self.residual_X0[j] = self.PSD_X0[j][self.f!=self.fd]/self.PSD_X_fit[self.f!=self.fd]
  
        # PSD_Y ################################################################                   
        Pf = self.PSD_Y[self.f!=self.fd]
        
        # MLE fitting      
        D1_Y = 0.01          
        fc1_Y = self.power*0.5
        D2_Y = 0.5 
        fc2_Y = self.power*5

        p0 = [D1_Y, fc1_Y, D2_Y, fc2_Y, a_QPD, f_QPD]
        bnds = ((D1_Y/10, D1_Y*10), (fc1_Y/10, fc1_Y*10), (D2_Y/10, D2_Y*10), (fc2_Y/10, fc2_Y*10), (0, 1), (f_QPD/10, f_QPD*10))

        result = minimize(LL_H, x0=p0, args=(Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm), bounds = bnds)  
        p = result.x

        info.write('D1_Y = %f \n' %(p[0]))
        info.write('fc1_Y = %f \n' %(p[1]))
        info.write('D2_Y = %f \n' %(p[2]))
        info.write('fc2_Y = %f \n' %(p[3]))
        info.write('a_QPD_Y = %f \n' %(p[4]))
        info.write('f_QPD_Y = %f \n\n' %(p[5]))

        self.D_Y = p[2]
        self.fc_Y = p[3]   

        dx = self.D_Y/1000
        self.dD_Y = (abs(LL_H([p[0], p[1], p[2]+dx, p[3], p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm) 
                       + LL_H([p[0], p[1], p[2]-dx, p[3], p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm) 
                     - 2*LL_H([p[0], p[1], p[2],    p[3], p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm))/dx**2)**-0.5

        dx = self.fc_Y/1000
        self.dfc_Y = (abs(LL_H([p[0], p[1], p[2], p[3]+dx, p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm) 
                        + LL_H([p[0], p[1], p[2], p[3]-dx, p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm) 
                      - 2*LL_H([p[0], p[1], p[2], p[3]   , p[4], p[5]], Pf, f, self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm))/dx**2)**-0.5

                             
        self.PSD_Y_fit = PSD_H(self.f, p[0], p[1], p[2], p[3], p[4], p[5], self.R, self.L, self.f_sample, self.f_lp, self.fv, self.fm)
        self.residual_Y = self.PSD_Y[self.f!=self.fd]/self.PSD_Y_fit[self.f!=self.fd]
        
        self.residual_Y0 = np.zeros((self.N_avg, sum(self.f!=self.fd)))
        for j in range(self.N_avg): # per window
            self.residual_Y0[j] = self.PSD_Y0[j][self.f!=self.fd]/self.PSD_Y_fit[self.f!=self.fd]
            
        # Determine beta, kappa_fit, gamma_fit, R_fit
        if self.axis == 'X': 
            p, cov = curve_fit(sine, self.time, self.PZTx,  p0=[self.fd, self.Ad, self.fd, 0])
            self.PZT_A = abs(p[1])   
            self.fc = self.fc_X
            self.dfc = self.dfc_X
            self.D_V = self.D_X    
            self.dD_V = self.dD_X       
            self.W_ex = self.df * (self.PSD_X[self.f==self.fd] - self.PSD_X_fit[self.f==self.fd])             
        else:
            p, cov = curve_fit(sine, self.time, self.PZTy,  p0=[self.fd, self.Ad, self.fd, 0])
            self.PZT_A = abs(p[1])            
            self.fc = self.fc_Y  
            self.dfc = self.dfc_Y                        
            self.D_V = self.D_Y
            self.dD_V = self.dD_Y            
            self.W_ex = self.df * (self.PSD_Y[self.f==self.fd] - self.PSD_Y_fit[self.f==self.fd])          
 
        self.W_th = W_H(self.PZT_A, self.fd, self.fc, self.fv, self.fm0)      
        self.beta = (self.W_th / self.W_ex)**0.5 
        self.D = self.D_V * self.beta**2    
        self.kappa = 2*pi*self.fc*kBT/self.D
        self.gamma = kBT/self.D
        self.ratio_stoke = self.gamma / self.gamma_0

        self.dW_th = abs(W_H(self.PZT_A, self.fd, self.fc+self.dfc, self.fv, self.fm0) - W_H(self.PZT_A, self.fd, self.fc-self.dfc, self.fv, self.fm0))/2
        self.dbeta = abs(((self.W_th+self.dW_th)/self.W_ex)**0.5 - ((self.W_th-self.dW_th)/self.W_ex)**0.5)/2         
        self.dD = abs((self.D_V+self.dD_V)*(self.beta+self.dbeta)**2 - (self.D_V-self.dD_V) * (self.beta-self.dbeta)**2)/2    
        self.dkappa = 2*pi*abs((self.fc+self.dfc)*kBT/(self.D-self.dD) - (self.fc-self.dfc)*kBT/(self.D+self.dD))/2
        self.dgamma = abs(kBT/(self.D-self.dD) - kBT/(self.D+self.dD))/2
        self.drs = self.dgamma / self.gamma_0

        print('A_fit = %.1f [nm])' % (abs(self.PZT_A)))
        print('beta = %.1f +/- %.1f [nm/V]' %(self.beta, self.dbeta))  
        print('fc = %.1f +/- %.1f [Hz]' % (self.fc, self.dfc))
        print('kappa = %.3f +/- %.3f [pN/nm]' %(self.kappa, self.dkappa))
        print('Stoke ratio = %.1f +/- %.1f \n' %(self.ratio_stoke, self.drs)) # Due to viscosity and surface hydrodynamic effect  

        info.write('Power = %.1f [%%] \n' %(self.power))
        info.write('Oscillation axis = %s \n' %(self.axis))
        info.write('fd = %d [Hz] \n' %(self.fd))   
        info.write('Ad = %.1f [nm] \n' %(abs(self.PZT_A)))   
        info.write('W_ex = %f \n' %(self.W_ex))   
        info.write('W_th = %f \n\n' %(self.W_th))   

        info.write('beta = %d +/- %d [nm/V] \n' %(self.beta, self.dbeta))  
        info.write('fc = %d +/- %d [Hz] \n' % (self.fc, self.dfc))
        info.write('kappa = %.3f +/- %.3f [pN/nm] \n' %(self.kappa, self.dkappa))
        info.write('Stoke ratio = %.3f +/- %.3f\n' %(self.ratio_stoke, self.drs)) # Due to viscosity and surface hydrodynamic effect    

        info.close()
             
    def plot_fig1(self):
        fig = plt.figure(1, figsize = (20, 10), dpi=300) 
        grid = gridspec.GridSpec(ncols=2, nrows=2)
        
        # X (V)
        sp = fig.add_subplot(grid[0,0])
        sp.hist(self.QPDx, bins=100, color='k', histtype='step', density=True)
        x = np.linspace(min(self.QPDx), max(self.QPDx), 100)
        sp.plot(x, norm.pdf(x, loc=0, scale=np.std(self.QPDx)), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('X (V)')
        sp.set_title('Width = %f (V)' %(np.std(self.QPDx))) 

        # Y (V)
        sp = fig.add_subplot(grid[1,0])
        y = np.linspace(min(self.QPDy), max(self.QPDy), 100)
        sp.hist(self.QPDy, bins=100, color='k', histtype='step', density=True)
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
#        self.fd
        fig = plt.figure(2, figsize = (20, 10), dpi=300)  

        # PSD_X
        sp = fig.add_subplot(221)
        sp.loglog(self.f, self.PSD_X, 'k.', ms=1)
        sp.loglog(self.f, self.PSD_X_fit, 'r', lw=2)       
        sp.set_xlabel('Frequency [Hz]')
        sp.set_ylabel('PSD_X [$V^2$ s]')
        if self.axis == 'X': 
            sp.set_title('beta = %d +/- %d [nm/V], k = %.3f +/- %.3f [pN/nm], fc = %d +/- %d [Hz]' 
                        %(self.beta, self.dbeta, self.kappa, self.dkappa, self.fc_X, self.dfc_X))
        else:
            sp.set_title('fc = %d +/- %d (Hz)' %(self.fc_X, self.dfc_X))            

        # Residual_X
        sp = fig.add_subplot(222)
        sp.plot(self.f[self.f!=self.fd], self.residual_X, 'k.', ms=1)
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlabel('Frequency [Hz]')
        sp.set_ylabel('Normalized PSD_X (Exp/Fit)')

        sp = fig.add_subplot(223)        
        sp.hist(self.residual_X, bins=20, color='k', histtype='step', density=True)
        x = np.linspace(min(self.residual_X), max(self.residual_X), 100)
        sp.plot(x, norm.pdf(x, loc=1, scale=1/(self.N_avg)**0.5), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Normalized PSD_X (Exp/Fit)')

        sp = fig.add_subplot(224)     
        x = np.linspace(0, max(self.residual_X0.flatten()), 100) 
        sp.hist(self.residual_X0.flatten(), bins=20, color='k', histtype='step', density=True)
        sp.plot(x, expon.pdf(x), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Normalized PSD_X0 (Exp/Fit)')
        
        fig.savefig(os.path.join(self.dir, 'Fig2_PSD_X.png'))
        plt.close(fig)                


    def plot_fig3(self):
        fig = plt.figure(3, figsize = (20, 10), dpi=300)  

        # PSD_Y
        sp = fig.add_subplot(221)
        sp.loglog(self.f, self.PSD_Y, 'k.', ms=1)
        sp.loglog(self.f, self.PSD_Y_fit, 'r', lw=2)
        sp.set_xlabel('Frequency (Hz)')
        sp.set_ylabel('PSD_Y [$V^2$ s]')

        if self.axis == 'Y': 
            sp.set_title('beta = %d +/- %d [nm/V], k = %.3f +/- %.3f [pN/nm], fc = %d +/- %d [Hz]' 
                        %(self.beta, self.dbeta, self.kappa, self.dkappa, self.fc_Y, self.dfc_Y))
        else:
            sp.set_title('fc = %d +/- %d (Hz)' %(self.fc_Y, self.dfc_Y)) 

        # Residual_Y
        sp = fig.add_subplot(222)
        sp.plot(self.f[self.f!=self.fd], self.residual_Y, 'k.', ms=1)
        sp.axhline(y=1, color='r', linestyle='solid', linewidth=2)
        sp.set_xlabel('Frequency [Hz]')
        sp.set_ylabel('Normalized PSD_Y (Exp/Fit)')             

        sp = fig.add_subplot(223)     
        sp.hist(self.residual_Y, bins=20, color='k', histtype='step', density=True)
        y = np.linspace(min(self.residual_Y), max(self.residual_Y), 100)
        sp.plot(y, norm.pdf(y, loc=1, scale=1/(self.N_avg)**0.5), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Normalized PSD_Y (Exp/Fit)')

        sp = fig.add_subplot(224)     
        y = np.linspace(0, max(self.residual_Y0.flatten()), 100)
        sp.hist(self.residual_Y0.flatten(), bins=20, color='k', histtype='step', normed=True)
        sp.plot(y, expon.pdf(y), 'r')
        sp.set_yscale('log')
        sp.set_xlabel('Normalized PSD_Y0 (Exp/Fit)')
                     
#        # PSD_XY
#        sp = fig.add_subplot(336)
#        sp.plot(self.f, self.PSD_XY, 'k', lw=1)
#        sp.set_xlabel('Frequency (Hz)')
#        sp.set_ylabel('PSD_XY')   
         
        fig.savefig(os.path.join(self.dir, 'Fig3_PSD_Y.png'))
        plt.close(fig)                

    def plot(self):
        self.plot_fig1()                                                
        self.plot_fig2()                                                                                                                                                  
        self.plot_fig3()  

def main(fname, f_sample, f_lp, R, power, axis, fd, Ad, height):
        
    data = Data(fname, f_sample, f_lp, R, power, axis, fd, Ad, height)
    data.read()
    data.analyze()
    data.plot()
    print('test')
    
    return data.PZT_A, data.beta, data.dbeta, data.kappa, data.dkappa, data.ratio_stoke, data.drs

if __name__ == "__main__":
    sys.exit(main())


# To-do
# Determine oscillation axis from the PZT_A
# Parameters from the input 


