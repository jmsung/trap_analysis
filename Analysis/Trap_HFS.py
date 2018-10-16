##################################################################
#
# Analysis of Harmonic force spectroscopy data (by Jongmin Sung)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import os
import shutil
from scipy.optimize import curve_fit
from scipy.special import erf
from math import pi, atan
from scipy import convolve

### User input ##################################

calibrate = False
QPD_nm2V = [100, 60]      # QPD sensitivity (nm/V) at V_sum = 8 V.
stiffness_pN2nm = [0.1, 0.1]  # Stiffness [pN/nm]
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor

R = 430     # Bead radius, expected according to the spec (nm)  
L = R + 250/2 # Axoneme diameter = 250 nm 
L = R
rho = 1e-21 # Density of water (and bead) (pN s^2/nm^4)
nu = 1e12   # Kinematic viscosity of water (nm^2/s)
gamma0 = 6.*pi*rho*nu*R
gamma = gamma0/(1-9*R/(16*L)+R**3/(8*L**3)-45*R**4/(256*L**4)-R**5/(16*L**5))  # Zero frequency Stokes drag coefficent (pN s/nm)
fc = stiffness_pN2nm[1] / (2*pi*gamma)

f_drive = 100 # Hz
A_drive = 100 # nm
t_block = 10

#A_cut = 35
#P_cut = -0.9 
n_avg = 30

###############################################

def step(t, tb, tu, Ab, Au, s1, s2):
    return (Ab-Au) * (erf(s1*(t-tb)) - erf(s2*(t-tu)))/2 + Au

def step1(t, tb, tu, yb):
    return yb * ((np.sign(t-tb)+1) - (np.sign(t-tu)+1))/2 
              
def sine(t, A, ph, b): # Sine function
    return A * np.sin(2*pi*f_drive*t - ph) + b    

def running_mean(x, N = n_avg): # Running mean
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s<m]

def outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    cutoff = np.median(data) + m * mdev
    outliers = data > cutoff
    return cutoff, outliers

class Event(object):
    def __init__(self):
        pass

class Data(object):
    def __init__(self, name):
        self.name = name

    def read(self):          
        tdms_file = TdmsFile(self.name) # Reads a tdms file.
        root_object = tdms_file.object() # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap"                             
        channels = tdms_file.group_channels(group_name) 
        self.channel_num = len(channels)                     
        self.channel_name = [str(channels[i].channel) for i in range(len(channels))] # Channel name
        self.dt = channels[0].properties[u'wf_increment'] # Sampling time
        self.fs = int(1.0/self.dt)                         # Sampling frequency
        self.N = len(channels[0].time_track())

        print("Channel number: %d" % self.channel_num)
        print("Channel name: %s" % self.channel_name) 
        print("Sampling rate: %d Hz" % self.fs)   
        print("Data size: %d sec \n" % int(self.N*self.dt))  

        # Read data
        print("Reading raw data ... \n")
        self.t = channels[0].time_track(); 
        self.QPDy = (channels[1].data - np.mean(channels[1].data)) * QPD_nm2V[1]     
        self.QPDs = (channels[2].data)
        self.PZTy = -(channels[4].data - np.mean(channels[4].data)) * PZT_nm2V[1]

        self.T = int(self.fs / f_drive)       # Oscillation period in number
        self.n_os = int(self.N / self.T)           # Number of oscillation
        self.t = self.t[:self.T*self.n_os]      
        self.QPDy = self.QPDy[:self.T*self.n_os]  
        self.QPDs = self.QPDs[:self.T*self.n_os]     
        self.PZTy = self.PZTy[:self.T*self.n_os]    


    def wavelet_transformation(self):
        t = self.t
        QPD = self.QPDy
        PZT = self.PZTy
        T = self.T

        # param = [A, phi, b]
        p0 = [A_drive*0.8, -pi/2, 0]
        bnds = ((A_drive*0.5, -pi, -10), (A_drive, pi, 10))
        PZT_param, PZT_cov = curve_fit(sine, t, PZT, p0, bounds=bnds)
        self.PZT_A = PZT_param[0]
        self.PZT_phi = PZT_param[1] 
        self.PZT_b = PZT_param[2] 
        
        self.PZT_fit = sine(t, self.PZT_A, self.PZT_phi, self.PZT_b)
        pdata1 = QPD * sine(t, self.PZT_A, self.PZT_phi, self.PZT_b)
        pdata2 = QPD * sine(t, self.PZT_A, self.PZT_phi + pi/2, self.PZT_b)   

        y1 = 2 * convolve(pdata1, np.ones(T)/T, mode='valid')
        y2 = 2 * convolve(pdata2, np.ones(T)/T, mode='valid')
        y = (y1**2 + y2**2)**0.5        
        
        QPD_A = y/self.PZT_A          
        self.QPD_A = np.concatenate((QPD_A[0]*np.ones(int(T/2)), QPD_A, QPD_A[-1]*np.ones(int(T/2))))                     
       
        test = np.zeros(len(y))
        for n in range(len(y)):
            if y1[n] > 0.0 and y2[n] > 0.0:
                test[n] = atan(y2[n]/y1[n])
            elif y1[n] < 0.0 and y2[n] > 0.0:
                test[n] = atan(y2[n]/y1[n]) + pi
            elif y1[n] > 0.0 and y2[n] < 0.0:
                test[n] = atan(y2[n]/y1[n])
            elif y1[n] < 0.0 and y2[n] < 0.0:
                test[n] = atan(y2[n]/y1[n]) - pi
                
        QPD_P = test
#        QPD_P = np.mod(QPD_P + pi, 2*pi) - pi

        self.QPD_p = np.concatenate((QPD_P[0]*np.ones(int(T/2)), QPD_P, QPD_P[-1]*np.ones(int(T/2))))

    def find_index_over_cutoff(self, t, QPD_A, QPD_P):
            t_m = running_mean(t)
            A_m = running_mean(QPD_A)
            P_m = running_mean(QPD_P)

            i_A_m = A_m > A_cut
            i_P_m = P_m > P_cut
            i_AP_m = np.array(i_A_m, dtype=bool) & np.array(i_P_m, dtype=bool)
            i_AP = np.concatenate((np.zeros(int(n_avg/2), dtype=bool), i_AP_m, np.zeros(int(n_avg/2-1), dtype=bool)))

            return t_m, A_m, P_m, i_A_m, i_P_m, i_AP_m, i_AP       

    def traces(self, path):              
        n = int(self.fs * t_block)    # Number of data in a block
        m = int(self.N / n)      # Number of block

        t = self.t[:n*m].reshape((m,n))       
        QPD = self.QPDy[:n*m].reshape((m,n))     
        QPDs = self.QPDs[:n*m].reshape((m,n))   
        PZT = self.PZTy[:n*m].reshape((m,n))       
        PZT_fit = self.PZT_fit[:n*m].reshape((m,n))     
        QPD_A = self.QPD_A[:n*m].reshape((m,n))       
        QPD_P = self.QPD_p[:n*m].reshape((m,n))       

        A_cut, A_out = outliers(self.QPD_A)
        P_cut, P_out = outliers(self.QPD_p)        
        AP_out = A_out & P_out

        A_out = A_out[:n*m].reshape((m,n)) 
        P_out = P_out[:n*m].reshape((m,n)) 
        AP_out = AP_out[:n*m].reshape((m,n)) 
  
        p0 = atan(f_drive/fc) - pi/2
        A0 = np.max(PZT_fit)/(1 + (fc/f_drive)**2)**0.5                
       
        print("Plotting traces ... \n")
        for i in range(m):                   
#            [t_m, A_m, P_m, i_A_m, i_P_m, i_AP_m, i_AP] = self.find_index_over_cutoff(t[i], QPD_A[i], QPD_P[i])
            t_m = running_mean(t[i])
            A_m = running_mean(QPD_A[i])
            P_m = running_mean(QPD_P[i])
                                                                
            fig = plt.figure(i, figsize = (20, 10), dpi=300) 

            sp = fig.add_subplot(611)                     
            sp.plot(t[i], PZT[i] - PZT_fit[i], 'k', linewidth=0.5)              
            sp.set_ylabel('PZT - PZT_fit (nm)')       
                  
            sp = fig.add_subplot(612)
            sp.plot(t[i], QPD[i], 'k', linewidth=1)      
            sp.plot(t[i][AP_out[i]], QPD[i][AP_out[i]], 'r.', ms=2)                                                                
            sp.axhline(y=0, color='g', linestyle='dashed', linewidth=1)   
            sp.set_ylabel('QPD (nm)')
            
            sp = fig.add_subplot(613)   
            sp.plot(t[i], QPD_A[i], 'k', linewidth=1)    
            sp.plot(t[i][A_out[i]], QPD_A[i][A_out[i]], 'b.', ms=2)             
            sp.plot(t[i][AP_out[i]], QPD_A[i][AP_out[i]], 'r.', ms=2)           
            sp.axhline(y=np.median(reject_outliers(QPD_A)), color='g', linestyle='dashed', linewidth=1)
            sp.axhline(y=A_cut, color='b', linestyle='dashed', linewidth=1) 
#            sp.axhline(y=A0, color='k', linestyle='dashed', linewidth=1)  
            sp.set_ylabel('Amplitude (nm)')
            
            sp = fig.add_subplot(614)   
            sp.plot(t_m, A_m, 'k', linewidth=1)    
            sp.plot(t[i][A_out[i]], QPD_A[i][A_out[i]], 'b.', ms=2)             
            sp.plot(t[i][AP_out[i]], QPD_A[i][AP_out[i]], 'r.', ms=2)              
            sp.axhline(y=np.median(reject_outliers(QPD_A)), color='g', linestyle='dashed', linewidth=1)
            sp.axhline(y=A_cut, color='b', linestyle='dashed', linewidth=1) 
#            sp.axhline(y=A0, color='k', linestyle='dashed', linewidth=1)                                         
            sp.set_ylabel('Amplitude_Avg (nm)')
                    
            sp = fig.add_subplot(615)     
            sp.plot(t[i], QPD_P[i], 'k', linewidth=1)   
            sp.plot(t[i][P_out[i]], QPD_P[i][P_out[i]], 'b.', ms=2)             
            sp.plot(t[i][AP_out[i]], QPD_P[i][AP_out[i]], 'r.', ms=2)       
            sp.axhline(y=np.median(reject_outliers(QPD_P)), color='g', linestyle='dashed', linewidth=1)                    
#            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)       
            sp.axhline(y=P_cut, color='b', linestyle='dashed', linewidth=1)    
#            sp.axhline(y=p0, color='k', linestyle='dashed', linewidth=1) 
            sp.set_ylabel('Phase (rad)')            
                            
            sp = fig.add_subplot(616)      
            sp.plot(t_m, P_m, 'k', linewidth=1) 
            sp.plot(t[i][P_out[i]], QPD_P[i][P_out[i]], 'b.', ms=2)             
            sp.plot(t[i][AP_out[i]], QPD_P[i][AP_out[i]], 'r.', ms=2)                          
            sp.axhline(y=np.median(reject_outliers(QPD_P)), color='g', linestyle='dashed', linewidth=1)
#            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)                            
            sp.axhline(y=P_cut, color='b', linestyle='dashed', linewidth=1)    
#            sp.axhline(y=p0, color='k', linestyle='dashed', linewidth=1)                                              
            sp.set_ylabel('Phase_Avg (rad)')
            sp.set_xlabel('Time (s)')   
            
#            sp = fig.add_subplot(311)
#            sp.plot(t[i], QPDs[i], 'k', linewidth=0.5)                                
#            sp.set_ylabel('QPDs (nm)')   

  
            fig_name = self.name[:-5] + '_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)          

    def events(self, path):
        t = self.t      
        QPD = self.QPDy   
        PZT_fit = self.PZT_fit    
        QPD_A = self.QPD_A      
        QPD_P = self.QPD_p 
               
        [t_m, A_m, P_m, i_A_m, i_P_m, i_AP_m, i_AP] = self.find_index_over_cutoff(t, QPD_A, QPD_P)

        i_on = []
        i_off = []

        for i in range(len(t)-1):
            if i_AP[i] == False and i_AP[i+1] == True:
                i_on.append(i)
            if i_AP[i] == True and i_AP[i+1] == False:
                i_off.append(i)                                                     

        if i_off[0] < i_on[0]:
            i_off = i_off[1:]            
        if len(i_on) > len(i_off):
            i_on = i_on[:-1]

        print(self.name)
        if len(i_on) == len(i_off):
            print("%d events are detected. \n" %(len(i_on)))    
        else:
            print("Error: incomplete binding and unbinding events")        
                                    
        for j in range(len(i_on)):
            i = np.arange(max(i_on[j]-50, 0), min(i_off[j]+50, len(t)))
                                           
            # Stepping finding with Amp and Phase 
            i_b = np.array(QPD_A[i] > A_cut) & np.array(QPD_P[i] > P_cut)   
            i_u = ~i_b           
            
            tb0 = t[i][i_b].min()
            tu0 = t[i][i_b].max()
            
            Ab0 = np.median(reject_outliers(QPD_A[i][i_b]))
            Au0 = np.median(reject_outliers(QPD_A))
            
            Pb0 = np.median(reject_outliers(QPD_P[i][i_b]))
            Pu0 = np.median(reject_outliers(QPD_P))

            # Amplitude fitting
            p0 = [tb0, tu0, Ab0, Au0, self.fs, self.fs]
            lb = (t[i][1], t[i][1], min(QPD_A[i][i_b]), min(QPD_A[i][i_u]), self.fs/2, self.fs/2)
            ub = (t[i][-2], t[i][-2], max(QPD_A[i][i_b]), max(QPD_A[i][i_u]), self.fs*2, self.fs*2)           
            A_param, A_cov = curve_fit(step, t[i], QPD_A[i], p0, bounds = (lb, ub))
            tAb_fit = A_param[0]
            tAu_fit = A_param[1]
            Ab_fit = A_param[2]
            Au_fit = A_param[3]
            As1_fit = A_param[4]
            As2_fit = A_param[5]            
            t_fit = np.arange(t[i][0], t[i][-1], 0.0001) 
            QPD_A_fit = step(t_fit, tAb_fit, tAu_fit, Ab_fit, Au_fit, As1_fit, As2_fit) 

            # Phase fitting
            p0 = [tb0, tu0, Pb0, Pu0, self.fs, self.fs]
            lb = (t[i][1], t[i][1], -pi, -pi, self.fs/2, self.fs/2)
            ub = (t[i][-2], t[i][-2], pi, pi, self.fs*2, self.fs*2)
            P_param, P_cov = curve_fit(step, t[i], QPD_P[i], p0, bounds = (lb, ub))

            tPb_fit = P_param[0]
            tPu_fit = P_param[1]
            Pb_fit = P_param[2]
            Pu_fit = P_param[3]            
            Ps1_fit = A_param[4]
            Ps2_fit = A_param[5] 
            QPD_P_fit = step(t_fit, tPb_fit, tPu_fit, Pb_fit, Pu_fit, Ps1_fit, Ps2_fit) 

            if abs(tAb_fit - tPb_fit) < 0.01:
                tb_fit = (tAb_fit + tPb_fit) * 0.5         
            else:
                tb_fit = max(tAb_fit, tPb_fit)         
      
            if abs(tAu_fit - tPu_fit) < 0.01:
                tu_fit = (tAu_fit + tPu_fit) * 0.5         
            else:
                tu_fit = min(tAu_fit, tPu_fit)    
                   
            # QPD fitting
            i_b1 = np.array(t[i] > tb_fit, dtype=bool) & np.array(t[i] < tu_fit, dtype=bool)
            i_u1 = ~i_b1
             
            b_param, b_cov = curve_fit(lambda t, ph, b: sine(t, Ab_fit, ph, b), t[i][i_b1], QPD[i][i_b1], [0, 0])
            Pb_fit = b_param[0]
            bb_fit = b_param[1]            
            u_param, u_cov = curve_fit(lambda t, ph, b: sine(t, Ab_fit, ph, b), t[i][i_u1], QPD[i][i_u1], [0, 0])
            Pu_fit = u_param[0]
            bu_fit =u_param[1]  

            # QPD plot
            i_b2 = np.array(t_fit > tb_fit, dtype=bool) & np.array(t_fit < tu_fit, dtype=bool)
            i_u2 = ~i_b2

            tb = np.arange(t_fit[i_b2][0], t_fit[i_b2][-1], 0.0001)
            tu1 = np.arange(t_fit[i_u2][0], t_fit[i_b2][0], 0.0001)
            tu2 = np.arange(t_fit[i_b2][-1], t_fit[i_u2][-1], 0.0001)

            QPD_b_fit = sine(tb, Ab_fit, Pb_fit, bb_fit) 
            QPD_u1_fit = sine(tu1, Au_fit, Pu_fit, bu_fit) 
            QPD_u2_fit = sine(tu2, Au_fit, Pu_fit, bu_fit) 

            # Plot
            fig = plt.figure(j, figsize = (20, 10), dpi=300)   
            
            sp = fig.add_subplot(411)    
            PZT_fit2 = sine(t_fit, self.PZT_A, self.PZT_phi, self.PZT_b)                 
            sp.plot(t_fit, PZT_fit2, 'r', linewidth=2)  
            sp.plot(t[i], PZT_fit[i], 'ko', ms=2)   
            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)             
            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)              
            sp.set_ylabel('PZT (nm)') 
            
            sp = fig.add_subplot(412)
            sp.plot(tb, QPD_b_fit, 'r', lw=2)       
            sp.plot(tu1, QPD_u1_fit, 'b', lw=2)     
            sp.plot(tu2, QPD_u2_fit, 'b', lw=2)   
            sp.plot(t[i], QPD[i], 'ko', ms=2)                                                                                
            sp.axhline(y=bb_fit, color='r', linestyle='dashed', linewidth=1)   
            sp.axhline(y=bu_fit, color='b', linestyle='dashed', linewidth=1)               
            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.set_ylabel('QPD (nm)')               

            sp = fig.add_subplot(413)   
            sp.plot(t_fit, QPD_A_fit, 'r', lw=2)   
            sp.plot(t[i], QPD_A[i], 'ko', ms=2)         
            sp.axhline(y=Au0, color='k', linestyle='dashed', linewidth=1)
            sp.axhline(y=A_cut, color='b', linestyle='dashed', linewidth=1) 
            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.set_ylabel('Amplitude (nm)')
                                  
            sp = fig.add_subplot(414)     
            sp.plot(t_fit, QPD_P_fit, 'r', linewidth=2)               
            sp.plot(t[i], QPD_P[i], 'ko', ms=2)      
            sp.axhline(y=Pu0, color='k', linestyle='dashed', linewidth=1)                    
            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)   
            sp.axhline(y=P_cut, color='b', linestyle='dashed', linewidth=1)    
            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)      
            sp.set_ylabel('Phase (rad)')            
            sp.set_xlabel('Time (s)')      

            fig_name = self.name[:-5] + '_' + str(j) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)    

               
class Molecule(object):
    def __init__(self):
        path = os.getcwd()              
        file_list = os.listdir(path) 
        self.data_list = []
        for file_name in file_list:
            if file_name[-4:] == 'tdms':
                if file_name[:3].lower() == 'cal': 
                    self.cal_name = file_name
                else: 
                    self.data_list.append(file_name)      
        
    def calibrate(self):
        pass
            
    def read_data(self):
        self.data = []
        for name in self.data_list:
            data = Data(name)
            data.read()
            self.data.append(data)
            
    def transform(self):
        for i in range(len(self.data)):
            data = self.data[i]
            data.wavelet_transformation()

    def plot_traces(self):
        # Make a directory to save the results 
        path = os.path.join(os.getcwd(), 'Traces')
        
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)     

        for i in range(len(self.data)):
            data = self.data[i]
            data.traces(path)

    def detect_events(self):
        # Make a directory to save the results 
        path = os.path.join(os.getcwd(), 'Events')
        
        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)     

        for i in range(len(self.data)):
            data = self.data[i]
            data.events(path)

    def combine_events(self):
        pass
        
    def fitting(self):
        pass


                                                       
def main():
    mol = Molecule()
    mol.calibrate()
    mol.read_data()
    mol.transform()
    mol.plot_traces()    
#    mol.detect_events()
    mol.combine_events()
    mol.fitting()
    



if __name__ == "__main__":
    main()



"""""""""
To-do

Be more conservative to exclude bad data. Only take good data. 
Finish event picking and show force dependent result. 
I will collect more data at fs = 10 kHz, then I have to update the code anyway. 
Hence, I don't need to over optimize the code. 

split two events close 

better event picking: two events too close 


maxfev error detour. using try error


Histogram of veocity or phase at tb and tu

binding phase distribuion (clustered?)
unbinding phase distribution (random?)

stalk stiffness at diff nt states
subtract QPD noise from sum
check correlation btw QPDs, Amp, Phase oscillation
Fourier transform with QPDs, Amp, Phase whether they are from the same noise source h hy hy


> Check Kim's code: HFS_event_detection, HFS_fitting
> Global MLE fitting (instead of binning) as in the xxx Goldman paper. Talk to Chao. 
> Simultaneous QPDx and Fx
> step finding? use 8 nm grid to find offset > fitting with grid as ref
> How to offset QPDx and QPDy for Force??
> In the setup: beta, stiffness, MTA, 
> Directly save/read from tmds (beta, stiffness, MTA, f_sample, f_lowpass, ch_name)
> More details in the description (sample, bead, axoneme vs MT, coverslip, conc, stall force or feedback, )
> Feedback axis?


> module_name, package_name, ClassName, method_name, ExceptionName, 
> function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, 
> function_parameter_name, local_var_name



"""""""""





