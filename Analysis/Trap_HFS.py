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
from math import pi
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
n_avg = 20

###############################################

def step(t, tb, tu, Ab, Au, s1, s2):
    return (Ab-Au) * (erf(s1*(t-tb)) - erf(s2*(t-tu)))/2 + Au
              
def sine(t, A, ph, b): # Sine function
    return A * np.sin(2*pi*f_drive*t - ph) + b    

def running_mean(x, N = n_avg): # Running mean
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]

def find_outliers(data, m = 3.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    cutoff = np.median(data) + m*mdev
    i_outliers = data > cutoff
    return cutoff, i_outliers

def make_folder(name):
    path = os.path.join(os.getcwd(), name)       
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)    
    return path


class Event(object):
    def __init__(self):
        pass

class Data(object):
    def __init__(self, name):
        self.name = name

    def read(self):          
        tdms_file = TdmsFile(self.name)      # Reads a tdms file.
        root_object = tdms_file.object()     # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap"                             
        channels = tdms_file.group_channels(group_name) 
        self.ch_num = len(channels)                     
        self.ch_name = [str(channels[i].channel) for i in range(len(channels))] 
        self.dt = channels[0].properties[u'wf_increment']  # Sampling time
        self.fs = int(1.0/self.dt)                         # Sampling frequency
        self.N = len(channels[0].time_track())             # Number of time points

        print("Channel number: %d" % self.ch_num)
        print("Channel name: %s" % self.ch_name) 
        print("Sampling rate: %d Hz" % self.fs)   
        print("Data size: %d sec \n" % int(self.N*self.dt))  

        # Read data
        print("Reading raw data ... \n")
        self.t = channels[0].time_track(); 
        self.QPDy = (channels[1].data - np.median(reject_outliers(channels[1].data))) * QPD_nm2V[1]     
        self.PZTy = -(channels[4].data - np.median(channels[4].data)) * PZT_nm2V[1]
#        self.QPDs = (channels[2].data)

        self.T = int(self.fs / f_drive)       # Oscillation period in number
#        self.N_os = int(self.N / self.T)      # Number of oscillation
#        self.t = self.t[:self.T*self.n_os]      
#        self.QPDy = self.QPDy[:self.T*self.n_os]     
#        self.PZTy = self.PZTy[:self.T*self.n_os]    

    def wavelet_transformation(self):
        print("Wavelet transformation ... \n")
        t = self.t
        QPD = self.QPDy
        PZT = self.PZTy
        T = self.T

        # QPD fitting (Amp, Phase, Bg)
        QPD_cutoff, QPD_out = find_outliers(QPD)
        QPD_param, QPD_cov = curve_fit(sine, t[~QPD_out], QPD[~QPD_out], [A_drive/4, 0, 0])
        self.QPD_A0 = QPD_param[0]
        self.QPD_P0 =QPD_param[1] 
        self.QPD_B0 =QPD_param[2] 
        self.QPD0 = sine(t, self.QPD_A0, self.QPD_P0, self.QPD_B0)       
        self.dQPD = QPD - self.QPD0

        # PZT fitting (Amp, Phase, Bg)
        p0 = [A_drive*0.8, 0, 0]
        bnds = ((A_drive*0.5, -pi, -10), (A_drive, pi, 10))
        PZT_param, PZT_cov = curve_fit(sine, t, PZT, p0, bounds=bnds)
        self.PZT_A = PZT_param[0]
        self.PZT_P = PZT_param[1] 
        self.PZT_B = PZT_param[2]   
        self.PZT_fit = sine(t, self.PZT_A, self.PZT_P, self.PZT_B)

        # Wavelet transformation
        pdata1 = self.dQPD * sine(t, self.PZT_A, self.PZT_P, self.PZT_B)
        pdata2 = self.dQPD * sine(t, self.PZT_A, self.PZT_P + pi/2, self.PZT_B)   
        y1 = 2 * convolve(pdata1, np.ones(T)/T, mode='valid')
        y2 = 2 * convolve(pdata2, np.ones(T)/T, mode='valid')
        y = (y1**2 + y2**2)**0.5            
        dQPD_A = y/self.PZT_A          
        self.dQPD_A = np.concatenate((dQPD_A[0]*np.ones(int(T/2)), 
                                      dQPD_A, dQPD_A[-1]*np.ones(int(T/2))))  

    def binding(self):
        print("Find binding events ... \n")
        # Running_mean
        # cutoff, outliers
        # find tb, tu
        # exclude bad events
        # index for tb, tu

        t = self.t
        dQPD = self.dQPD
        PZT = self.PZTy
        T = self.T



            
        self.dQPD_cut, self.dQPD_out = find_outliers(self.QPD_A)
        
       
    def find_index_over_cutoff(self, t, QPD_A, A_cut):
            t_m = running_mean(t)
            A_m = running_mean(QPD_A)

            i_A_m = A_m > A_cut
            i_A = np.concatenate((np.zeros(int(n_avg/2-1), dtype=bool), i_A_m, np.zeros(int(n_avg/2-1), dtype=bool)))

            return t_m, A_m, i_A_m, i_A      

    def traces(self, path):              
        n = int(self.fs * t_block)    # Number of data in a time block
        m = int(self.N / n)           # Number of block

        if m == 0:
            print("Time block is too short... \n")
            n = self.N
            m = 1

        t = self.t[:n*m].reshape((m,n))       
        QPD = self.QPDy[:n*m].reshape((m,n))      
        PZT = self.PZTy[:n*m].reshape((m,n))       
        PZT_fit = self.PZT_fit[:n*m].reshape((m,n))     
        QPD_A = self.QPD_A[:n*m].reshape((m,n))            
        A_out = self.A_out[:n*m].reshape((m,n))


        QPD_param, QPD_cov = curve_fit(sine, self.t, self.QPDy, [50, 0, 0])
        QPD_A_fit = QPD_param[0]
        QPD_P_fit =QPD_param[1] 
        QPD_b_fit =QPD_param[1] 

        self.QPD0 = sine(t, QPD_A_fit, QPD_P_fit, QPD_b_fit) 
        QPD0 = self.QPD0[:n*m].reshape((m,n))      
        dQPD = QPD - QPD0       
     
        print("Plotting traces ... \n")
        for i in range(m):                   
            [t_m, A_m, P_m, i_A_m, i_P_m, i_AP_m, i_AP] = self.find_index_over_cutoff(t[i], QPD_A[i], QPD_A[i], self.A_cut, self.A_cut)
            t_m = running_mean(t[i])
            A_m = running_mean(QPD_A[i])                                                             
                                                                                                                                                                                                
            fig = plt.figure(i, figsize = (20, 10), dpi=300) 
            
            sp = fig.add_subplot(311)
            sp.plot(t[i], dQPD[i], 'k', linewidth=1)                                                                    
            sp.axhline(y=0, color='g', linestyle='dashed', linewidth=1)   
            sp.set_ylabel('dQPD (nm)')            
            
            sp = fig.add_subplot(312)   
            sp.plot(t[i], QPD_A[i], 'k', linewidth=1)                                  
            sp.axhline(y=np.median(reject_outliers(self.QPD_A)), color='g', linestyle='dashed', linewidth=1)
            sp.axhline(y=self.A_cut, color='b', linestyle='dashed', linewidth=1) 
            sp.set_ylabel('Amplitude (nm)') 
            
            sp = fig.add_subplot(313)   
            sp.plot(t_m, A_m, 'k', linewidth=1)    
            sp.plot(t_m[i_A_m], A_m[i_A_m], 'r.', ms=2)                        
            sp.axhline(y=np.median(reject_outliers(self.QPD_A)), color='g', linestyle='dashed', linewidth=1)
            sp.axhline(y=self.A_cut, color='b', linestyle='dashed', linewidth=1)                                      
            sp.set_ylabel('Amplitude_Avg (nm)')
            
            fig_name = self.name[:-5] + '_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)     

    def events(self, path):
        t = self.t      
        QPD = self.QPDy   
        PZT_fit = self.PZT_fit    
        QPD_A = self.QPD_A      
        QPD_P = self.QPD_p 
               
        self.A_cut, self.A_out = outliers(self.QPD_A)
        self.P_cut, self.P_out = outliers(self.QPD_p)        
        self.AP_out = self.A_out & self.P_out               
               
        [t_m, A_m, P_m, i_A_m, i_P_m, i_AP_m, i_AP] = self.find_index_over_cutoff(t, QPD_A, QPD_P, self.A_cut, self.P_cut)

        i_on = []
        i_off = []

        for i in range(len(t)-1):
            if i_AP[i] == False and i_AP[i+1] == True:
                i_on.append(i)
            if i_AP[i] == True and i_AP[i+1] == False:
                i_off.append(i)                                                     

        # Remove incomplete events
        if i_off[0] < i_on[0]:
            i_off = i_off[1:]            
        if len(i_on) > len(i_off):
            i_on = i_on[:-1]

        # Remove too short or too long events
        i_ex = []
        if len(i_on) > 0:
            for i in range(len(i_on)):
                if (i_off[i] - i_on[i] > self.fs*0.2) or (i_off[i] - i_on[i] < self.T):
                    i_ex.append([i_on[i], i_off[i]])
            if len(i_ex) > 0:
                for i in range(len(i_ex)):
                    i_on.remove(i_ex[i][0])
                    i_off.remove(i_ex[i][1])
                 
        print(self.name)
        
        if len(i_on) == len(i_off):
            print("%d events are detected. \n" %(len(i_on)))    
            if len(i_on) == 0:
                return
        else:
            print("Error: incomplete binding and unbinding events")      
            return  
                                    
        for j in range(len(i_on)):
            i = np.arange(max(i_on[j]-self.T*2, 0), min(i_off[j]+self.T*2, len(t)))
                                           
            # Step finding with Amp and Phase 
            i_b = np.array(QPD_A[i] > self.A_cut) & np.array(QPD_P[i] > self.P_cut)   
            i_u = ~i_b           
            
            tb0 = t[i][i_b].min()
            tu0 = t[i][i_b].max()
           
            Ab0 = np.median(reject_outliers(QPD_A[i][i_b]))
            Au0 = np.median(reject_outliers(QPD_A))
            
            Pb0 = np.median(reject_outliers(QPD_P[i][i_b]))
            Pu0 = np.median(reject_outliers(QPD_P))

            # Amplitude fitting
            p0 = [tb0, tu0, Ab0, Au0, self.fs, self.fs]
            lb = (t[i][1], t[i][1], min(QPD_A[i][i_b]), Au0-0.1, self.fs/2, self.fs/2)
            ub = (t[i][-2], t[i][-2], max(QPD_A[i][i_b]), Au0+0.1, self.fs*2, self.fs*2)           
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
            lb = (t[i][1], t[i][1], -pi, Pu0-0.01, self.fs/2, self.fs/2)
            ub = (t[i][-2], t[i][-2], pi, Pu0+0.01, self.fs*2, self.fs*2)
            P_param, P_cov = curve_fit(step, t[i], QPD_P[i], p0, bounds = (lb, ub))

            tPb_fit = P_param[0]
            tPu_fit = P_param[1]
            Pb_fit = P_param[2]
            Pu_fit = P_param[3]            
            Ps1_fit = A_param[4]
            Ps2_fit = A_param[5] 
            QPD_P_fit = step(t_fit, tPb_fit, tPu_fit, Pb_fit, Pu_fit, Ps1_fit, Ps2_fit) 

            if abs(tAb_fit - tPb_fit) < 0.02:
                tb_fit = (tAb_fit + tPb_fit) * 0.5         
            else:
                continue      
      
            if abs(tAu_fit - tPu_fit) < 0.02:
                tu_fit = (tAu_fit + tPu_fit) * 0.5         
            else:
                continue   
                   
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
            sp.axhline(y=self.A_cut, color='b', linestyle='dashed', linewidth=1) 
            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)   
            sp.set_ylabel('Amplitude (nm)')
                                  
            sp = fig.add_subplot(414)     
            sp.plot(t_fit, QPD_P_fit, 'r', linewidth=2)               
            sp.plot(t[i], QPD_P[i], 'ko', ms=2)      
            sp.axhline(y=Pu0, color='k', linestyle='dashed', linewidth=1)                    
            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)   
            sp.axhline(y=self.P_cut, color='b', linestyle='dashed', linewidth=1)    
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
        if calibrate == True:
            print(self.cal_name)
        else:
            pass
            
    def read_data(self):
        self.data = []
        for name in self.data_list:
            data = Data(name)
            data.read()
            self.data.append(data)
            
    def transform(self):
        for i in range(len(self.data)):
            self.data[i].wavelet_transformation()

    def find_binding(self):
        for i in range(len(self.data)):
            self.data[i].binding()

    def plot_traces(self):
        path = make_folder('Traces')
        for i in range(len(self.data)):
            self.data[i].traces(path)
            
    def fit_events(self):
        path = make_folder('Events')        
        for i in range(len(self.data)):
            self.data[i].events(path)

    def combine_events(self):
        pass
        
    def fitting(self):
        pass


                                                       
def main():
    mol = Molecule()
    mol.calibrate()
    mol.read_data()
    mol.transform()
    mol.find_binding()
    mol.plot_traces()    
#    mol.fit_events()
    mol.combine_events()
    mol.fitting()
    



if __name__ == "__main__":
    main()



"""""""""
To-do

Trim the code for dQPD transformation

Triangle oscillation and get binding ratio of forward vs backward motion, get freq dep
slower harmonic oscillation to get better phase estimate. 

Better step initial time (tb0, tu0) based on the max

split two events close 
better event picking: two events too close 

Be more conservative to exclude bad data. Only take good data. 
Time threshold for events. Upper and lower bound.


Solve fitting error
maxfev error detour. using try error

Finish event picking and show force dependent result. 
Analyze fs = 10 kHz data

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





