##################################################################
#
# Analysis of Harmonic force spectroscopy data (by Jongmin Sung)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nptdms 
import os #import path, makedirs, getcwd, listdir
import shutil #import rmtree
import scipy
from scipy.optimize import curve_fit
from scipy.special import erf
import math 
import sys 


### User input ##################################
f_sample = 2000
dt = 1/f_sample
QPD_nm2V = [551.5, 375.1]      # QPD sensitivity (nm/V) at V_sum = 8 V.
stiffness_pN2nm = [0.103, 0.103]  # Stiffness [pN/nm] 0.005 pN/nm per % power
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor
pi = 3.141592

# PZT oscillation
f_drive = 10 # Hz
A_drive = 1000 # nm
N_window = 5  # Number of oscillation per window
t_window = N_window  / f_drive
n_avg = 1

t_short = 1.0/f_drive*1
t_long = 0.5
A_RMSD_cut = 20
QPD_RMSD_cut = 20
Abu_cut = 1.2
outlier_cut = 5


###############################################

def step(t, tb, tu, Ab, Au, s1, s2):
    return (Ab-Au) * (erf(s1*(t-tb)) - erf(s2*(t-tu)))/2 + Au
              
def sine(t, A, ph, b): # Sine function
    return A * np.sin(2*pi*f_drive*t - ph) + b    

def triangle(t, A, ph, b):
    t = 2 * np.pi * f_drive * t - ph + np.pi/2
    
    t, w = np.asarray(t), np.asarray(0.5)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (np.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (np.pi * (wsub + 1) - tsub) / (np.pi * (1 - wsub)))
    return A*y + b

def trapzoid(t, A, ph, b, m):
    t = 2 * np.pi * f_drive * t - ph + np.pi/2
    
    t, w = np.asarray(t), np.asarray(0.5)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (np.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (np.pi * (wsub + 1) - tsub) / (np.pi * (1 - wsub)))

    y[y > A*m] = A*m
    y[y < -A*m] = -A*m   
        
    return A*y + b

def square(t, A, ph, b):
    duty = 0.5
    t = 2 * np.pi * f_drive * t - ph

    t, w = np.asarray(t), np.asarray(duty)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = np.mod(t, 2 * pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * pi)
    np.place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    np.place(y, mask3, -1)
    return A*y + b

def exp(F, t0, dF):
    dF = abs(dF)
    return t0*np.exp(-F/dF)

def running_mean(x, N = n_avg): # Running mean
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)
    # [:N], [-N:]

def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]

def find_outliers(data, m = outlier_cut):
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


class Event:
    def __init__(self, name):
        self.name = name
 
        
class Data:
    def __init__(self, name):
        self.name = name

    def read_data(self):          
        tdms_file = nptdms.TdmsFile(self.name)      # Reads a tdms file.
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
        t = channels[0].time_track()
        QPDy = (channels[1].data - np.median(reject_outliers(channels[1].data))) * QPD_nm2V[1]     
        self.PZTy = (channels[4].data - np.median(channels[4].data)) * PZT_nm2V[1]
        QPDs = (channels[2].data)
        QPD = QPDy / QPDs
        
        # Subtract low-frequency noise
        QPD_cut = np.array(QPD)        
        self.QPD_max = 3*np.median(np.abs(QPD_cut))
        QPD_cut[QPD_cut > self.QPD_max] = self.QPD_max
        QPD_cut[QPD_cut < -self.QPD_max] = -self.QPD_max        
        
        N_lp = int(f_sample/f_drive)+1
        self.t_lp = running_mean(t, N_lp)
        self.QPD_lp = running_mean(QPD_cut, N_lp) 

        self.t = t[int(N_lp/2):-int(N_lp/2)]
        self.QPD = QPD[int(N_lp/2):-int(N_lp/2)] - self.QPD_lp
            
        # Fit Sine       
        p0 = [50, 0, 0]
        lb = (0, -pi, -10)
        ub = (100, pi, 10)        
        p, cov = curve_fit(sine, self.t[np.abs(self.QPD) < self.QPD_max], self.QPD[np.abs(self.QPD) < self.QPD_max], p0, bounds = (lb, ub))  
        print(p)         
                     
        # Fit and subtract Trapzoid   
        p0 = [20*p[0], p[1], p[2], 1e-4]
        lb = (4*p[0], p[1]-0.2, -1, 1e-8)
        ub = (100*p[0], p[1]+0.2, 1, 1e-2)
        p, cov = curve_fit(trapzoid, self.t[np.abs(self.QPD) < self.QPD_max], self.QPD[np.abs(self.QPD) < self.QPD_max], p0, bounds = (lb, ub))          
        self.QPD_fit = trapzoid(self.t, p[0], p[1], p[2], p[3])   
        print(p)              
        
        self.dQPD = self.QPD - self.QPD_fit
        
        # Subtract low-frequency noise
        dQPD_cut = np.array(self.dQPD)
        self.dQPD_max = 3*np.median(np.abs(self.dQPD))        
        dQPD_cut[dQPD_cut > self.dQPD_max] = self.dQPD_max
        dQPD_cut[dQPD_cut < -self.dQPD_max] = -self.dQPD_max        
        
        N_lp = int(f_sample/f_drive/4)+1
        self.dt_lp = running_mean(self.t, N_lp)
        self.dQPD_lp = running_mean(dQPD_cut, N_lp) 

        self.ddQPD = self.dQPD[int(N_lp/2):-int(N_lp/2)] - self.dQPD_lp
                                     
        self.Force = self.ddQPD * stiffness_pN2nm[1]                                             

    def find_binding(self):
        print(self.name)

                          
                                                                              
    def plot_traces(self, path):         
        print("%s. Plotting traces ... " % (self.name))
        n = int(self.fs * N_window / f_drive)    # Number of data in a time block
        m = int(self.N / n)           # Number of block

        if m == 0:
            print("Time window (%.2f s) is too long... \n" %(t_window))
            n = self.N
            m = 1


I need to make them all the same number for synchronized plotting. 

        t = self.t[:n*m].reshape((m,n))    
        QPD = self.QPD[:n*m].reshape((m,n)) 
        QPD_fit = self.QPD_fit[:n*m].reshape((m,n)) 
        dQPD = self.dQPD[:n*m].reshape((m,n)) 
#        dQPD_fit = self.dQPD_fit[:n*m].reshape((m,n))
        dt_lp = self.dt_lp[:n*m].reshape((m,n)) 
        dQPD_lp = self.dQPD_lp[:n*m].reshape((m,n)) 
        ddQPD = self.ddQPD[:n*m].reshape((m,n))         
        PZT = self.PZTy[:n*m].reshape((m,n))     
        F = self.Force[:n*m].reshape((m,n))     

        for i in range(m):                                                                                            
                                                                                                                                                                                                       
            fig = plt.figure(i, figsize = (20, 10), dpi=300) 
        
            sp = fig.add_subplot(411)
            sp.plot(t[i], PZT[i], 'k', lw=1)                                                                               
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)               
            sp.set_ylabel('PZT (nm)')           
        
            sp = fig.add_subplot(412)
            sp.plot(t[i], QPD[i], 'k', lw=1)     
            sp.plot(t[i], QPD_fit[i], 'r', lw=1)                                                                            
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)  
            sp.axhline(y=self.QPD_max, color='k', linestyle='dashed', lw=1)  
            sp.axhline(y=-self.QPD_max, color='k', linestyle='dashed', lw=1)                        
            sp.set_ylabel('QPD (nm)')        
            
            sp = fig.add_subplot(413)
            sp.plot(t[i], dQPD[i], 'k', lw=1)   
            sp.plot(dt_lp[i], dQPD_lp[i], 'r', lw=1)                                                                                           
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)   
            sp.axhline(y=self.dQPD_max , color='k', linestyle='dashed', lw=1)               
            sp.axhline(y=-self.dQPD_max , color='k', linestyle='dashed', lw=1)              
            sp.set_ylabel('dQPD (nm)')                      
            
            sp = fig.add_subplot(414)
            sp.plot(dt_lp[i], ddQPD[i], 'k', lw=1)                                                                               
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)              
            sp.set_ylabel('ddQPD (nm)')                           
                                                     
                                                                                       
            fig_name = self.name[:-5] + '_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)     




    def find_events(self): 
        t = self.t
        QPD = self.QPDy
        dQPD = self.dQPD     
        A = self.dQPD_A
        PZT = self.PZT_fit
        ib0 = self.ib0
        iu0 = self.iu0      
        
        if len(ib0) == 0:
            print("No event.")
            return 
        else:
            print("%s. Fitting events ... " %(self.name))
        
        self.events = []        
        for i in range(len(ib0)):
            event = Event(self.name[:-5]+'_event_'+str(i))          
            j = range(max(ib0[i]-self.T*10, 0), min(iu0[i]+self.T*10, len(t)))
            if event.fit_step(t[j], A[j], t[ib0[i]], t[iu0[i]], np.median(A[ib0[i]:iu0[i]]), 0):
                if event.fit_QPD(t[j], QPD[j], dQPD[j], event.tb, event.tu):
                    event.fit_PZT(t[j], PZT[j], event.tb, event.tu)
                    self.events.append(event)

    def plot_events(self, path):
        print("%s. Plotting evens ... " % (self.name))
        for i in range(len(self.events)):
            e = self.events[i]
            
            # Plot
            fig = plt.figure(i, figsize = (20, 10), dpi=300)   
            
            sp = fig.add_subplot(311)
            sp.plot(e.t, e.QPD, 'k', lw=1)    
            sp.plot(e.t_fit, e.QPD_fit, 'b', lw=2)     
            sp.plot(e.t_fit[e.ib], e.QPD_fit[e.ib], 'r', lw=2)                                                                                
            sp.axvline(x=e.tb, c='k', linestyle='dotted', lw=1)   
            sp.axvline(x=e.tu, c='k', linestyle='dotted', lw=1) 
            sp.axhline(y=e.offset_u, xmin = 0, xmax = (e.tb-e.t[0])/(e.t[-1]-e.t[0]), c='b', ls='dashed', lw=1)
            sp.axhline(y=e.offset_b, xmin = (e.tb-e.t[0])/(e.t[-1]-e.t[0]), xmax = (e.tu-e.t[0])/(e.t[-1]-e.t[0]), c='r', ls='dashed', lw=1)
            sp.axhline(y=e.offset_u, xmin = (e.tu-e.t[0])/(e.t[-1]-e.t[0]), xmax = 1, c='b', ls='dashed', lw=1)
            sp.set_xlim(e.t[0], e.t[-1])   
            sp.set_ylabel('QPD (nm)')   
            sp.set_title("(RMSD = %.2f nm)" %(e.QPD_RMSD))           
      
#            sp = fig.add_subplot(312)
#            sp.plot(e.t, e.dQPD, 'k', lw=1)                                                                                  
#            sp.axvline(x=e.tb, c='k', linestyle='dotted', lw=1)   
#            sp.axvline(x=e.tu, c='k', linestyle='dotted', lw=1) 
#            sp.set_xlim(e.t[0], e.t[-1]) 
#            sp.set_ylabel('dQPD (nm)')             
                                    
            sp = fig.add_subplot(312)   
            sp.plot(e.t, e.A, 'k', lw=1)              
            sp.plot(e.t_fit, e.A_fit, 'b', lw=2)     
            sp.plot(e.t_fit[e.ib], e.A_fit[e.ib], 'r', lw=2)       
            sp.axhline(y=self.Am_cutoff, color='k', ls='dashed', lw=1) 
            sp.axvline(x=e.tb, c='k', ls='dotted', lw=1)   
            sp.axvline(x=e.tu, c='k', ls='dotted', lw=1)   
            sp.set_xlim(e.t[0], e.t[-1])
            sp.set_ylabel('Amplitude (nm)')
            sp.set_title("(RMSD = %.2f nm)" %(e.A_RMSD))  
            
            sp = fig.add_subplot(313)   
            sp.plot(e.t, e.PZT, 'k', lw=1)              
            sp.plot(e.t_fit, e.PZT_fit, 'b', lw=2) 
            sp.plot(e.t_fit[e.ib], e.PZT_fit[e.ib], 'r', lw=2)                     
            sp.axvline(x=e.tb, c='k', ls='dotted', lw=1)   
            sp.axvline(x=e.tu, c='k', ls='dotted', lw=1)   
            sp.set_xlim(e.t[0], e.t[-1])
            sp.set_ylabel('PZT (nm)')                                                     

#            sp = fig.add_subplot(313)    
#            PZT_fit2 = sine(t_fit, self.PZT_A, self.PZT_phi, self.PZT_b)                 
#            sp.plot(t_fit, PZT_fit2, 'r', linewidth=2)  
#            sp.plot(t[i], PZT_fit[i], 'ko', ms=2)   
#            sp.axhline(y=0, color='k', linestyle='dashed', linewidth=1)             
#            sp.axvline(x=tb_fit, color='k', linestyle='dotted', linewidth=1)   
#            sp.axvline(x=tu_fit, color='k', linestyle='dotted', linewidth=1)              
#            sp.set_ylabel('PZT (nm)') 
     
            sp.set_xlabel('Time (s)')      
            fig_name = e.name + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)    

             
class Molecule:
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
            
    def read_data(self):
        self.data = []
        for name in self.data_list:
            data = Data(name)
            data.read_data()
            self.data.append(data)
            
    def transform(self):
        for i in range(len(self.data)):
            self.data[i].wavelet_transformation()

    def find_binding(self):
        for i in range(len(self.data)):
            self.data[i].find_binding()

    def plot_traces(self):
        path = make_folder('Traces')
        for i in range(len(self.data)):
            self.data[i].plot_traces(path)
            
    def find_events(self):       
        for i in range(len(self.data)):
            self.data[i].find_events()

    def plot_events(self):
        path = make_folder('Events') 
        for i in range(len(self.data)):
            self.data[i].plot_events(path)
                                                       
def main():
    mol = Molecule()
    mol.read_data()
#    mol.find_events()  
    mol.plot_traces()
#    mol.plt_events()

#    mol.transform()
#    mol.find_binding()
#    mol.plot_traces()    
#    mol.find_events()
#    mol.plot_events()

if __name__ == "__main__":
    sys.exit(main())

"""


"""




