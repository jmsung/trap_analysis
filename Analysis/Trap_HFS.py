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
import sys

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
t_block = 30
n_avg = 20

t_short = 1.0/f_drive/2
t_long = 0.5
A_RMSD_cut = 10
QPD_RMSD_cut = 10
Abu_cut = 1.2
outlier_cut = 6

###############################################

def step(t, tb, tu, Ab, Au, s1, s2):
    return (Ab-Au) * (erf(s1*(t-tb)) - erf(s2*(t-tu)))/2 + Au
              
def sine(t, A, ph, b): # Sine function
    return A * np.sin(2*pi*f_drive*t - ph) + b    

def exp(F, t0, dF):
    return t0*np.exp(-F/dF)

def running_mean(x, N = n_avg): # Running mean
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

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


class Event(object):
    def __init__(self, name):
        self.name = name

    def fit_step(self, t, A, tb, tu, Ab, Au):
        p0 = [tb, tu, Ab, Au, 1000, 1000]
        lb = (t[1], tb, Ab/2, 0, 800, 800)
        ub = (tu, t[-2], Ab*10, 50, 1200, 1200)

        try:        
            p, cov = curve_fit(step, t, A, p0, bounds = (lb, ub))  
            A_fit = step(t, p[0], p[1], p[2], p[3], p[4], p[5])
            self.A_RMSD = (np.mean((A - A_fit)**2))**0.5
                 
            if self.A_RMSD > A_RMSD_cut:  
#                print("A_RMSD = %.2f" %(self.A_RMSD))
                return False                
                
            self.ts = p[1] - p[0]
            if (self.ts < t_short):
#                print("Too short event. ts = %.2f ms" %(self.ts*1000))
                return False
            elif (self.ts > t_long):
#                print("Too long event. ts = %.2f ms" %(self.ts*1000))
                return False      
            else:
                pass             
                
            self.t = t  
            self.A = A
            self.t_fit = np.arange(t[0], t[-1], 0.0001) 
            self.A_fit = step(self.t_fit, p[0], p[1], p[2], p[3], p[4], p[5])                 
            self.tb = p[0]
            self.tu = p[1]
            self.Ab = p[2]
            self.Au = p[3]   
            self.ts = self.tu - self.tb       
            return True           
            
        except:
            print("Unexpected error", sys.exc_info()[0])
            return False

    def fit_QPD(self, t, QPD, dQPD, tb, tu):

        try:
            # Bound state
            ib = np.array(t >= tb, dtype=bool) & np.array(t <= tu, dtype=bool)    
            ib_fit = np.array(self.t_fit >= tb, dtype=bool) & np.array(self.t_fit <= tu, dtype=bool)            
            p0_b = [A_drive/2, 0, 0]
            lb_b = (0, -pi, -A_drive)
            ub_b = (A_drive, pi, A_drive)
            p_b, cov = curve_fit(sine, t[ib], QPD[ib], p0_b, bounds = (lb_b, ub_b))
            QPD_b = sine(self.t[ib], p_b[0], p_b[1], p_b[2])  

            # Unbound state
            iu = ~ib
            iu_fit = ~ib_fit
            p0_u = [A_drive/4, 0, 0]
            lb_u = (0, -pi, -A_drive)
            ub_u = (A_drive, pi, A_drive)
            p_u, cov = curve_fit(sine, t[iu], QPD[iu], p0_u, bounds = (lb_u, ub_u))
            QPD_u = sine(self.t[iu], p_u[0], p_u[1], p_u[2])  

            if p_b[0]/p_u[0] < Abu_cut:
#                print("A_b / A_u = %.2f" %(p_b[0] / p_u[0]))
                return False

            QPD_fit = np.zeros(len(t))
            QPD_fit[ib] = QPD_b
            QPD_fit[iu] = QPD_u
            self.QPD_RMSD = (np.mean((QPD - QPD_fit)**2))**0.5       

            if self.QPD_RMSD > QPD_RMSD_cut:
#                print("QPD RMSD = %.2f" %(self.QPD_RMSD))
                return False
            
            self.dQPD = dQPD
            self.QPD = QPD
            self.QPD_fit = np.zeros(len(self.t_fit))
            self.QPD_fit[ib_fit] = sine(self.t_fit[ib_fit], p_b[0], p_b[1], p_b[2]) 
            self.QPD_fit[iu_fit] = sine(self.t_fit[iu_fit], p_u[0], p_u[1], p_u[2]) 
            self.offset_b = p_b[2]
            self.offset_u = p_u[2]
            self.Ph_b = p_b[1]
            self.Ph_u = p_u[1]
            self.QPD_Ab = p_b[0]
            self.QPD_Au = p_u[0]
            self.ib = ib_fit
            self.Fs = self.offset_b * stiffness_pN2nm[1]
    
            X = sine(self.t_fit, p_b[0], p_b[1], p_b[2])
            V = sine(self.t_fit, p_b[0], p_b[1] + pi/2, p_b[2])
            i_b = np.abs(self.t_fit - tb).argmin() 
            Xb = X[i_b]  
            Vb = V[i_b] 
            Pb = atan(2*pi*f_drive * Xb / Vb) - p_b[1]
            self.QPD_Pb = np.mod(Pb, 2*pi)
    
            return True
            
        except:
            print("Unexpected error", sys.exc_info()[0])
            return False

    def fit_PZT(self, t, PZT, tb, tu):
        try:
            p0 = [A_drive*0.8, 0, 0]
            lb = (A_drive*0.5, -pi, -10)
            ub = (A_drive, pi, 10)
            p, cov = curve_fit(sine, t, PZT, p0, bounds=(lb, ub))
            self.PZT = PZT
            self.PZT_fit = sine(self.t_fit, p[0], p[1], p[2])
            self.PZT_vel = sine(self.t_fit, p[0], p[1]+pi/2, 0)*(2*pi*f_drive)
            i_b = np.abs(self.t_fit - tb).argmin()
            i_u = np.abs(self.t_fit - tu).argmin()
            self.Vb = self.PZT_vel[i_b]
            self.Vu = self.PZT_vel[i_u]     
            self.Xb = self.PZT_fit[i_b]  
            self.Xu = self.PZT_fit[i_u]  
            Pb = atan(2*pi*f_drive * self.Xb / self.Vb) - p[1]
            Pu = atan(2*pi*f_drive * self.Xu / self.Vu) - p[1]        
            self.PZT_Pb = np.mod(Pb, 2*pi)
            self.PZT_Pu = np.mod(Pu, 2*pi)    
        
            t = np.linspace(0, 1/f_drive, 10000) 
            PZT1 = sine(t, p[0], 0, p[2])
            QPD1 = sine(t, self.QPD_Au, self.Ph_u-p[1], self.offset_u)  
            PZT_QPD = PZT1-QPD1
            self.PZT_QPD = PZT_QPD / np.max(PZT_QPD)

        except:
            print("Unexpected error", sys.exc_info()[0])
            return False    
            
        
class Data(object):
    def __init__(self, name):
        self.name = name

    def read_data(self):          
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
        self.t = channels[0].time_track()
        self.QPDy = (channels[1].data - np.median(reject_outliers(channels[1].data))) * QPD_nm2V[1]     
        self.PZTy = -(channels[4].data - np.median(channels[4].data)) * PZT_nm2V[1]
#        self.QPDs = (channels[2].data)

        self.T = int(self.fs / f_drive)        # Oscillation period in number
#        self.N_os = int(self.N / self.T)      # Number of oscillation
#        self.t = self.t[:self.T*self.n_os]      
#        self.QPDy = self.QPDy[:self.T*self.n_os]     
#        self.PZTy = self.PZTy[:self.T*self.n_os]    

    def wavelet_transformation(self):
        t = self.t
        QPD = self.QPDy
        PZT = self.PZTy
        T = self.T

        # QPD fitting (Amp, Phase, Bg)
        p0 = [A_drive/4, 0, 0]
        lb = (0, -pi, -A_drive/2)
        ub = (A_drive/2, pi, A_drive/2)
        QPD_param, QPD_cov = curve_fit(sine, t, QPD, p0, bounds=(lb, ub)) 
        self.QPD_A0 = QPD_param[0]
        self.QPD_P0 = QPD_param[1] 
        self.QPD_B0 = QPD_param[2] 
        self.QPD0 = sine(t, self.QPD_A0, self.QPD_P0, self.QPD_B0)       
        self.dQPD = QPD - self.QPD0

        # PZT fitting (Amp, Phase, Bg)
        p0 = [A_drive*0.5, 0, 0]
        lb = (0, -pi, -10)
        ub = (A_drive, pi, 10)
        PZT_param, PZT_cov = curve_fit(sine, t[:min(10000,len(t))], PZT[:min(10000,len(t))], p0, bounds=(lb, ub))
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

    def find_binding(self):
        print(self.name)
        # Running_mean
        self.tm = running_mean(self.t)
        self.Am = running_mean(self.dQPD_A)
        self.Am_cutoff, self.im_b = find_outliers(self.Am)

        # Find binding and unbinding time       
        tb = []
        tu = []
        for i in range(len(self.tm)-1):
            if (self.im_b[i] == False) & (self.im_b[i+1] == True):
                tb.append(self.tm[i])
            elif (self.im_b[i] == True) & (self.im_b[i+1] == False):
                tu.append(self.tm[i])
            else:
                continue
           
        if len(tb) < 2 | len(tu) < 2:
            print("No event is found. \n")
            return 
    
        # Remove incomplete events
        if tu[0] < tb[0]:
            tu = tu[1:]
        if len(tb) > len(tu):
            tb = tb[:-1]

        if len(tb) != len(tu):
            print("Error: incomplete binding and unbinding events")
            return   
     
        # Remove too short or too long events
        tb_ex = []
        tu_ex = []       
        for i in range(len(tb)):
            ts = tu[i] - tb[i]
            if (ts < t_short):
#                print("Too short event. ts = %.2f ms" %(ts*1000)) 
                tb_ex.append(tb[i])
                tu_ex.append(tu[i])                     
            elif (ts > t_long):
#                print("Too long event. ts = %.2f ms" %(ts*1000)) 
                tb_ex.append(tb[i])
                tu_ex.append(tu[i])
            else:
                continue
        if len(tb_ex) > 0:
            for i in range(len(tb_ex)):
                tb.remove(tb_ex[i])
                tu.remove(tu_ex[i])
        print("%d events are found. \n" %(len(tb)))
        
        self.ib0 = np.zeros(len(tb), dtype=int)
        self.iu0 = np.zeros(len(tu), dtype=int)        
        self.bind0 = np.zeros(len(self.t), dtype=bool)
        
        for i in range(len(tb)):      
            self.ib0[i] = (np.abs(self.t - tb[i])).argmin()
            self.iu0[i] = (np.abs(self.t - tu[i])).argmin()
            self.bind0[self.ib0[i]:self.iu0[i]] = True
       
                     
    def plot_traces(self, path):         
        print("%s. Plotting traces ... " % (self.name))
        n = int(self.fs * t_block)    # Number of data in a time block
        m = int(self.N / n)           # Number of block

        if m == 0:
            print("Time block (%.2f s) is too long... \n" %(t_block))
            n = self.N
            m = 1

        t = self.t[:n*m].reshape((m,n))    
        Q = self.dQPD[:n*m].reshape((m,n))          
        A = self.dQPD_A[:n*m].reshape((m,n)) 
        b = self.bind0[:n*m].reshape((m,n)) 

        for i in range(m):                                                                           
            tm = running_mean(t[i])
            Am = running_mean(A[i])
            Am_cutoff, im_b = find_outliers(Am)
                                                                                                                                                                                            
            fig = plt.figure(i, figsize = (20, 10), dpi=300) 
            
            sp = fig.add_subplot(211)
            sp.plot(t[i], Q[i], 'k', lw=1)    
            sp.plot(t[i][b[i]], Q[i][b[i]], 'r.', ms=4)                                                                    
            sp.axhline(y=0, color='g', linestyle='dashed', lw=1)   
            sp.set_ylabel('dQPD (nm)')            
            
#            sp = fig.add_subplot(312)   
#            sp.plot(t[i], A[i], 'k', lw=1)     
#            sp.plot(t[i][b[i]], A[i][b[i]], 'r.', ms=4)                                          
#            sp.axhline(y=np.median(reject_outliers(A)), color='g', linestyle='dashed', lw=1)
#            sp.set_ylabel('Amplitude (nm)') 
            
            sp = fig.add_subplot(212)   
            sp.plot(tm, Am, 'k', lw=1)          
            sp.plot(tm[im_b], Am[im_b], 'r.', ms=4)                           
            sp.axhline(y=np.median(reject_outliers(A)), color='g', linestyle='dashed', lw=1)
            sp.axhline(y=self.Am_cutoff, color='b', linestyle='dashed', lw=1)                                      
            sp.set_ylabel('Amplitude_Avg (nm)')
        
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
        

    def show_results(self):
        path = make_folder('Results') 
        Fs = []
        ts = []
        Vb = []
        Vu = []
        Xb = []
        Xu = []
        Ab = []
        PZT_Pb = []
        PZT_Pu = [] 
        PZT_QPD = []      

        for i in range(len(self.data)):
            events = self.data[i].events
            for j in range(len(events)):
                e = events[j]
                Fs.append(e.Fs)
                ts.append(e.ts)
                Vb.append(e.Vb)
                Vu.append(e.Vu)
                Xb.append(e.Xb)
                Xu.append(e.Xu)
                Ab.append(e.QPD_Ab)
                PZT_Pb.append(e.PZT_Pb)
                PZT_Pu.append(e.PZT_Pu)
                PZT_QPD.append([e.PZT_QPD])

        print("\nFinal result:")
        print("%d events are detected." %(len(Fs))) 
        if len(Fs) == 0:
            return

        Fs = np.array(Fs)
        ts = np.array(ts)*1000 # [ms]
        Vb = np.array(Vb)/1000 # [nm/ms]
        Vu = np.array(Vu)/1000 # [nm/ms]
        Xb = np.array(Xb)
        Xu = np.array(Xu)
        Ab = np.array(Ab)
        PZT_Pb = np.array(PZT_Pb)
        PZT_Pu = np.array(PZT_Pu) 
        PZT_QPD = np.array(PZT_QPD)
        PZT_QPD = np.mean(PZT_QPD, axis=0)
        PZT_QPD = PZT_QPD.flatten()
        dPZT_QPD = PZT_QPD[1:]-PZT_QPD[:-1]
        dPZT_QPD = -(dPZT_QPD/np.max(dPZT_QPD))*0.5       

        # Get Force dependent mean dwell time
        Fm = np.linspace(min(Fs), max(Fs), 10)
        dm = (Fm[1]-Fm[0])*0.5
        tm = np.zeros(len(Fm))
#        tm_s = np.zeros(len(Fm))
        
        for i in range(len(Fm)):
            ix = np.array(Fs>Fm[i]-dm, dtype=bool) & np.array(Fs<=Fm[i]+dm, dtype=bool)
            if len(ix) > 0:
                tm[i] = np.mean(ts[ix])
#            tm_s[i] = np.std(ts[ix]) / (len(ts[ix]))**0.5

        if np.mean(tm[Fm>0]) < np.mean(tm[Fm<0]):
            Fm = -Fm
    
        params, cov = curve_fit(exp, Fm[tm>0], tm[tm>0], p0=[50, 2])
        t0, dF = params
        Fm_fit = np.linspace(min(Fs), max(Fs), 1000)
        tm_fit = exp(Fm_fit, t0, dF)


        # Figure: Fv
        fig = plt.figure('FV', figsize = (20, 10), dpi=300)      
        sp = fig.add_subplot(121)   
        sp.plot(Fs, ts, 'ko', ms=10, alpha=0.5)              
        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
        sp.set_ylim(0, max(ts)*1.1)
        sp.set_xlabel('Force (pN)')  
        sp.set_ylabel('Dwell time (ms)')
        sp.set_title("Force vs Dwell time")   

        sp = fig.add_subplot(122)   
        sp.plot(Fm, tm, 'ko', ms=10)     
        sp.plot(Fm_fit, tm_fit, 'r', lw=2)           
        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
        sp.set_ylim(0, max(tm)*1.1)
        sp.set_xlabel('Mean force (pN)')  
        sp.set_ylabel('Mean dwell time (ms)')
        sp.set_title("Mean force vs Mean dwell time")  

        fig.savefig(os.path.join(path, 'F-V.png'))
        plt.close(fig) 
        
        # Figure: VKb
        fig = plt.figure('P-B', figsize = (20, 10), dpi=300)    
                      
        sp = fig.add_subplot(121)   
        Pb_bins = np.linspace(0, 2*pi, 9)
        sp.hist(PZT_Pb, bins=Pb_bins, histtype='stepfilled', lw=2)
        sp.axvline(x=pi*0.5, c='k', ls='dotted', lw=1) 
        sp.axvline(x=pi*1.0, c='k', ls='dotted', lw=1) 
        sp.axvline(x=pi*1.5, c='k', ls='dotted', lw=1)         
        sp.set_xlim(0, 2*pi)                          
        sp.set_xticks([0, pi, 2*pi])
        sp.set_xlabel('Phase @ binding')        
        sp.set_ylabel('Count')  
        
        sp = fig.add_subplot(122)   
        sp.plot(np.linspace(0, 2*pi, 9999), dPZT_QPD, 'b', lw=2)
        sp.axvline(x=pi*0.5, c='k', ls='dotted', lw=1) 
        sp.axvline(x=pi*1.0, c='k', ls='dotted', lw=1) 
        sp.axvline(x=pi*1.5, c='k', ls='dotted', lw=1)  
        sp.axhline(y=0, c='k', ls='dotted', lw=1)                 
        sp.set_xlim(0, 2*pi)                          
        sp.set_xticks([0, pi, 2*pi])
        sp.set_xlabel('Phase @ binding')        
        sp.set_ylabel('Negative Velocity')          

        fig.savefig(os.path.join(path, 'P-B.png'))
        plt.close(fig)                          
                                                                           
#        sp = fig.add_subplot(122)   
#        sp.hist(PZT_Pu, 10, density=True, facecolor='b', alpha=0.25)                           
#        sp.set_xlabel('Phase @ unbinding')  
#        sp.set_ylabel('Count')  
                               
#        sp = fig.add_subplot(132)   
#        sp.hist(Vb, 10)              
#        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#        sp.set_xlabel('Velocity @ binding (nm/ms)')  
#        sp.set_ylabel('Count')
#        sp.set_title("[%.2f : %.2f]" %(sum(Vb<0)/len(Vb), sum(Vb>0)/len(Vb))) 
                                                                        
#        sp = fig.add_subplot(133)   
#        sp.hist(Xb, 10)              
#        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#        sp.set_xlabel('Position @ binding (nm)')  
#        sp.set_ylabel('Count')
#        sp.set_title("[%.2f : %.2f]" %(sum(Xb<0)/len(Xb), sum(Xb>0)/len(Xb))) 



#        sp = fig.add_subplot(222)   
#        sp.hist(Vu/1000, 10)              
#        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#        sp.set_xlabel('Velocity @ unbinding (nm/ms)')  
#        sp.set_ylabel('Count')
#        sp.set_title("[%.2f : %.2f]" %(sum(Vu<0)/len(Vu), sum(Vu>0)/len(Vu))) 

#        sp = fig.add_subplot(224)   
#        sp.hist(Xu, 10)              
#        sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#        sp.set_xlabel('Position @ unbinding (nm)')  
#        sp.set_ylabel('Count')
#        sp.set_title("[%.2f : %.2f]" %(sum(Xu<0)/len(Xu), sum(Xu>0)/len(Xu))) 


        
    def fitting(self):
        pass

                                                       
def main():
    mol = Molecule()
    mol.calibrate()
    mol.read_data()
    mol.transform()
    mol.find_binding()
    mol.plot_traces()    
    mol.find_events()
    mol.plot_events()
    mol.show_results()

    

if __name__ == "__main__":
    main()


"""""""""
To-do

Exp decay: Negative if mean time increases

event finding based on two rounds? 1) both QPD_A and QPD_P, 2) then use dQPD_A?

stiffness

Save sample, slide, mol name and display in the plots

lowpass > subtract > std > cutoff based on std

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

================================================================================

Done




"""""""""





