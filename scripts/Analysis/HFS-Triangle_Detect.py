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
from scipy.optimize import curve_fit
import sys 
from trap_func import sine, trapzoid, running_mean, reject_outliers, make_folder, triangle
import Calibration


### User input ##################################

power = 100          # Laser power
f_drive = 10         # PZT oscillation frequency (Hz)
A_drive = 1000       # PZT oscillation amplitude (nm)

f_sample = 2000      # Sampling frequency
dt = 1/f_sample

N_window = 20        # Number of oscillation per window
t_window = N_window  / f_drive
N_plot = 20          # Number of plots for traces and events
F_SNR = 5            # Binding event cutoff (SNR)
#F_std_cut = 1.5

# Calibration
R = 430              # Bead radius (nm)
f_sample_cal = 20000 # Sampling frequency for calibration (20 kHz)
f_lp_cal = 20000     # Lowpass filter frequency for calibration (20 kHz)
fd_cal = 50          # Oscillation frequency (50 Hz)
Ad_cal = 50          # Oscillation amplitude (50 nm)
height_cal = 500     # Calibration above the surface (500 nm)


###############################################

class Event:
    def __init__(self, b, u, t, PZT, QPD, QPD_fit, Force, F_cut):
        N = int(f_sample/f_drive/2)

        cond1 = u - b > N
        cond2 = u - b < 3        
        cond3 = len(Force[b:u]) - np.argmax(abs(Force[b:u])) > 3
        cond4 = (max(Force[b-N:u+N]) > F_cut) & (min(Force[b-N:u+N]) < -F_cut) 

        if (cond1 | cond2 | cond3 | cond4):
            self.real = False
            return

        self.b = b
        self.u = u
        self.t = t[b-N:u+N]
        self.PZT = PZT[b-N:u+N]
        self.QPD = QPD[b-N:u+N]
        self.QPD_fit = QPD_fit[b-N:u+N]
        self.Force = Force[b-N:u+N]   

#        self.F_std = np.std(self.Force)
#        if self.F_std > F_std_cut:
#            self.real = False
#            return
        
        if np.mean(Force[b:u]) > 0:
            self.direction = 1
        else:
            self.direction = -1     
        
        self.Force_max = self.direction * max(abs(Force[b-1:u+1]))  

        self.real = True    
         
class Data:
    def __init__(self, name, QPD_nm2V, stiffness_pN2nm):
        self.name = name
        self.QPD_nm2V = QPD_nm2V
        self.stiffness_pN2nm = stiffness_pN2nm

    def read_data(self):          
        tdms_file = nptdms.TdmsFile(self.name)      # Reads a tdms file.
        root_object = tdms_file.object()     # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap"                             
        channels = tdms_file.group_channels(group_name)                    
        self.dt = channels[0].properties[u'wf_increment']  # Sampling time
        self.fs = int(1.0/self.dt)                         # Sampling frequency
        self.N = len(channels[0].time_track())             # Number of time points
        print("Sampling rate: %d Hz" % self.fs)   
        print("Data size: %d sec \n" % int(self.N*self.dt))  

        PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor

        # Read data
        t0 = channels[0].time_track()
        QPDy = (channels[1].data - np.median(reject_outliers(channels[1].data))) * self.QPD_nm2V[1]     
        PZT0 = -(channels[4].data - np.median(channels[4].data)) * PZT_nm2V[1]
        QPDs = (channels[2].data)
        QPD0 = QPDy / QPDs
    
        # Fit PZT
        p0 = [f_drive, A_drive, 0, 0]
        lb = (f_drive-0.01, A_drive*0.9, -np.pi, -100)
        ub = (f_drive+0.01, A_drive*1.1, np.pi, 100)        
        p, cov = curve_fit(triangle, t0, PZT0, p0, bounds = (lb, ub))  
        PZT_fit0 = triangle(t0, p[0], p[1], p[2], p[3]) 
#        print("PZT fit = ", p)         

        # Subtract low-frequency noise
        QPD_cut = np.array(QPD0)        
        self.QPD_max = 2*np.median(np.abs(QPD_cut))
        QPD_cut[QPD_cut > self.QPD_max] = self.QPD_max
        QPD_cut[QPD_cut < -self.QPD_max] = -self.QPD_max        
        
        N_lp = int(f_sample/f_drive)
        if N_lp % 2 == 0:
            N_lp += 1        
        
        self.t_lp = running_mean(t0, N_lp)
        self.QPD_lp = running_mean(QPD_cut, N_lp) 

        t1 = t0
        QPD1 = QPD0 - self.QPD_lp
        PZT1 = PZT0
        PZT_fit1 = PZT_fit0   
    
        # Fit Sine       
        p0 = [f_drive, 50, 0, 0]
        lb = (f_drive-0.01, 0, -np.pi, -10)
        ub = (f_drive+0.01, 100, np.pi, 10)        
        p, cov = curve_fit(sine, t1[np.abs(QPD1) < self.QPD_max], QPD1[np.abs(QPD1) < self.QPD_max], p0, bounds = (lb, ub))          
               
        # Fit and subtract Trapzoid   
        p0 = [f_drive, 20*p[1], p[2], p[3], 1e-4]
        lb = (f_drive-0.01, 4*p[1], p[2]-0.2, p[3]-1, 1e-8)
        ub = (f_drive+0.01, 100*p[1], p[2]+0.2, p[3]+1, 1e-2)
        p, cov = curve_fit(trapzoid, t1[np.abs(QPD1) < self.QPD_max], QPD1[np.abs(QPD1) < self.QPD_max], p0, bounds = (lb, ub))          
        QPD_fit1 = trapzoid(t1, p[0], p[1], p[2], p[3], p[4])   
#        print("QPD_fit = ", p)              
        
        dQPD1 = QPD1 - QPD_fit1
        
        # Subtract low-frequency noise
        dQPD_cut = np.array(dQPD1)
        self.dQPD_max = 2*np.median(np.abs(dQPD1))        
        dQPD_cut[dQPD_cut > self.dQPD_max] = self.dQPD_max
        dQPD_cut[dQPD_cut < -self.dQPD_max] = -self.dQPD_max        
        
        N_lp = int(f_sample/f_drive/4)
        if N_lp % 2 == 0:
            N_lp += 1
        
        dQPD_lp = running_mean(dQPD_cut, N_lp) 
        dQPD2 = dQPD1                 
        ddQPD = dQPD2 - dQPD_lp
                   
        self.t = running_mean(t1, N_lp)   
        self.QPD = QPD1
        self.QPD_fit = QPD_fit1
        self.PZT = PZT1
        self.PZT_fit = PZT_fit1                        
        self.Force = ddQPD * self.stiffness_pN2nm[1]       

    def find_events(self): 
        print("\n%s. Finding events ... " % (self.name))

        # Find threshold of force
        self.F_cut = F_SNR*np.median(np.abs(self.Force))   
                                
        self.signal_p = (self.Force > self.F_cut) & (self.QPD_fit > 0)
        self.signal_n = (self.Force < -self.F_cut) & (self.QPD_fit < 0) 
        self.signal = self.signal_p + self.signal_n                  
                 
        # Collect indeces at binding & unbinding                 
        b = []
        u = []     
        
        for i in range(len(self.Force)-1):
            if (self.signal[i] == False) & (self.signal[i+1] == True):
                b.append(i)
            elif (self.signal[i] == True) & (self.signal[i+1] == False):
                u.append(i)
            else:
                continue
             
        self.events = []                 
             
        if len(b)*len(u) == 0:
            print("No event ...")
            return
            
        # Consider only completed events
        if b[0] > u[0]:
            u = u[1:]
        if len(b) > len(u):
            b = b[:-1]
            
        if len(b)*len(u) == 0:
            print("No event ...")
            return            
                                    
        for i in range(len(b)):
            event = Event(b[i], u[i], self.t, self.PZT, self.QPD, self.QPD_fit, self.Force, self.F_cut)
            if event.real == True:
                self.events.append(event)
            else:
                self.signal_p[b[i]-1:u[i]+1] = False
                self.signal_n[b[i]-1:u[i]+1] = False

        self.signal = self.signal_p + self.signal_n   

        print("Found events #", len(self.events))

    def plot_events(self, data_num, path):
        print("%s. Plotting events ... " % (self.name))
        for i in range(min(N_plot, len(self.events))):            
            e = self.events[i]
            
            # Plot
            fig = plt.figure(i, figsize = (20, 10), dpi=300)   
            
            sp = fig.add_subplot(311)
            sp.plot(e.t, e.PZT, 'k')
            sp.plot(e.t[e.QPD_fit>0], e.PZT[e.QPD_fit>0], 'r.')
            sp.plot(e.t[e.QPD_fit<0], e.PZT[e.QPD_fit<0], 'b.')            
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)  
            sp.set_ylabel('PZT (nm)')
            sp.set_title('Frequency = %d (Hz), Amplitude = %d (nm)' %(f_drive, A_drive))

            sp = fig.add_subplot(312)
            sp.plot(e.t, e.QPD, 'k')
            sp.plot(e.t[e.QPD_fit>0], e.QPD_fit[e.QPD_fit>0], 'r.')                        
            sp.plot(e.t[e.QPD_fit<0], e.QPD_fit[e.QPD_fit<0], 'b.')   
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1) 
            sp.axhline(y=100, color='k', linestyle='dashed', lw=1) 
            sp.axhline(y=-100, color='k', linestyle='dashed', lw=1) 
            sp.set_ylabel('QDP (nm)')  

            sp = fig.add_subplot(313)
            sp.plot(e.t, e.Force, 'k')
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)  
            sp.axhline(y=self.F_cut*e.direction, color='k', linestyle='dashed', lw=1)   
            sp.axhline(y=-self.F_cut*e.direction, color='k', linestyle='dashed', lw=1)              
            sp.set_ylabel('Force (pN)')    
            sp.set_title('Force_max = %.1f (pN), Force_cut = %.1f' %(e.Force_max, self.F_cut))           
              
            sp.set_xlabel('Time (s)')      
            fig_name = 'Data_' + str(data_num) + '_Event_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)                                                                                                  
                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                          
    def plot_traces(self, data_num, path):         
        print("%s. Plotting traces ... " % (self.name))
        n = int(self.fs * N_window / f_drive)    # Number of data in a time block
        m = int(len(self.t) / n)           # Number of block

        if m == 0:
            print("Time window (%.2f s) is too long... \n" %(t_window))
            n = self.t
            m = 1 

        t = self.t[:n*m].reshape((m,n))    
        QPD = self.QPD[:n*m].reshape((m,n)) 
        QPD_fit = self.QPD_fit[:n*m].reshape((m,n))         
        PZT = self.PZT[:n*m].reshape((m,n))     
        PZT_fit = self.PZT_fit[:n*m].reshape((m,n))           
        F = self.Force[:n*m].reshape((m,n))     
        signal_p = self.signal_p[:n*m].reshape((m,n))  
        signal_n = self.signal_n[:n*m].reshape((m,n))  
#        F_std = self.F_std[:n*m].reshape((m,n))   
            
        for i in range(min(N_plot, m)):                                                                                                      
                                                                                                                                                                                                       
            fig = plt.figure(i, figsize = (20, 10), dpi=300) 
        
            sp = fig.add_subplot(311)
            sp.plot(t[i], PZT[i], 'k', lw=1)     
            sp.plot(t[i][QPD_fit[i]>0], PZT_fit[i][QPD_fit[i]>0], 'r.', ms=2)  
            sp.plot(t[i][QPD_fit[i]<0], PZT_fit[i][QPD_fit[i]<0], 'b.', ms=2)                                                                               
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)               
            sp.set_ylabel('PZT (nm)')               
            sp.set_title('Frequency = %d (Hz), Amplitude = %d (nm)' %(f_drive, A_drive))
        
            sp = fig.add_subplot(312)
            sp.plot(t[i], QPD[i], 'k', lw=1)     
            sp.plot(t[i][QPD_fit[i]>0], QPD_fit[i][QPD_fit[i]>0], 'r.', ms=2)   
            sp.plot(t[i][QPD_fit[i]<0], QPD_fit[i][QPD_fit[i]<0], 'b.', ms=2)                                                                                       
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)  
            sp.axhline(y=100, color='k', linestyle='dashed', lw=1) 
            sp.axhline(y=-100, color='k', linestyle='dashed', lw=1) 
            sp.axhline(y=self.QPD_max, color='k', linestyle='dashed', lw=1)  
            sp.axhline(y=-self.QPD_max, color='k', linestyle='dashed', lw=1)                        
            sp.set_ylabel('QPD (nm)')          
            
            sp = fig.add_subplot(313)
            sp.plot(t[i], F[i], 'k.', lw=1)  
            sp.plot(t[i][signal_p[i]], F[i][signal_p[i]], 'r.', ms=5)    
            sp.plot(t[i][signal_n[i]], F[i][signal_n[i]], 'b.', ms=5)                                                                                                      
            sp.axhline(y=0, color='k', linestyle='dashed', lw=1)            
            sp.axhline(y=self.F_cut, color='k', linestyle='dashed', lw=1)    
            sp.axhline(y=-self.F_cut, color='k', linestyle='dashed', lw=1)    
            sp.set_ylabel('Force (pN)')                               
            sp.set_title('Force cut = %.1f (pN)' %(self.F_cut)) 
                 
#            sp = fig.add_subplot(414)
#            sp.plot(t[i], F_std[i], 'k', lw=1) 
#            sp.axhline(y=self.F_std_m, color='k', linestyle='dashed', lw=1)  
#            sp.axhline(y=self.F_std_m*5, color='k', linestyle='dashed', lw=1)  
#            sp.axhline(y=self.F_std_m*10, color='k', linestyle='dashed', lw=1)         
#            sp.set_ylabel('Force std')            
                 
            fig_name = 'Data_' + str(data_num) + '_Trace_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fig_name))
            plt.close(fig)     

             
class Molecule:
    def __init__(self):
        self.path = os.getcwd()              
        file_list = os.listdir(self.path) 
        self.data_list = []
        for file_name in file_list:
            if file_name[-5:] == '.tdms':
                self.data_list.append(file_name)      
            
    def calibrate(self):
        path = os.path.join(os.getcwd(), 'Calibration')
        file_list = os.listdir(path) 

        self.QPD_nm2V = np.zeros(2)      
        self.stiffness_pN2nm = np.zeros(2)    

        for fname in file_list:
            if fname[-5:] == '.tdms':   
                axis = fname[4]

                PZT_A, beta, db, kappa, dk, ratio, dr = Calibration.main(path, fname, 
                    f_sample_cal, f_lp_cal, R, power, axis, fd_cal, Ad_cal, height_cal)
      
                if axis == 'X':
                    self.QPD_nm2V[0] = beta
                    self.stiffness_pN2nm[0] = kappa
                else:
                    self.QPD_nm2V[1] = beta
                    self.stiffness_pN2nm[1] = kappa                    
                
    def read_data(self):
        self.data = []
        for name in self.data_list:
            data = Data(name, self.QPD_nm2V, self.stiffness_pN2nm)
            data.read_data()
            self.data.append(data)
            
    def find_events(self):       
        for i in range(len(self.data)):
            self.data[i].find_events()

    def plot_events(self):
        path = make_folder('Events') 
        for i in range(len(self.data)):
            self.data[i].plot_events(i, path)                        
            
    def plot_traces(self):
        path = make_folder('Traces')
        for i in range(len(self.data)):
            self.data[i].plot_traces(i, path)
            
    def show_results(self):
        print("Show result ...")
        path = make_folder('Results')  
      
        # Combine all the forces
        Force_p = []
        Force_n = []
        F_cut = np.zeros(len(self.data))
        w = np.zeros(len(self.data))

        for i in range(len(self.data)):    
            F_cut[i] = self.data[i].F_cut     
            events = self.data[i].events
            w[i] = (len(events))**0.5 
            for j in range(len(events)):             
                e = events[j]
                if e.Force_max > 0:
                    Force_p.append(e.Force_max)
                else:
                    Force_n.append(e.Force_max)

        F_cut_mean = np.average(F_cut, weights = w)
        Force_p = np.array(Force_p)        
        Force_n = np.array(Force_n)  

        fig = plt.figure(i, figsize = (20, 10), dpi=300) 

        sp = fig.add_subplot(121)  
        sp.hist(abs(Force_p), 'scott', normed=False, color='r', histtype='step', linewidth=2)                                                                                
        sp.hist(abs(Force_n), 'scott', normed=False, color='b', histtype='step', linewidth=2)                                                                                
        sp.axvline(x=0, color='k', linestyle='dashed', lw=1)       
        sp.axvline(x=F_cut_mean, color='k', linestyle='dashed', lw=1)          
        sp.set_xlabel('Detachment force (pN)')   
        sp.set_ylabel('Counts')
        sp.set_title('# events = %d (R), %d (B)' %(len(Force_p), len(Force_n)))                    
#        sp.set_yscale('log')

        sp = fig.add_subplot(122)  
        sp.hist(abs(Force_p), 'scott', normed=True, color='r', histtype='step', linewidth=2)                                                                                
        sp.hist(abs(Force_n), 'scott', normed=True, color='b', histtype='step', linewidth=2)                                                                                
        sp.axvline(x=0, color='k', linestyle='dashed', lw=1)               
        sp.axvline(x=F_cut_mean, color='k', linestyle='dashed', lw=1) 
        sp.set_xlabel('Detachment force (pN)')  
        sp.set_ylabel('Probability') 
        sp.set_title('Mean force (pN) = %.1f (R), %.1f (B), F_cut = %.1f (pN)' %(np.mean(Force_p), np.mean(Force_n), F_cut_mean))                    

        fig.savefig(os.path.join(path, 'Force.png'))
        plt.close(fig)                          
                                                       
def main():
    mol = Molecule()
    mol.calibrate()
    mol.read_data()
    mol.find_events()    
    mol.show_results()
    mol.plot_traces()   
    mol.plot_events()    


if __name__ == "__main__":
    sys.exit(main())

"""
clean code for running_mean

"""




