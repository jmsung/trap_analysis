##################################################################
#
# Analysis of Trap force-feedback data (by Jongmin Sung)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
import os
import shutil

### User input ##################################

# First you need to change directory (cd) to where the file is located

# Update the file name
fname = 'Trap3_Motor1_16_01_18' 

# Parameters
QPD_nm2V = [100, 60]      # QPD sensitivity (nm/V) at V_sum = 8 V.
stiffness_pN2nm = [0.1, 0.1]  # Stiffness [pN/nm]
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor
MTA_nm2V = [1400, 1000]    # MTA_x [nm/V]

t_block = 20
f_lp = 20       # f_lowpass = 10 Hz
force = [0, -1]    # Set force x for force-feedback [pN]

###############################################

def running_mean(x, N=1000): # Smoothening by running averaging
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


class Data(object):
    def __init__(self):
        pass

   
    def read(self):  
        # File information below
        tdms_file = TdmsFile(fname+'.tdms') # Reads a tdms file.
        root_object = tdms_file.object() # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap"                             # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object
        self.channel_num = len(channels)                     # Channel number
        self.channel_name = [str(channels[i].channel) for i in range(len(channels))] # Channel name
        self.dt = channels[0].properties[u'wf_increment'] # Sampling time
        self.fs = int(1.0/self.dt)                         # Sampling frequency
        self.N = len(channels[0].time_track())

        print("Channel number: %d" % self.channel_num)
        print("Channel name: %s" % self.channel_name) 
        print("Sampling rate: %d Hz" % self.fs)   
        print("Data size: %d sec \n" % self.N*self.dt)  

        # Read data
        print("Reading raw data ... \n")
        self.t = channels[0].time_track(); 
        self.QPDx = (channels[0].data - np.mean(channels[0].data)) * QPD_nm2V[0]
        self.QPDy = (channels[1].data - np.mean(channels[1].data))* QPD_nm2V[1]  
        self.QPDs = channels[2].data
        self.PZTx = (channels[3].data - np.mean(channels[3].data)) * PZT_nm2V[0]
        self.PZTy = (channels[4].data - np.mean(channels[4].data)) * PZT_nm2V[1]
        self.PZTz = (channels[5].data - np.mean(channels[5].data)) * PZT_nm2V[2]
        self.MTAx = (channels[6].data - channels[6].data[0]) * MTA_nm2V[0]
        self.MTAy = (channels[7].data - channels[7].data[0]) * MTA_nm2V[1]
        self.Fx = self.QPDx * stiffness_pN2nm[0]
        self.Fy = self.QPDy * stiffness_pN2nm[1]
             
        # Make a directory to save the results
        self.path_data = os.getcwd()
        self.path_save = os.path.join(self.path_data, fname)

        if os.path.exists(self.path_save):
            shutil.rmtree(self.path_save)
            os.makedirs(self.path_save)
        else:
            os.makedirs(self.path_save)            
            
           
    def trace(self): # Time series    
        print("Plotting time traces ...")
        fig = plt.figure(1, figsize = (20, 10), dpi=300) 
        sp = fig.add_subplot(8, 1, 1)        
        sp.plot(self.t, self.QPDx, 'k', linewidth=0.5); sp.set_ylabel('QPDx [nm]')
        sp = fig.add_subplot(8, 1, 2)        
        sp.plot(self.t, self.QPDy, 'k', linewidth=0.5); sp.set_ylabel('QPDy [nm]')
        sp = fig.add_subplot(8, 1, 3)        
        sp.plot(self.t, self.QPDs, 'k', linewidth=0.5); sp.set_ylabel('QPDs [V]')
        sp = fig.add_subplot(8, 1, 4)        
        sp.plot(self.t, self.PZTx, 'b', linewidth=0.5); sp.set_ylabel('PZTx [nm]')                   
        sp = fig.add_subplot(8, 1, 5)        
        sp.plot(self.t, self.PZTy, 'b', linewidth=0.5); sp.set_ylabel('PZTy [nm]')
        sp = fig.add_subplot(8, 1, 6)        
        sp.plot(self.t, self.PZTz, 'b', linewidth=0.5); sp.set_ylabel('PZTz [nm]')
        sp = fig.add_subplot(8, 1, 7)        
        sp.plot(self.t, self.MTAx, 'r', linewidth=0.5); sp.set_ylabel('MTAx [nm]')
        sp = fig.add_subplot(8, 1, 8)        
        sp.plot(self.t, self.MTAy, 'r', linewidth=0.5); sp.set_ylabel('MTAy [nm]')    
        sp.set_xlabel('Time (s)')  
        fig.tight_layout()
        fig.savefig(os.path.join(self.path_save, 'Trace.png'))
        plt.close(fig)            

    def stall(self): # Stall force analysis
        print("Analyzing stall force data ...")
                

    def feedback(self): # Feedback data
        print("Analyzing force-feedback data ...")
        
        if np.std(self.MTAx) > np.std(self.MTAy):
            Trap = self.MTAx
            Bead = Trap + self.QPDx + 20
            axis = 0
        else:
            Trap = self.MTAy
            Bead = Trap + self.QPDy + 20    
            axis = 1        
          
        n = self.fs * t_block    # Number of data in a block
        m = int(self.N / n)      # Number of block

        t = self.t[:n*m]            
        Trap = Trap[:n*m]           
        Bead = Bead[:n*m]           


        t = t.reshape((m,n)) 
        x = x.reshape((m,n))           
        y = y.reshape((m,n))         
        QPDx = QPDx.reshape((m,n))           
        QPDy = QPDy.reshape((m,n))  

        # Make a directory to save the fig2
        path = os.path.join(self.path_save, 'Feedback')

        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)     

        for i in range(m):
            t_lp = running_mean(t[i], int(self.fs/f_lp))
            QPDy_lp = running_mean(QPDy[i], int(self.fs/f_lp))              
            
            fig = plt.figure(i+10, figsize = (20, 10), dpi=300) 
                                     
            sp = fig.add_subplot(111)                     
            sp.plot(t[i], y[i], 'r', linewidth=0.5)
#            sp.plot(t[i], QPDy[i], 'b', linewidth=1.0)   
            sp.plot(t_lp, QPDy_lp, 'b', linewidth=1.0)                      
            for j in np.arange(np.min(y[i]), np.max(y[i])+8, 8):
                sp.axhline(j, color='k', linestyle=':', linewidth=0.3)
            sp.set_ylabel('Step (nm)')
            sp.set_xlabel('Time (s)')      

            fname = 'Fig2_Feedback_' + str(i) + '.png'
#            fig.tight_layout()
            fig.savefig(os.path.join(path, fname))
            plt.close(fig)                                                                                                                                                                                                                           

    def analyze(self):
        self.trace()
        if np.std(self.MTAx) < 1 and np.std(self.MTAy) < 1:                 
            self.stall()
        else:
            self.feedback()                           
                                                                                 

def main():
    data = Data()
    data.read()
    data.analyze()




if __name__ == "__main__":
    main()


"""""""""
To-do

> Simultaneous QPDx and Fx
> step finding? use 8 nm grid to find offset > fitting with grid as ref
> How to offset QPDx and QPDy for Force??
> In the setup: beta, stiffness, MTA, 
> Directly save/read from tmds (beta, stiffness, MTA, f_sample, f_lowpass, ch_name)
> More details in the description (sample, bead, axoneme vs MT, coverslip, conc, stall force or feedback, )
> Feedback axis?



"""""""""





