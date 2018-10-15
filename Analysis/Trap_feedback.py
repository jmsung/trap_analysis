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

# First, change directory (cd) to where the file is located

# Update the file name
fname = 'Trap1_Motor1_15_55_02' 

# Parameters
QPD_nm2V = [100, 60]      # QPD sensitivity (nm/V) at V_sum = 8 V.
stiffness_pN2nm = [0.1, 0.1]  # Stiffness [pN/nm]
PZT_nm2V = [5000, 5000, 3000]  # PZT Volt to nm conversion factor
MTA_nm2V = [1400, 1000]    # MTA_x [nm/V]

# Stall force


# Force feedback
t_feedback = 10
f_lp = 1000       # f_lowpass = 100 Hz
force = [0, -1]    # Set force x for force-feedback [pN]

# HFS
t_HFS = 0.5
fd = 100
Ad = 100


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
        print("Data size: %d sec \n" % int(self.N*self.dt))  

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
        print("Plotting time traces ... \n")
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
        print("Analyzing stall force data ... \n")

        # Make a directory to save the stall force result
        path = os.path.join(self.path_save, 'Stall')

        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)  

    def HFS(self):
        print("Analyzing HFS data ... \n")   

        if np.std(self.PZTx) > np.std(self.PZTy):
            PZT = self.PZTx
            Bead = self.QPDx
            axis = 0
        else:
            PZT = self.PZTy
            Bead = self.QPDy
            axis = 1       
          
        n = int(self.fs * t_HFS)    # Number of data in a block
        m = int(self.N / n)      # Number of block

        t = self.t[:n*m]            
        PZT = PZT[:n*m]           
        Bead = Bead[:n*m]           

        t = t.reshape((m,n)) 
        PZT = PZT.reshape((m,n))                  
        Bead = Bead.reshape((m,n))           

        # Make a directory to save the Force feedback result
        path = os.path.join(self.path_save, 'HFS')

        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)     

        for i in range(m):            
            fig = plt.figure(i+1000, figsize = (20, 10), dpi=300) 
                                     
            sp = fig.add_subplot(211)                     
            sp.plot(t[i], PZT[i], 'k', linewidth=0.5)   
            sp.set_ylabel('PZT (nm)') 
            sp = fig.add_subplot(212)
            sp.plot(t[i], Bead[i], 'k', linewidth=0.5)                                
            sp.set_ylabel('Bead (nm)')
            sp.set_xlabel('Time (s)')      

            fname = 'HFS_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fname))
            plt.close(fig)          
        
        
        
        # Make a directory to save the HFS result
        path = os.path.join(self.path_save, 'HFS')

        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)        
                

    def feedback(self): # Feedback data
        print("Analyzing force-feedback data ... \n")
        
        if np.std(self.MTAx) > np.std(self.MTAy):
            Trap = self.MTAx
            Bead = Trap + self.QPDx + 5*force[0]/stiffness_pN2nm[0]
            axis = 0
        else:
            Trap = self.MTAy
            Bead = Trap + self.QPDy + 5*force[1]/stiffness_pN2nm[1]    
            axis = 1        
          
        n = self.fs * t_block    # Number of data in a block
        m = int(self.N / n)      # Number of block

        t = self.t[:n*m]            
        Trap = Trap[:n*m]           
        Bead = Bead[:n*m]           

        t = t.reshape((m,n)) 
        Trap = Trap.reshape((m,n))                  
        Bead = Bead.reshape((m,n))           

        # Make a directory to save the Force feedback result
        path = os.path.join(self.path_save, 'Feedback')

        if os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path)
        else:
            os.makedirs(path)     

        for i in range(m):
            t_lp = running_mean(t[i], int(self.fs/f_lp))
            Bead_lp = running_mean(Bead[i], int(self.fs/f_lp))              
            
            fig = plt.figure(i+10, figsize = (20, 10), dpi=300) 
                                     
            sp = fig.add_subplot(111)                     
            sp.plot(t[i], Trap[i], 'r', linewidth=0.5)
            sp.plot(t_lp, Bead_lp, 'b', linewidth=1.0)                      
            for j in np.arange(np.min([Bead[i],Trap[i]])-8, np.max([Bead[i],Trap[i]])+8, 8):
                sp.axhline(j, color='k', linestyle=':', linewidth=0.3)
            sp.set_ylabel('Step (nm)')
            sp.set_xlabel('Time (s)')      

            fname = 'Feedback_' + str(i) + '.png'
            fig.savefig(os.path.join(path, fname))
            plt.close(fig)                                                                                                                                                                                                                           

    def analyze(self):
        self.trace()
        if np.std(self.PZTx) > 10 or np.std(self.PZTy) > 10:
            self.HFS()
        elif np.std(self.MTAx) > 10 or np.std(self.MTAy) > 10:                 
            self.feedback() 
        else:
            self.stall()                          
                                                                                 

def main():
    data = Data()
    data.read()
    data.analyze()



if __name__ == "__main__":
    main()


"""""""""
To-do

> Running_avg > Block_avg: since runing avg smooth out (do not show sudden step)
> Simultaneous QPDx and Fx
> step finding? use 8 nm grid to find offset > fitting with grid as ref
> How to offset QPDx and QPDy for Force??
> In the setup: beta, stiffness, MTA, 
> Directly save/read from tmds (beta, stiffness, MTA, f_sample, f_lowpass, ch_name)
> More details in the description (sample, bead, axoneme vs MT, coverslip, conc, stall force or feedback, )
> Feedback axis?



"""""""""





