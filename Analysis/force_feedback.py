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
fname = 'M1_12_33_16' 

beta = [100, 60]      # QPD sensitivity (nm/V)
kappa = [0.05, 0.05]  # Stiffness [pN/nm]
MTA = [1400, 1000]    # MTA_x [nm/V]

force_set = 0   # Set force for force-feedback [pN]

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

        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object
        channel_num = len(channels)
        channel_name = [str(channels[i].channel) for i in range(len(channels))]
        dt = channels[0].properties[u'wf_increment'] # Sampling frequency
        fs = int(1.0/dt)
        N = len(channels[0].time_track())

        x = np.empty([channel_num, N]) # Make a 2D array (ch, timetrace) for trap data
        for i, channel in enumerate(channels):
            x[i,] = channel.data[range(N)] #Get data (time trace) for each channel
        t = dt*np.arange(N)
        
        
             
        print("Channel number: %d" % len(channels))
        print("Channel name: %s" % channel_name) 
        print("Sampling rate: %d Hz" % fs)                  


      
    def analyze(self):
        pass

    
    def save(self):
        # Make a directory to save the results
        self.data_path = os.getcwd()
        self.dir = os.path.join(self.data_path, fname)

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
            os.makedirs(self.dir)
        else:
            os.makedirs(self.dir)            
         
           
    def plot_fig1(self): # Time series 
        
        fig = plt.figure(1, figsize = (20, 10), dpi=300)  

#        fig.savefig(os.path.join(self.dir, 'Fig1.png'))
#        plt.close(fig)
                                  


    def plot(self):
        self.plot_fig1()

                

def main():
    data = Data()
    data.read()
    data.analyze()
    data.save()
    data.plot()


if __name__ == "__main__":
    main()


"""""""""
To-do

> In the setup: beta, stiffness, MTA, 
> Directly save/read from tmds (beta, stiffness, MTA, f_sample, f_lowpass, ch_name)
> More details in the description (sample, bead, axoneme vs MT, coverslip, conc, stall force or feedback, )
> Feedback axis?



"""""""""





