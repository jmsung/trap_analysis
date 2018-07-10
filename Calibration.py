# nptdms is installed 
# download: https://pypi.python.org/pypi/npTDMS)
# documents: http://nptdms.readthedocs.io/en/latest/index.html

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

############### User Input ##############################
fname = "X_50Hz_100nm_2018_07_06_17_23_48.tdms"
fs = 10000
fd = 50
tmrs = 100
index = range(0, fs*tmrs)
index_all =False
N = 100
############### End of User Input #######################

class Data(object):
    def __init__(self):
        pass

    def read_tdms(self):

        # File information below
        tdms_file = TdmsFile(fname) # Reads a tdms file.
        root_object = tdms_file.object() # tdms file information 

        for name, value in root_object.properties.items():
            print("{0}: {1}".format(name, value))

        group_name = "Trap" # Get the group name
        channels = tdms_file.group_channels(group_name) # Get the channel object
        channel_name = [str(channels[i].channel) for i in range(0, len(channels))]
        #fs = round(1.0/channels[0].properties[u'wf_increment']) # Sampling frequency

        print("Channel number: %d" % len(channels))
        print("Channel name: %s" % channel_name) 
        print("Sampling rate: %d Hz" % fs)

        # Get data
        if index_all == True:
            index = range(0, len(channels[0].time_track()))

        x = np.empty([len(channels),len(index)]) # Make a 2D array (ch, timetrace) for trap data

        for i, channel in enumerate(channels):
            x[i,:] = channel.data[index] #Get data (time trace) for each channel
        t = channels[0].time_track()[index]
        y = x[0]

    def calculate_PSD(self):
        # PSD
        ps = np.abs(np.fft.fft(y))**2
        dt = 1/fs
        f = np.fft.fftfreq(y.size, dt)
        idx = np.argsort(f)

        mps = runningMean(ps[idx], N)
        mf = runningMean(f[idx], N)
        midx = np.argsort(mf)
        #midx = midx[4*N:]

    def plot(self):
        # Plot
        plt.close('all')

        plt.figure(1)
        for i in range(0, len(channels)):
            plt.subplot(len(channels), 1, i+1), plt.plot(t, x[i], 'k'), plt.ylabel(channel_name[i])
        plt.xlabel('Time (s)')

        plt.figure(2)
        #plt.loglog(f[idx], ps[idx], 'ko')
        plt.loglog(mf[midx], mps[midx], 'ko')
        plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (V^2/Hz)')

        plt.show()




def main():
    data = Data()
    data.read_tdms()
    data.calculate_PSD
    data.plot()

if __name__ == "__main__":
    main()
