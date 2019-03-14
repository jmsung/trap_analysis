# Test running variance

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import nptdms 
import os #import path, makedirs, getcwd, listdir
from scipy.optimize import curve_fit
import sys 
from trap_func import sine, trapzoid, running_mean, running_std


x = np.arange(0, 10, 0.01)
y = np.sin(x)

y_m = running_mean(y, 501)
y_s = running_std(y, 501)


fig = plt.figure(0)   
            
sp = fig.add_subplot(111)
sp.plot(x, y, 'k')
sp.plot(x, y_m, 'b')
sp.plot(x, y_s, 'r')        
fig.show()