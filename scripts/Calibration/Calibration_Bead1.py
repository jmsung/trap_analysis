# Trap calibration: Bead with stage oscillation

from __future__ import division, print_function, absolute_import
import Calibration1
import sys
from pathlib import Path  
from inspect import currentframe, getframeinfo
fname = getframeinfo(currentframe()).filename # current file name
current_dir = Path(fname).resolve().parent
data_directory = current_dir.parent.parent/'data'/'trap_calibration_data'/'19-10-08 Calibration'

files = ['test_15_32_54_X', 'test_15_32_54_Y']

f_sample = 30000
f_lp = 15000
power = 100
R = 500
fd = 150
Ad = 50
height = 500

def main(): 
    for i in range(2):      
        fname =  files[i]       
        axis = fname[-1]

        PZT_A, beta, db, kappa, dk, ratio, dr = Calibration1.main(data_directory, fname, f_sample, f_lp, R, power, axis, fd, Ad, height)

        info = open(fname+'.txt', 'w')

        info.write('A_fit = %.1f [nm] \n' % (abs(PZT_A)))
        info.write('beta = %.1f +/- %.1f [nm/V] \n' %(beta, db))  
        info.write('kappa = %.3f +/- %.3f [pN/nm] \n' %(kappa, dk))
        info.write('Stoke ratio = %.1f +/- %.3f \n\n' %(ratio, dr))   

        info.close()                
                                                
if __name__ == "__main__": 
    sys.exit(main())

























