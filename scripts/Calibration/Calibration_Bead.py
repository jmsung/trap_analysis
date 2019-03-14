# Trap calibration: Bead with stage oscillation

from __future__ import division, print_function, absolute_import
import Calibration1
import sys


files = ['Cal_X', 'Cal_Y']

f_sample = 10000
f_lp = 10000
power = 100
R = 240
fd = 100
Ad = 100
height = 500

def main(): 
    for i in range(2):       
        fname = files[i]
        axis = fname[-1]
        print(fname)
    
        PZT_A, beta, db, kappa, dk, ratio, dr = Calibration1.main(fname, f_sample, f_lp, R, power, axis, fd, Ad, height)


        info = open(fname+'.txt', 'w')

        info.write('A_fit = %.1f [nm] \n' % (abs(PZT_A)))
        info.write('beta = %.1f +/- %.1f [nm/V] \n' %(beta, db))  
        info.write('kappa = %.3f +/- %.3f [pN/nm] \n' %(kappa, dk))
        info.write('Stoke ratio = %.1f +/- %.3f \n\n' %(ratio, dr))   

        info.close()                
                                                
if __name__ == "__main__": 
    main()

























