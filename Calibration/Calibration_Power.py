# Trap calibration: power dependence

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import Calibration
from scipy.optimize import curve_fit

files = [['X_P010', 'X_P020', 'X_P030', 'X_P040', 'X_P050', 'X_P060', 'X_P070', 'X_P080', 'X_P090', 'X_P100'],
         ['Y_P010', 'Y_P020', 'Y_P030', 'Y_P040', 'Y_P050', 'Y_P060', 'Y_P070', 'Y_P080', 'Y_P090', 'Y_P100']]


def func(x, a, b):
    return a * x + b

R = 430
fd = 50
Ad = 50
height = 1000

def main():
    
    fig = plt.figure(10, figsize = (20, 10), dpi=300)     

    for i in range(2):

        p = np.zeros(len(files[i]))
        b = np.zeros(len(files[i]))
        db = np.zeros(len(files[i]))
        k = np.zeros(len(files[i]))
        dk = np.zeros(len(files[i]))
        r = np.zeros(len(files[i]))
        dr = np.zeros(len(files[i]))
        
        for j in range(len(files[i])):
                
            fname = files[i][j]
            axis = fname[0]
            p[j] = int(fname[-3:])
            print(fname)
        
            b[j], db[j], k[j], dk[j], r[j], dr[j] = Calibration.main(fname, axis, p[j], fd, Ad, R+height)
#            b[j], db[j], k[j], dk[j], r[j], dr[j] = Calibration_Bead.main(fname, axis, p[j], fd, Ad)
 
        
        x = np.linspace(0, max(p)+10, 100)
        
        # Beta
        p_b, cov = curve_fit(func, p, b, sigma = db)
        b_fit = func(x, p_b[0], p_b[1])
        pb = p_b[0]        
        b0 = p_b[1]
    
        # Kappa
        p_k, cov = curve_fit(func, p, k, sigma = dk)
        k_fit = func(x, p_k[0], p_k[1])
        pk = p_k[0]        
        k0 = p_k[1]
        
        # Stoke ratio        
        mr = np.average(r, weights = 1/dr**2)
                                                          
        sp = fig.add_subplot(2,3,3*i+1)
        sp.plot(x, b_fit, 'r', lw=1)
        sp.errorbar(p, b, yerr=db, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Beta [nm/V]')
        sp.set_title('Beta [nm/V] = %.1f - %.1f * power [%%] ' %(b0, abs(pb)))

        sp = fig.add_subplot(2,3,3*i+2)
        sp.plot(x, k_fit, 'r', lw=1)
        sp.errorbar(p, k, yerr=dk, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Kappa [pN/nm]')
        sp.set_title('Kappa [pN/nm] = %.5f + %.5f * power [%%] ' %(k0, pk))               
    
        sp = fig.add_subplot(2,3,3*i+3)
        sp.axhline(y=mr, color='r', linestyle='solid', lw=1)
        sp.axhline(y=1, color='k', linestyle='solid', lw=1)
        sp.errorbar(p, r, yerr=dr, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Stoke ratio')
        sp.set_title('Stoke ratio = %.1f +/- %.1f' %(mr, np.std(r)))

    fig.savefig('Calibration result.png')
    plt.close(fig)     

if __name__ == "__main__":
    main()

























