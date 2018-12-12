# Trap calibration: Height dependence

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import Calibration_Bead
import Calibration_Bead_Hydro
from scipy.optimize import curve_fit

files = [['X_H0100', 'X_H0200', 'X_H0300', 'X_H0400', 'X_H0500', 'X_H0600', 'X_H0700', 'X_H0800', 'X_H0900', 'X_H1000', 'X_H1100', 'X_H1200', 'X_H1300', 'X_H1400', 'X_H1500'],
         ['Y_H0100', 'Y_H0200', 'Y_H0300', 'Y_H0400', 'Y_H0500', 'Y_H0600', 'Y_H0700', 'Y_H0800', 'Y_H0900', 'Y_H1000', 'Y_H1100', 'Y_H1200', 'Y_H1300', 'Y_H1400', 'Y_H1500']]

files = [['X_H0200', 'X_H0300', 'X_H0400', 'X_H0500', 'X_H0600', 'X_H0700', 'X_H0800', 'X_H0900', 'X_H1000', 'X_H1100', 'X_H1200', 'X_H1300', 'X_H1400', 'X_H1500'],
         ['Y_H0200', 'Y_H0300', 'Y_H0400', 'Y_H0500', 'Y_H0600', 'Y_H0700', 'Y_H0800', 'Y_H0900', 'Y_H1000', 'Y_H1100', 'Y_H1200', 'Y_H1300', 'Y_H1400', 'Y_H1500']]


#files = [['X_H0100', 'X_H0500', 'X_H0900'],
#         ['Y_H0100', 'Y_H0500', 'Y_H0900']]


R = 430
fd = 50
Ad = 50
power = 100

def Faxen(H, offset, B):
    h = H+offset
    x = 1 - 9*R/16/h + (R**3)/8/(h**3) - 45*(R**4)/256/(h**4) - (R**5)/16/(h**5)
    return B/x


def main():
    
    fig = plt.figure(10, figsize = (20, 10), dpi=300)     

    for i in range(2):

        h = np.zeros(len(files[i]))
        b = np.zeros(len(files[i]))
        db = np.zeros(len(files[i]))
        k = np.zeros(len(files[i]))
        dk = np.zeros(len(files[i]))
        r = np.zeros(len(files[i]))
        dr = np.zeros(len(files[i]))
        
        for j in range(len(files[i])):
                
            fname = files[i][j]
            axis = fname[0]
            h[j] = int(fname[-4:])+R
            print(fname)
            print(h[j])
        
            b[j], db[j], k[j], dk[j], r[j], dr[j] = Calibration_Bead_Hydro.main(fname, axis, power, fd, Ad, h[j])
#            b[j], db[j], k[j], dk[j], r[j], dr[j] = Calibration_Bead.main(fname, axis, power, fd, Ad)

        # Beta * Kappa
        bk = b*k
        dbk = bk*((db/b)**2 + (dk/k)**2)**0.5
        mbk = np.average(bk, weights = 1/dbk**2)
       
        # Beta
        mb = np.average(b, weights = 1/db**2)
        
        # Kappa
        mk = np.average(k, weights = 1/dk**2)    
        
        # Stoke ratio        
        p_r, cov = curve_fit(Faxen, h, r, p0=[0, 3], sigma=dr)
        offset = p_r[0]
        ratio = p_r[1]
        
        h = h + offset
        x = np.linspace(min(h)-10, max(h)+100, 100)        
        r_fit = Faxen(x, 0, ratio)
                                                                                                                                                                                   
        sp = fig.add_subplot(2,4,4*i+1)
        sp.axhline(y=mb, color='k', linestyle='solid', linewidth=1)
        sp.axvline(x=R, color='k', linestyle='dashed', linewidth=1)   
        sp.errorbar(h, b, yerr=db, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, max(h)+100))
        sp.set_xlabel('Bead center to surface [nm]')
        sp.set_ylabel('Beta [nm/V]')
        sp.set_title('Beta [nm/V] = %d +/- %d' %(mb, np.std(b)))

        sp = fig.add_subplot(2,4,4*i+2)
        sp.errorbar(h, k, yerr=dk, fmt='o', ecolor='k', color='k')
        sp.axvline(x=R, color='k', linestyle='dashed', linewidth=1)   
        sp.axhline(y=mk, color='k', linestyle='solid', linewidth=1)
        sp.set_xlim((0, max(h)+100))
        sp.set_xlabel('Bead center to surface [nm]')
        sp.set_ylabel('Kappa [pN/nm]')
        sp.set_title('Kappa [pN/nm] = %.3f +/- %.3f' %(mk, np.std(k)))               
    
        sp = fig.add_subplot(2,4,4*i+3)
        sp.errorbar(h, b*k, yerr=dbk, fmt='o', ecolor='k', color='k')
        sp.axvline(x=R, color='k', linestyle='dashed', linewidth=1)   
        sp.axhline(y=mbk, color='k', linestyle='solid', linewidth=1)
        sp.set_xlim((0, max(h)+100))
        sp.set_xlabel('Bead center to surface [nm]')
        sp.set_ylabel('Beta*Kappa [pN/V]')
        sp.set_title('Beta*Kappa [pN/V] = %.1f +/- %.1f' %(mbk, np.std(dbk)))    


        sp = fig.add_subplot(2,4,4*i+4)
        sp.plot(x, r_fit, 'r')
        sp.errorbar(h, r, yerr=dr, fmt='o', ecolor='k', color='k')
        sp.axvline(x=R, color='k', linestyle='dashed', linewidth=1)        
        sp.axhline(y=ratio, color='r', linestyle='dashed', linewidth=1)
        sp.axhline(y=1, color='k', linestyle='solid', linewidth=1)
        sp.set_xlim((0, max(h)+100))
        sp.set_xlabel('Bead center to surface [nm]')
        sp.set_ylabel('Stoke ratio')
        sp.set_title('Stoke ratio = %.1f, Offset = %.1f nm' %(ratio, offset))

    fig.savefig('Calibration result.png')
    plt.close(fig)     

if __name__ == "__main__":
    main()



























