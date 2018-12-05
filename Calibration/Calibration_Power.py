# Trap calibration: power dependence

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import Calibration_Bead
from scipy.optimize import curve_fit

files = [['X_P010', 'X_P020', 'X_P030', 'X_P040', 'X_P050', 'X_P060', 'X_P070', 'X_P080', 'X_P090', 'X_P100'],
         ['Y_P010', 'Y_P020', 'Y_P030', 'Y_P040', 'Y_P050', 'Y_P060', 'Y_P070', 'Y_P080', 'Y_P090', 'Y_P100']]

files = [['X_P020', 'X_P040', 'X_P060', 'X_P080', 'X_P100'],
         ['Y_P020', 'Y_P040', 'Y_P060', 'Y_P080', 'Y_P100']]

def func(x, a, b):
    return a * x + b

fd = 50
Ad = 50

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
        
            b[j], db[j], k[j], dk[j], r[j], dr[j] = Calibration_Bead.main(fname, axis, p[j], fd, Ad)
 
        
        x = np.linspace(0, 110, 10)
        
        # Beta
        mb = np.average(b, weights = 1/db**2)
        
        # Kappa
        p_k, cov = curve_fit(func, p, k, sigma = dk)
        k_fit = func(x, p_k[0], p_k[1])
        dPdk = p_k[0]        
        
        # Stoke ratio        
        mr = np.average(r, weights = 1/dr**2)
                                                          
        sp = fig.add_subplot(2,3,3*i+1)
        sp.axhline(y=mb, color='k', linestyle='solid', linewidth=2)
        sp.errorbar(p, b, yerr=db, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Beta [nm/V]')
        sp.set_title('Beta = %d +/- %d' %(mb, np.std(b)))

        sp = fig.add_subplot(2,3,3*i+2)
        sp.plot(x, k_fit, 'r')
        sp.errorbar(p, k, yerr=dk, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Kappa [pN/nm]')
        sp.set_title('Kappa/power = %.4f' %(dPdk))               
    
        sp = fig.add_subplot(2,3,3*i+3)
        sp.axhline(y=mr, color='k', linestyle='solid', linewidth=2)
        sp.errorbar(p, r, yerr=dr, fmt='o', ecolor='k', color='k')
        sp.set_xlim((0, 110))
        sp.set_xlabel('Power [%]')
        sp.set_ylabel('Stoke ratio')
        sp.set_title('Stoke ratio = %.1f +/- %.1f' %(mr, np.std(r)))

    fig.savefig('Calibration result.png')
    plt.close(fig)     

if __name__ == "__main__":
    main()


"""
# Data #########################################################################

fd = 50 
Ad = 50

power = np.array([20, 40, 60, 80, 100])

# X-axis oscillation
bx = np.array([465.3, 433.3, 409.3, 414.9, 358.3])
dbx = np.array([54.4, 36.4, 28.1, 24.3, 16.6])

fcx = np.array([762.6, 1483.0, 2244.8, 3108.2, 4749.3])
dfcx = np.array([86.6, 122.5, 152.3, 180.5, 218.3])

kx = np.array([0.127, 0.296, 0.507, 0.676, 0.986])
dkx = np.array([0.053, 0.089, 0.127, 0.148, 0.178])

# Y-axis oscillation
by = np.array([314.5, 327.8, 327.7, 333.3, 346.8])
dby = np.array([24.3, 18.5, 15.8, 14.2, 13.5])

fcy = np.array([648.2, 1136.5, 1659.2, 2204.4, 2702.2])
dfcy = np.array([49.6, 63.9, 79.7, 93.2, 104.7])

ky = np.array([0.106, 0.181, 0.272, 0.361, 0.443])
dky = np.array([0.029, 0.037, 0.049, 0.058, 0.067])

# Result #######################################################################

x = np.linspace(0, 110, 10)

# [X] beta
mbx = np.average(bx, weights = 1/dbx**2)

# [X] fc
p, cov = curve_fit(func, power, fcx, sigma = dfcx)
fcx_fit = func(x, p[0], p[1])
dPdfcx = p[0]
dPdfcx_err = cov[0][0]/(len(power)-1)**0.5

# [X] kappa
p, cov = curve_fit(func, power, kx, sigma = dkx)
kx_fit = func(x, p[0], p[1])
dPdkx = p[0]
dPdkx_err = cov[0][0]/(len(power)-1)**0.5

################################################################################
# [Y] Beta
mby = np.average(by, weights = 1/dby**2)

# [Y] fc
p, cov = curve_fit(func, power, fcy, sigma = dfcy)
fcy_fit = func(x, p[0], p[1])
dPdfcy = p[0]
dPdfcy_err = cov[0][0]/(len(power)-1)**0.5

# [Ykappa
p, cov = curve_fit(func, power, ky, sigma = dky)
ky_fit = func(x, p[0], p[1])
dPdky = p[0]
dPdky_err = cov[0][0]/(len(power)-1)**0.5

# Plot #########################################################################        
fig = plt.figure(1, figsize = (20, 10), dpi=300)  

sp = fig.add_subplot(231)
sp.axhline(y=mbx, color='k', linestyle='solid', linewidth=2)
sp.errorbar(power, bx, yerr=dbx, fmt='o', ecolor='k', color='k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('beta_X [nm/V]')
sp.set_title('beta = %d +/- %d' %(mbx, np.std(bx)/(len(power)-1)**0.5))

sp = fig.add_subplot(232)
sp.errorbar(power, fcx, yerr=dfcx, fmt='o', ecolor='k', color='k')
sp.plot(x, fcx_fit, 'k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('fc_X [Hz]')
sp.set_title('fc/power = %.1f +/- %.1f' %(dPdfcx, dPdfcx_err))

sp = fig.add_subplot(233)
sp.errorbar(power, kx, yerr=dkx, fmt='o', ecolor='k', color='k')
sp.plot(x, kx_fit, 'k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('k_X [pN/nm]')
sp.set_title('kappa/power = %.4f' %(dPdkx))

################################################################################
sp = fig.add_subplot(234)
sp.axhline(y=mby, color='k', linestyle='solid', linewidth=2)
sp.errorbar(power, by, yerr=dby, fmt='o', ecolor='k', color='k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('beta_Y [nm/V]')
sp.set_title('beta = %d +/- %d' %(mby, np.std(by)/(len(power)-1)**0.5))

sp = fig.add_subplot(235)
sp.errorbar(power, fcy, yerr=dfcy, fmt='o', ecolor='k', color='k')
sp.plot(x, fcy_fit, 'k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('fc_Y [Hz]')
sp.set_title('fc/power = %.1f +/- %.1f' %(dPdfcy, dPdfcy_err))

sp = fig.add_subplot(236)
sp.errorbar(power, ky, yerr=dky, fmt='o', ecolor='k', color='k')
sp.plot(x, ky_fit, 'k')
sp.set_xlabel('Power [%]')    
sp.set_ylabel('k_Y [pN/nm]')
sp.set_title('kappa/power = %.4f' %(dPdky))



fig.savefig('Calibration result.png')
plt.close(fig)       

"""

























