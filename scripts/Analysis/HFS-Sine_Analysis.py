##################################################################
#
# Analysis of Harmonic force spectroscopy data (by Jongmin Sung)
#
##################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os 
import shutil 
import scipy
import math 
import sys 

### User input ##################################


def make_folder(name):
    path = os.path.join(PATH, name)       
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)    
    return path

def outliers(data, m = 5.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return s > m

def Exp_pdf(m, x):
    return np.exp(-x/abs(m))/abs(m) 



###############################################
def main():

    Fs = []
    ts = []
    As = []
    
    
    with open(os.path.join(PATH, 'Result.txt')) as f:
        for line in f:
            result = line.split(" ")
            Fs.append(float(result[0]))
            ts.append(float(result[1]))


            
    print("%d events are detected. \n" %(len(Fs)))             
    if len(Fs) == 0:
        return
                        
    Fs = np.array(Fs)
    ts = np.array(ts)

    if np.mean(ts[Fs>0]) > np.mean(ts[Fs<0]):
        Fs = -Fs          
    
    ks = 1/ts
           
    # Get Force dependent mean dwell time
#    F_bin = np.linspace(min(Fs)-0.1, max(Fs)+0.1, 11)
#    Fm = np.zeros(len(F_bin)-1)
#    tm = np.zeros(len(F_bin)-1)
#    tm_s = np.zeros(len(F_bin)-1)
    
#    for i in range(len(F_bin)-1):
#        cond1 = Fs > F_bin[i]
#        cond2 = Fs <= F_bin[i+1]
#        Fm[i] = np.mean(Fs[cond1 & cond2])
#        tm[i] = np.mean(ts[cond1 & cond2])
#    params, cov = scipy.optimize.curve_fit(exp, Fm, tm, p0=[np.mean(ts[abs(Fs)<0.1]), 1])
#    t0, dF = params
#    Fm_fit = np.linspace(min(Fs), max(Fs), 1000)
#    tm_fit = exp(Fm_fit, t0, dF)    
      
      
    path = make_folder('Results')       
    
    # Figure: Fv
    fig = plt.figure('FT', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(111)   
    sp.plot(Fs, ts, 'ko', ms=10, alpha=0.5)              
    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
    sp.set_ylim(0, max(ts)*1.1)
    sp.set_xlabel('Force (pN)', fontsize=30)  
    sp.set_ylabel('Dwell time (s)', fontsize=30)
    sp.set_title("Force vs Dwell time", fontsize=30)           
    fig.savefig(os.path.join(path, 'Force-Time.png'))
    plt.close(fig) 
        
    # Figure: Dwell
    ts_n = ts[Fs<0]
    ts_p = ts[Fs>0]
    
    m_n = (np.median(ts_n-min(ts_n))*np.log(2) + min(ts_n))
#    m_n = np.median(ts_n)*np.log(2)    
    m_p = (np.median(ts_p-min(ts_p))*np.log(2) + min(ts_p))
#    m_p = np.median(ts_p)*np.log(2)    
    t = np.linspace(0, max(abs(ts)), 100)
    
    fig = plt.figure('Dwell', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(121)   
    sp.hist(ts_n, color='k', histtype='step', density='True', lw=2)   
    sp.plot(t, Exp_pdf(m_n, t), 'r', lw=2)   
    sp.set_xlim([0, max(t)])       
    sp.set_title("Mean time = %.3f (s) (F < 0, N = %d)" %(m_n, len(ts_n)), fontsize=30)
    sp.set_xlabel('Dwell time (s)', fontsize=30)
    sp.set_ylabel('Probability density', fontsize=30)
    
    sp = fig.add_subplot(122)   
    sp.hist(ts_p, color='k', histtype='step', density='True', lw=2)  
    sp.plot(t, Exp_pdf(m_p, t), 'r', lw=2)      
    sp.set_xlim([0, max(t)])
    sp.set_title("Mean time = %.3f (s) (F > 0, N = %d)" %(m_p, len(ts_p)), fontsize=30)    
    sp.set_ylabel('Probability density', fontsize=30)    
    sp.set_xlabel('Dwell time (s)', fontsize=30)
    fig.savefig(os.path.join(path, 'Dwell.png'))
    plt.close(fig) 

    
"""

    fig = plt.figure('MFT', figsize = (20, 10), dpi=300)    
    sp = fig.add_subplot(111)   
    sp.plot(Fm, tm, 'ko', ms=10)     
    sp.plot(Fm_fit, tm_fit, 'r', lw=2)           
    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
    sp.set_ylim(0, max(tm)*1.1)
    sp.set_xlabel('Mean force (pN)')  
    sp.set_ylabel('Mean dwell time (ms)')
    sp.set_title("Mean force vs Mean dwell time")  
    fig.savefig(os.path.join(path, 'Mean Force-Time.png'))
    plt.close(fig) 

    

    # Figure: VKb
    fig = plt.figure('P-B', figsize = (20, 10), dpi=300)    
                  
    sp = fig.add_subplot(121)   
    Pb_bins = np.linspace(0, 2*pi, 9)
    sp.hist(PZT_Pb, bins=Pb_bins, histtype='stepfilled', lw=2)
    sp.axvline(x=pi*0.5, c='k', ls='dotted', lw=1) 
    sp.axvline(x=pi*1.0, c='k', ls='dotted', lw=1) 
    sp.axvline(x=pi*1.5, c='k', ls='dotted', lw=1)         
    sp.set_xlim(0, 2*pi)                          
    sp.set_xticks([0, pi, 2*pi])
    sp.set_xlabel('Phase @ binding')        
    sp.set_ylabel('Count')  
    
    sp = fig.add_subplot(122)   
    sp.plot(np.linspace(0, 2*pi, 9999), dPZT_QPD, 'b', lw=2)
    sp.axvline(x=pi*0.5, c='k', ls='dotted', lw=1) 
    sp.axvline(x=pi*1.0, c='k', ls='dotted', lw=1) 
    sp.axvline(x=pi*1.5, c='k', ls='dotted', lw=1)  
    sp.axhline(y=0, c='k', ls='dotted', lw=1)                 
    sp.set_xlim(0, 2*pi)                          
    sp.set_xticks([0, pi, 2*pi])
    sp.set_xlabel('Phase @ binding')        
    sp.set_ylabel('Negative Velocity')          
    fig.savefig(os.path.join(path, 'P-B.png'))
    plt.close(fig)                       
                                                                       
#    sp = fig.add_subplot(122)   
#    sp.hist(PZT_Pu, 10, density=True, facecolor='b', alpha=0.25)                           
#    sp.set_xlabel('Phase @ unbinding')  
#    sp.set_ylabel('Count')  
                           
#    sp = fig.add_subplot(132)   
#    sp.hist(Vb, 10)              
#    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#    sp.set_xlabel('Velocity @ binding (nm/ms)')  
#    sp.set_ylabel('Count')
#    sp.set_title("[%.2f : %.2f]" %(sum(Vb<0)/len(Vb), sum(Vb>0)/len(Vb))) 
                                                                    
#    sp = fig.add_subplot(133)   
#    sp.hist(Xb, 10)              
#    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#    sp.set_xlabel('Position @ binding (nm)')  
#    sp.set_ylabel('Count')
#    sp.set_title("[%.2f : %.2f]" %(sum(Xb<0)/len(Xb), sum(Xb>0)/len(Xb))) 
#    sp = fig.add_subplot(222)   
#    sp.hist(Vu/1000, 10)              
#    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#    sp.set_xlabel('Velocity @ unbinding (nm/ms)')  
#    sp.set_ylabel('Count')
#    sp.set_title("[%.2f : %.2f]" %(sum(Vu<0)/len(Vu), sum(Vu>0)/len(Vu))) 
#    sp = fig.add_subplot(224)   
#    sp.hist(Xu, 10)              
#    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
#    sp.set_xlabel('Position @ unbinding (nm)')  
#    sp.set_ylabel('Count')
#    sp.set_title("[%.2f : %.2f]" %(sum(Xu<0)/len(Xu), sum(Xu>0)/len(Xu))) 
    

"""

if __name__ == "__main__":
    main()


"""""""""
To-do

Exp decay: Negative if mean time increases

event finding based on two rounds? 1) both QPD_A and QPD_P, 2) then use dQPD_A?

stiffness

Save sample, slide, mol name and display in the plots

lowpass > subtract > std > cutoff based on std

Triangle oscillation and get binding ratio of forward vs backward motion, get freq dep
slower harmonic oscillation to get better phase estimate. 

Better step initial time (tb0, tu0) based on the max

split two events close 
better event picking: two events too close 

Be more conservative to exclude bad data. Only take good data. 
Time threshold for events. Upper and lower bound.


Solve fitting error
maxfev error detour. using try error

Finish event picking and show force dependent result. 
Analyze fs = 10 kHz data

Histogram of veocity or phase at tb and tu

binding phase distribuion (clustered?)
unbinding phase distribution (random?)

stalk stiffness at diff nt states
subtract QPD noise from sum
check correlation btw QPDs, Amp, Phase oscillation
Fourier transform with QPDs, Amp, Phase whether they are from the same noise source h hy hy


> Check Kim's code: HFS_event_detection, HFS_fitting
> Global MLE fitting (instead of binning) as in the xxx Goldman paper. Talk to Chao. 
> Simultaneous QPDx and Fx
> step finding? use 8 nm grid to find offset > fitting with grid as ref
> How to offset QPDx and QPDy for Force??
> In the setup: beta, stiffness, MTA, 
> Directly save/read from tmds (beta, stiffness, MTA, f_sample, f_lowpass, ch_name)
> More details in the description (sample, bead, axoneme vs MT, coverslip, conc, stall force or feedback, )
> Feedback axis?


> module_name, package_name, ClassName, method_name, ExceptionName, 
> function_name, GLOBAL_CONSTANT_NAME, global_var_name, instance_var_name, 
> function_parameter_name, local_var_name

================================================================================

Done




"""""""""





