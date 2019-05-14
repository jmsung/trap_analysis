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
from scipy.optimize import curve_fit
import math 
import sys 

### User input ##################################

#PATH = r'C:\Users\Jongmin Sung\Box Sync\Research\Project\Trap\18-12-07 1067_HFS\Slide1_ATP1mM\Mol1_P50%_fd100Hz_Ad100nm'

f_drive = 100 # Hz
fs = 2000
dt = 1/fs
N_QPD = fs/f_drive*2

def make_folder(name):
    path = os.path.join(os.getcwd(), name)       
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

def sine(t, A, ph, b): # Sine function
    return A * np.sin(2*np.pi*f_drive*t - ph) + b    

###############################################
def main():

    Fs = []
    ts = []
    Ab = []
    Au = []
    QPD_Xb = []
    QPD_Xu = []
    QPD_Vb = []
    QPD_Vu = []
    PZT_Xb = []
    PZT_Xu = [] 
    PZT_Vb = []
    PZT_Vu = []
    
    with open(os.path.join(os.getcwd() , 'Result.txt')) as f:
        for line in f:
            result = line.split(" ")
            Fs.append(float(result[0]))
            ts.append(float(result[1]))
            Ab.append(float(result[2]))
            Au.append(float(result[3]))            
            QPD_Xb.append(float(result[4]))
            QPD_Xu.append(float(result[5]))
            QPD_Vb.append(float(result[6]))
            QPD_Vu.append(float(result[7]))
            PZT_Xb.append(float(result[8]))
            PZT_Xu.append(float(result[9]))
            PZT_Vb.append(float(result[10]))
            PZT_Vu.append(float(result[11]))

    print("%d events are detected. \n" %(len(Fs)))             
    if len(Fs) == 0:
        return
                    
    Fs = np.array(Fs)
    ts = np.array(ts)
    Ab = np.array(Ab)
    Au = np.array(Au)
    QPD_Xb = np.array(QPD_Xb)
    QPD_Xu = np.array(QPD_Xu)
    QPD_Vb = np.array(QPD_Vb)
    QPD_Vu = np.array(QPD_Vu)
    PZT_Xb = np.array(PZT_Xb)
    PZT_Xu = np.array(PZT_Xu)
    PZT_Vb = np.array(PZT_Vb)
    PZT_Vu = np.array(PZT_Vu)    

    if np.mean(ts[Fs>0]) > np.mean(ts[Fs<0]):
        Fs = -Fs          
    
    ks = 1/ts

    # QPD after binding
    QPD_binding = []
    with open(os.path.join(os.getcwd() , 'QPD_binding.txt')) as f:
        for line in f:
            x = line.split(" ")
            QPD = [float(i) for i in x[:-1]]
            QPD_binding.append(QPD)
    QPD_binding = np.array(QPD_binding)
    QPD_binding = np.mean(QPD_binding, axis=0)
    QPD_binding = QPD_binding - np.mean(QPD_binding)

    p0_b = [max(QPD_binding), 0, 0]
    lb_b = (max(QPD_binding)/2, -np.pi, -max(QPD_binding))
    ub_b = (max(QPD_binding)*2, np.pi, max(QPD_binding))

    t = np.arange(0, N_QPD*dt, dt)    
    p_b, cov = curve_fit(sine, t, QPD_binding, p0_b, bounds = (lb_b, ub_b))            
    QPD_binding_fit = sine(t, p_b[0], p_b[1], p_b[2])  
    stroke_binding = QPD_binding - QPD_binding_fit
  
    # QPD before unbinding 
    QPD_unbinding = []
    with open(os.path.join(os.getcwd() , 'QPD_unbinding.txt')) as f:
        for line in f:
            x = line.split(" ")
            QPD = [float(i) for i in x[:-1]]
            QPD_unbinding.append(QPD)
    QPD_unbinding = np.array(QPD_unbinding)
    QPD_unbinding = np.mean(QPD_unbinding, axis=0)
    QPD_unbinding = QPD_unbinding - np.mean(QPD_unbinding)      
  
    p0_u = [max(QPD_unbinding), 0, 0]
    lb_u = (max(QPD_unbinding)/2, -np.pi, -max(QPD_unbinding))
    ub_u = (max(QPD_unbinding)*2, np.pi, max(QPD_unbinding))
    
    p_u, cov = curve_fit(sine, t, QPD_unbinding, p0_u, bounds = (lb_u, ub_u))            
    QPD_unbinding_fit = sine(t, p_u[0], p_u[1], p_u[2])  
    stroke_unbinding = QPD_unbinding - QPD_unbinding_fit
         
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
    
    # Figure: F-T
    fig = plt.figure('FT', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(121)   
    sp.plot(Fs, ts, 'ko', ms=7, alpha=0.5)              
    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
    sp.set_ylim(0, max(ts)*1.1)
    sp.set_xlabel('Force (pN)', fontsize=20)  
    sp.set_ylabel('Dwell time (s)', fontsize=20)
    sp.set_title("Force vs Dwell time", fontsize=20)           
   
    sp = fig.add_subplot(122)   
    sp.plot(Fs, Ab-Au, 'ko', ms=7, alpha=0.5)              
    sp.axvline(x=0, c='k', ls='dotted', lw=1)   
    sp.set_ylim(0, max(Ab-Au)*1.1)
    sp.set_xlabel('Force (pN)', fontsize=20)  
    sp.set_ylabel('Amplitude (nm)', fontsize=20)
    sp.set_title("Force vs Amplitude", fontsize=20) 

    fig.savefig(os.path.join(path, 'Force-Time-Amp.png'))
    plt.close(fig) 
        
    # Figure: Dwell
    ts_n = ts[Fs<0]
    ts_p = ts[Fs>0]
    
    m_n = (np.median(ts_n-min(ts_n))*np.log(2) + min(ts_n))
#    m_n = np.median(ts_n)*np.log(2)    
    m_p = (np.median(ts_p-min(ts_p))*np.log(2) + min(ts_p))
#    m_p = np.median(ts_p)*np.log(2)    
    t = np.linspace(0, max(abs(ts)), 100)


    #######################################################################    
    fig = plt.figure('Dwell', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(121)   
    bins = sp.hist(ts_n, bins = 10, color='k', histtype='step', lw=2)   
    sp.plot(t, Exp_pdf(m_n, t)*len(ts_n)*(bins[1][1]-bins[1][0]), 'r', lw=2)   
    sp.set_xlim([0, max(t)])       
    sp.set_title("Mean time = %.3f (s) (F < 0, N = %d)" %(m_n, len(ts_n)), fontsize=20)
    sp.set_xlabel('Dwell time (s)', fontsize=20)
    sp.set_ylabel('Counts', fontsize=20)
    
    sp = fig.add_subplot(122)   
    bins = sp.hist(ts_p, bins = 10, color='k', histtype='step', lw=2)  
    sp.plot(t, Exp_pdf(m_p, t)*len(ts_p)*(bins[1][1]-bins[1][0]), 'r', lw=2)      
    sp.set_xlim([0, max(t)])
    sp.set_title("Mean time = %.3f (s) (F > 0, N = %d)" %(m_p, len(ts_p)), fontsize=20)    
    sp.set_ylabel('Counts', fontsize=20)    
    sp.set_xlabel('Dwell time (s)', fontsize=20)
    fig.savefig(os.path.join(path, 'Dwell.png'))
    plt.close(fig) 


    #######################################################################
    fig = plt.figure('Binding_QPD', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(221)   
    sp.hist(QPD_Xb, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Position (nm)') 
    sp.set_ylabel('Counts')
    sp.set_title('QPD position at binding', fontsize=20)

    sp = fig.add_subplot(222)   
    sp.hist(QPD_Xu, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Position (nm)') 
    sp.set_ylabel('Counts')
    sp.set_title('QPD position at unbinding', fontsize=20)

    sp = fig.add_subplot(223)   
    sp.hist(QPD_Vb/1000, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Velocity (um/s)') 
    sp.set_ylabel('Counts')    
    sp.set_title('QPD velocity at binding', fontsize=20)

    sp = fig.add_subplot(224)   
    sp.hist(QPD_Vu/1000, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Velocity (um/s)') 
    sp.set_ylabel('Counts')
    sp.set_title('QPD velocity at unbinding', fontsize=20)
    fig.savefig(os.path.join(path, 'Binding_QPD.png'))
    plt.close(fig)     
    
    
    #######################################################################    
    fig = plt.figure('Binding_PZT', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(221)   
    sp.hist(PZT_Xb, color='k', bins=10, histtype='step', lw=2)  
    sp.set_xlabel('Position (nm)') 
    sp.set_ylabel('Counts')
    sp.set_title('PZT position at binding', fontsize=20)

    sp = fig.add_subplot(222)   
    sp.hist(PZT_Xu, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Position (nm)') 
    sp.set_ylabel('Counts')
    sp.set_title('PZT position at unbinding', fontsize=20)

    sp = fig.add_subplot(223)   
    sp.hist(PZT_Vb/1000, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Velocity (um/s)') 
    sp.set_ylabel('Counts')
    sp.set_title('PZT velocity at binding', fontsize=20)

    sp = fig.add_subplot(224)   
    sp.hist(PZT_Vu/1000, color='k', bins=10, histtype='step', lw=2)   
    sp.set_xlabel('Velocity (um/s)') 
    sp.set_ylabel('Counts')
    sp.set_title('PZT velocity at unbinding', fontsize=20)
    fig.savefig(os.path.join(path, 'Binding_PZT.png'))
    plt.close(fig) 

    #######################################################################    
    t = np.arange(0, N_QPD*dt, dt) 

    fig = plt.figure('QPD_binding', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(211)   
    sp.plot(t*1000, QPD_binding, 'k') 
    sp.plot(t*1000, QPD_binding_fit, 'r')     
    sp.set_title('QPD at binding', fontsize=20)
    sp = fig.add_subplot(212)   
    sp.plot(t*1000, stroke_binding, 'b')     
    sp.set_title('Stroke at binding', fontsize=20)    
    
    fig.savefig(os.path.join(path, 'QPD_binding.png'))
    plt.close(fig) 

    #######################################################################    
    fig = plt.figure('QPD_unbinding', figsize = (20, 10), dpi=300)     
    sp = fig.add_subplot(211)   
    sp.plot(t*1000, QPD_unbinding, 'k') 
    sp.plot(t*1000, QPD_unbinding_fit, 'r')   
    sp.set_title('QPD at unbinding', fontsize=20)
    sp = fig.add_subplot(212)   
    sp.plot(t*1000, stroke_unbinding, 'b')     
    sp.set_title('Stroke at unbinding', fontsize=20)    
    fig.savefig(os.path.join(path, 'QPD_unbinding.png'))
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





