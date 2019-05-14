# Functions used for trap analysis 

import numpy as np
import scipy 
import os
import shutil



def step(t, tb, tu, Ab, Au, s1, s2):
    return (Ab-Au) * (scipy.special.erf(s1*(t-tb)) - scipy.special.erf(s2*(t-tu)))/2 + Au


def sine(t, f, A, ph, b): # Sine function
    return A * np.sin(2*np.pi*f*t - ph) + b  


def triangle(t, f, A, ph, b):
    
    t = 2 * np.pi * f * t - ph + np.pi/2
    
    t, w = np.asarray(t), np.asarray(0.5)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (np.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (np.pi * (wsub + 1) - tsub) / (np.pi * (1 - wsub)))
    return A*y + b


def trapzoid(t, f, A, ph, b, m):
    
    t = 2 * np.pi * f * t - ph + np.pi/2
    
    t, w = np.asarray(t), np.asarray(0.5)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'
    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # take t modulo 2*pi
    tmod = np.mod(t, 2 * np.pi)

    # on the interval 0 to width*2*pi function is
    #  tmod / (pi*w) - 1
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    tsub = np.extract(mask2, tmod)
    wsub = np.extract(mask2, w)
    np.place(y, mask2, tsub / (np.pi * wsub) - 1)

    # on the interval width*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    tsub = np.extract(mask3, tmod)
    wsub = np.extract(mask3, w)
    np.place(y, mask3, (np.pi * (wsub + 1) - tsub) / (np.pi * (1 - wsub)))
    
    y[y > A*m] = A*m
    y[y < -A*m] = -A*m   
    
    return A*y + b


def square(t, f, A, ph, b):
    duty = 0.5
    t = 2 * np.pi * f * t - ph

    t, w = np.asarray(t), np.asarray(duty)
    w = np.asarray(w + (t - t))
    t = np.asarray(t + (w - w))
    if t.dtype.char in ['fFdD']:
        ytype = t.dtype.char
    else:
        ytype = 'd'

    y = np.zeros(t.shape, ytype)

    # width must be between 0 and 1 inclusive
    mask1 = (w > 1) | (w < 0)
    np.place(y, mask1, np.nan)

    # on the interval 0 to duty*2*pi function is 1
    tmod = np.mod(t, 2 * np.pi)
    mask2 = (1 - mask1) & (tmod < w * 2 * np.pi)
    np.place(y, mask2, 1)

    # on the interval duty*2*pi to 2*pi function is
    #  (pi*(w+1)-tmod) / (pi*(1-w))
    mask3 = (1 - mask1) & (1 - mask2)
    np.place(y, mask3, -1)
    return A*y + b


def exp(F, t0, dF):
    dF = abs(dF)
    return t0*np.exp(-F/dF)


def running_mean(x, N = 10): # Running mean
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    x0 = (cumsum[N:] - cumsum[:-N]) / float(N)
    x1 = np.mean(x[:N])*np.ones(int(N/2))
    x2 = np.mean(x[-N:])*np.ones(int(N/2))
    return np.concatenate((x1, x0, x2))


def running_std(x, N = 11): # Running mean
    s = np.ones(len(x))*100

    s[:int(N/2)] = np.std(x[:int(N/2)])
    s[-int(N/2):] = np.std(x[-int(N/2):])    

    for i in range(len(x)-(N-1)):
        s[i+int(N/2)] = np.std(x[i:N+i])

    if any(x == 100):
        print('wrong')

    return s


def reject_outliers(data, m = 2.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    return data[s < m]


def find_outliers(data, m = 5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    cutoff = np.median(data) + m*mdev
    i_outliers = data > cutoff
    return cutoff, i_outliers


def make_folder(name):
    path = os.path.join(os.getcwd(), name)       
    if os.path.exists(path):
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        os.makedirs(path)    
    return path

