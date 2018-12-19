# Test triangle function

import numpy as np
import matplotlib.pyplot as plt

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

t = np.linspace(0,2,1000) 

y_triangle = triangle(t, f=1, A=1, ph=0, b=0)
y_square = square(t, f=1, A=1, ph=0, b=0)
y_sine = sine(t, f=1, A=1, ph=0, b=0)
y_trapzoid = trapzoid(t, f=1, A=10, ph=0, b=0, m=0.01)


plt.figure(1)
plt.clf()
plt.plot(t, y_triangle, 'k')
plt.plot(t, y_square, 'b')
plt.plot(t, y_sine, 'r')
plt.plot(t, y_trapzoid, 'g')
plt.axhline(y=0, color='k', linestyle='dashed', lw=1)  
plt.show()
