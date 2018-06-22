"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Trap calibration (Jongmin Sung)

Bead displacement

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib import cm
from scipy import optimize


pixel_nm = 134.6 
Bead_nm = 1000
Bead_px = Bead_nm / pixel_nm
PM_mV = 20 


class Data(object):
    def __init__(self):
        pass
        
    def read_tif(self):
        data_path = os.getcwd()
        path_split = data_path.split('\\')          
        self.data_name = path_split[-1]      
        print('Data name = %r' %(self.data_name))
        file_list = os.listdir(data_path) 

        for i in range(len(file_list)):
            if file_list[i][-3:] == 'tif':
                movie_name = file_list[i]

        movie_path = data_path + '\\' + movie_name   
        movie = Image.open(movie_path)
        self.n_frame = movie.n_frames
        self.n_row = movie.size[1]
        self.n_col = movie.size[0]
        print('[frame, row, col] = [%d, %d, %d]' %(self.n_frame, self.n_row, self.n_col))

        self.I0 = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=float)
        self.I1 = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=float)
        for i in range(self.n_frame): 
            movie.seek(i) # Move to i-th frame
            self.I0[i,] = np.array(movie, dtype=float)
            I = self.I0[i,].copy()
            I_bg = np.mean([list(I[0,])+list(I[-1,])+list(I[:,0])+list(I[:,-1])])
            I = I - I_bg
            I = I / np.max(I)
            I[I<0.1] = 0
            self.I1[i,] = I
            
    def find_center(self, I):
        imax = np.where(I == np.max(I))  
        return imax[0][0], imax[1][0]  

        
    def crop(self, I, x, y, s):
        return I[x-s:x+s+1, y-s:y+s+1]

    def gaussian(self, height, center_x, center_y, width_x, width_y):
        """Returns a gaussian function with the given parameters"""
        width_x = float(width_x)
        width_y = float(width_y)
        return lambda x,y: height*np.exp(
                    -(((center_x - (x))/width_x)**2
                    +((center_y - (y))/width_y)**2)/2)


    def moments(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution by calculating its
        moments """
        total = data.sum()
        X, Y = np.indices(data.shape)
        x = (X*data).sum()/total
        y = (Y*data).sum()/total
        col = data[:, int(y)]
        width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/col.sum())
        row = data[int(x), :]
        width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/row.sum())
        height = data.max()
        return height, x, y, width_x, width_y

    def fitgaussian(self, data):
        """Returns (height, x, y, width_x, width_y)
        the gaussian parameters of a 2D distribution found by a fit"""
        params = self.moments(data)
        errorfunction = lambda p: np.ravel(self.gaussian(*p)(*np.indices(data.shape)) - data)
        p, success = optimize.leastsq(errorfunction, params)
        return p
        
    def correlation(self):
        size = int(Bead_px)  
        I0 = self.I1[0,] # 0th frame 
        cx, cy = self.find_center(I0)   
        I0s = self.crop(I0, cx, cy, size)     
        self.I0s = I0s
     
        self.corr = np.zeros((self.n_frame, self.n_row, self.n_col), dtype=float)  
        self.params = [] 
        self.fit_x = []
        self.fit_y = []

        for i in range(self.n_frame):
            I1 = self.I1[i,] # ith frame 
            cx, cy = self.find_center(I1)   
            for j in range(2*size+1):
                for k in range(2*size+1):
                    I1s = self.crop(I1, cx-size+j, cy-size+k, size)
                    corr = np.sum(I0s*I1s)
                    self.corr[i, cx-size+j, cy-size+k] = corr
            self.corr[i] = self.corr[i]/self.corr[i].max()
            self.params += [self.fitgaussian(self.corr[i])]
            self.fit_x += [self.params[-1][1]]
            self.fit_y += [self.params[-1][2]]

    def conversion(self):
        x = np.array(self.fit_x)
        y = np.array(self.fit_y)
        
        x = (x - np.mean(x)) * pixel_nm
        y = (y - np.mean(y)) * pixel_nm

        x = x - np.mean(x[(x<x.max()/5)&(x>x.min()/5)])
        y = y - np.mean(y[(y<y.max()/5)&(y>y.min()/5)])
        
        cut = 0.7
        self.x_p = np.mean(x[x>x.max()*cut]); print(self.x_p)
        self.x_n = np.mean(x[x<x.min()*cut]); print(self.x_n)       
        self.y_p = np.mean(y[y>y.max()*cut]); print(self.y_p)
        self.y_n = np.mean(y[y<y.min()*cut]); print(self.y_n) 

        self.x_fit = x*0
        self.y_fit = y*0

        self.x_fit[x>x.max()*cut] = self.x_p
        self.x_fit[x<x.min()*cut] = self.x_n
        self.y_fit[y>y.max()*cut] = self.y_p
        self.y_fit[y<y.min()*cut] = self.y_n

        self.dx = (self.x_p - self.x_n)/2
        self.dy = (self.y_p - self.y_n)/2

        self.x = x
        self.y = y


    def analyze(self):
        self.read_tif()
        self.correlation()
        self.conversion()

        
    def plot_fig1(self):                    
        # Figure 1: Bead image & Correlation
        fig1 = plt.figure(1, figsize = (20, 10), dpi=300)    
        fig1.clf()
        
        sp1 = fig1.add_subplot(231)  
        im1 = sp1.imshow(self.I0[0], cmap=cm.gray)
        plt.colorbar(im1)
        sp1.set_title('Bead (original)')

        sp2 = fig1.add_subplot(232)  
        im2 = sp2.imshow(self.I1[0], cmap=cm.gray)
        plt.colorbar(im2)
        sp2.set_title('Bead (normalized/filtered)')
        
        sp3 = fig1.add_subplot(233)  
        im3 = sp3.imshow(self.I0s, cmap=cm.gray)
        plt.colorbar(im3)
        sp3.set_title('Kernel (Frame = 0)')

        frame = 10   
        fit = self.gaussian(*self.params[frame])
        corr_fit = fit(*np.indices(self.corr[frame].shape))
        res = self.corr[frame] - corr_fit
        RMSD = (np.mean(res**2))**0.5

        sp4 = fig1.add_subplot(234)  
        im4 = sp4.imshow(self.corr[frame], cmap=cm.gray)
        plt.colorbar(im4)
        sp4.set_title('Correlation (Frame = %d)' %(frame))

        sp5 = fig1.add_subplot(235)  
        im5 = sp5.imshow(corr_fit, cmap=cm.gray)
        plt.colorbar(im5)
        sp5.set_title('Fit (x = %.1f, y = %.1f)' %(self.fit_x[frame], self.fit_y[frame]))  

        sp6 = fig1.add_subplot(236)  
        im6 = sp6.imshow(res, cmap=cm.gray)
        plt.colorbar(im6)        
        sp6.set_title('Residual (RMSD = %.3f)' %(RMSD)) 

        fig1.savefig('Fig1_Correlation.png')   
        plt.close(fig1)
        
        
    def plot_fig2(self):                    
        # Figure 2: Tracking of bead position
        fig2 = plt.figure(2, figsize = (20, 10), dpi=300)    
        fig2.clf()

        sp1 = fig2.add_subplot(211)  
        sp1.plot(self.y, 'ko', self.y_fit, 'b')
        sp1.set_title('X (%.1f nm = %.1f mV)' % (self.dy, PM_mV)) 

        sp1.set_ylabel('X (nm)') 

        sp2 = fig2.add_subplot(212)  
        sp2.plot(self.x, 'ko', self.x_fit, 'r')
        sp2.set_title('Y (%.1f nm = %.1f mV)' % (self.dx, PM_mV)) 
        sp2.set_xlabel('Frame')
        sp2.set_ylabel('Y (nm)')  
           
        fig2.savefig('Fig2_Displacement.png')   
        plt.close(fig2)
        
        
    def plot(self):
        self.plot_fig1()
        self.plot_fig2()


def main():
    data = Data()
    data.analyze()
    data.plot()

if __name__ == "__main__":
    main()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
To-do

# Rotate and elongate
# http://scipy-cookbook.readthedocs.io/items/FittingData.html  




"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
