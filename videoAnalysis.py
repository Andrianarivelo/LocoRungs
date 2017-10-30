'''
        Class to perform general transformations of membrane potential recordings such as artifact removal, 
        filtering, detrending, etc. 
        
'''

#import ezodf
import pickle
#import math, scipy.io
import pdb
#from pylab import *
#from scipy.signal import butter, lfilter, filtfilt, iirfilter, lfilter, remez, convolve, get_window
import sys
import tifffile as tiff
import matplotlib.animation as animation
import h5py
import numpy as np

from tools.h5pyTools import h5pyTools

class videoAnalysis:
    def __init__(self,ExptDay,trial,path):
        
        self.exptDay  = ExptDay
        self.trial    = trial
        self.h5pyTools = h5pyTools()
        
        fData   = h5py.File(path+'frames.ma','r')
        self.frames  = fData['data'].value 
        self.frameTimes = fData['info/0/values'].value
        
    
    def storeAsTiffFile(self,path,Nframes=None):
        if Nframes :
            tiff.imsave(path+'%s-%s_mouseImageStack.tif' % (self.exptDay,self.trial), self.frames[:Nframes])
        else:
            tiff.imsave(path+'%s-%s_mouseImageStack.tif' % (self.exptDay,self.trial), self.frames)

    def storeAsVideoFile(self,path,fps=100,dpi=200):
        
        fileName = path+'%s-%s_video.mp4' % (self.exptDay,self.trial) 
        Nframes = np.shape(self.frames)[0]
        
        dpi = 200 #self.config['videoParameter']['dpi']
        
        fig = plt.figure(111)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        im = ax.imshow(transpose(self.frames[0]),cmap='viridis',interpolation='nearest')
        #im = ax.imshow(rand(300,300),cmap='gray',interpolation='nearest')
        #im.set_clim([0,1]) # set maximal value
        fig.set_size_inches([3.73,2.8])
        
        tight_layout()

        def update_img(n):
            tmp = self.frames[n]
            im.set_data(transpose(tmp))
            #tmp = rand(300,300)
            #im.set_data(tmp)
            return im
        
        ani = animation.FuncAnimation(fig,update_img,Nframes)
        writer = animation.writers['ffmpeg'](fps=framesPerSec)
        ani.save(fileName,writer=writer,dpi=dpi)
        clf()
        
    
