'''
        Class to provide images and videos
        
'''

import numpy as np
#import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pdb
import scipy
#from pylab import *
import tifffile as tiff
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from collections import OrderedDict
#import sima
#import sima.motion
#import sima.segment
from scipy.stats.stats import pearsonr
from scipy.interpolate import interp1d
#from mtspec import mt_coherence
from scipy import stats
import matplotlib.ticker as ticker


import tools.dataAnalysis as dataAnalysis
from tools.pyqtgraph.configfile import *

# default parameter , please note that changes here don't take effect
# if config file already exists
params= OrderedDict([
    ('videoParameter',{
        'dpi': 500,
        }),
    ('projectionInTimeParameters', {
        'horizontalCuts' : [5,8,9,10,12] ,
        'verticalCut' : 50. ,
        'stimStart': 5. ,
        'stimLength' : 0.2 ,
        'fitStart' : 5. ,
        'baseLinePeriod' : 5. ,
        'threeDAspectRatio' : 3,
        'stimulationBarLocation' : -0.1,
        }),
    ('caEphysParameters', {
        'leaveOut' : 0.1,
        }),
    ])

class createVisualizations:
    
    ##########################################################################################
    def __init__(self,figureDir,mouse):

        self.mouse = mouse
        self.figureDirectory = figureDir
        if not os.path.isdir(self.figureDirectory):
            os.system('mkdir %s' % self.figureDirectory)

        configFile = self.figureDirectory + '%s.config' % mouse
        if os.path.isfile(configFile):
            self.config = readConfigFile(configFile)
        else:
            self.config = params
            writeConfigFile(self.config,configFile)

        self.pawID = ['FR', 'FL', 'HL', 'HR']

    ##########################################################################################
    def determineFileName(self,reco,what=None,date=None):
        if (what is None) and (date is None):
            ff = self.figureDirectory + '%s' % (reco)
        elif date is None:
            ff = self.figureDirectory + '%s_%s' % (reco,what)
        else:
            ff = self.figureDirectory + '%s_%s_%s' % (date,reco,what)
        return ff

    ##########################################################################################
    def layoutOfPanel(self, ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #
        if xyInvisible[0]:
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_visible(False)
        else:
            ax.spines['bottom'].set_position(('outward', 10))
            ax.xaxis.set_ticks_position('bottom')
        #
        if xyInvisible[1]:
            ax.spines['left'].set_visible(False)
            ax.yaxis.set_visible(False)
        else:
            ax.spines['left'].set_position(('outward', 10))
            ax.yaxis.set_ticks_position('left')


        if xLabel != None :
            ax.set_xlabel(xLabel)

        if yLabel != None :
            ax.set_ylabel(yLabel)

        if Leg != None :
            ax.legend(loc=Leg[0], frameon=False)
            if len(Leg)>1 :
                legend = ax.get_legend()  # plt.gca().get_legend()
                ltext = legend.get_texts()
                plt.setp(ltext, fontsize=Leg[1])

    ##########################################################################################
    def generateVideoFromImageStack(self,data,fName,framesPerSec=8):
        
        fileName = fName + '_video.mp4'
        Nframes = shape(data)[0]
        
        dpi = self.config['videoParameter']['dpi']
        
        fig = plt.figure(111)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        im = ax.imshow(data[0],cmap='gray',interpolation='nearest')
        #im.set_clim([0,1]) # set maximal value
        fig.set_size_inches([5,5])
        
        tight_layout()

        def update_img(n):
            tmp = data[n]
            im.set_data(rot90(transpose(tmp)))
            return im
        
        ani = animation.FuncAnimation(fig,update_img,Nframes)
        writer = animation.writers['ffmpeg'](fps=framesPerSec)

        ani.save(fileName,writer=writer,dpi=dpi)
        
    
    ##########################################################################################
    def saveProjectionsAsTiff(self,projo,fileName):
        
        dataTypeForTiff = uint8
        maxValue = 255
        
        
        xyFile = fileName + '_xyProjection.tiff'
        xzFile = fileName + '_xzProjection.tiff'
        yzFile = fileName + '_yzProjection.tiff'
        
        def projectToDTypeRange(dd):
            ma = dd.max()
            mi = dd.min()
            return (dd-mi)*maxValue/(ma-mi) 
        
        xyimgStackUint16 = array(projectToDTypeRange(projo['xyProjection'].value)+0.5,dtype=dataTypeForTiff)
        tiff.imsave(xyFile,xyimgStackUint16)

        xzimgStackUint16 = array(projectToDTypeRange(projo['xzProjection'].value)+0.5,dtype=dataTypeForTiff)
        tiff.imsave(xzFile,xzimgStackUint16)
        
        yzimgStackUint16 = array(projectToDTypeRange(projo['yzProjection'].value)+0.5,dtype=dataTypeForTiff)
        tiff.imsave(yzFile,yzimgStackUint16)
        
    ##########################################################################################
    def generateROIImage(self,date,rec,img,ttime,rois,raw_signals,imageMetaInfo,motionCoordinates):
        '''
            test
        '''
        #img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        #ttime = data['raw_data/caImagingTime'].value
        
        #dataSet = sima.ImagingDataset.load(self.sima_path)
        #rois = dataSet.ROIs['stica_ROIs']
        
        nRois = len(rois)
        
        plotsPerFig = 5
        nFigs = int(nRois/(plotsPerFig+1) + 1.)
        # Extract the signals.
        #dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')
        
        #raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        deltaX= imageMetaInfo[2]
        #deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ' , deltaX)
        
        # figure #################################
        fig_width = 7 # width in inches
        fig_height = 4*nFigs+8  # height in inches
        fig_size =  [fig_width,fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi' : 600,
                  'axes.linewidth' : 1.3,
                  'ytick.major.size' : 4,      # major tick size in points
                  'xtick.major.size' : 4      # major tick size in points
                  #'edgecolor' : None
                  #'xtick.major.size' : 2,
                  #'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        
        # create figure instance
        fig = plt.figure()
        
        
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2+nFigs, 1#,
                               #width_ratios=[1.2,1]
                               #height_ratios=[1,1]
                               )
        
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3,hspace=0.4)
        
        # possibly change outer margins of the figure
        #plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)
        
        # sub-panel enumerations
        #plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        
        
        # first sub-plot #######################################################
        gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],hspace=0.2)
        ax0 = plt.subplot(gssub[0])
        
        # title
        #ax0.set_title('sub-plot 1')
        
        ax0.imshow(img,cmap=plt.cm.gray,extent=[0,np.shape(img)[1]*deltaX,0,np.shape(img)[0]*deltaX])
        #pdb.set_trace()
        for n in range(len(rois)):
            x,y = rois[n].polygons[0].exterior.xy
            colors = plt.cm.jet(float(n)/float(nRois-1))
            ax0.plot(np.array(x)*deltaX,(np.shape(img)[0]-np.array(y))*deltaX,'-',c=colors,zorder=1)
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        ax0.set_xlim(0,np.shape(img)[1]*deltaX)
        ax0.set_ylim(0,np.shape(img)[0]*deltaX)
        #ax0.set_xlim()
        #ax0.set_ylim()
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        
        print(nFigs, nRois)
        # third sub-plot #######################################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1],hspace=0.2)
        # sub-panel 1 #############################################
        ax01 = plt.subplot(gssub1[0])

        # title
        # ax0.set_title('sub-plot 1')

        ax01.plot(ttime,motionCoordinates[:,1]*deltaX,label='x')
        ax01.plot(ttime,motionCoordinates[:,2]*deltaX,label='y')

        self.layoutOfPanel(ax01,xLabel='time (sec)',yLabel='displacement ($\mu$m)',Leg=[1,10])

        ax011 = plt.subplot(gssub1[1])

        # title
        # ax0.set_title('sub-plot 1')
        mask = motionCoordinates[:,1]*deltaX < 30.
        ax011.plot(motionCoordinates[:,1][mask]*deltaX,motionCoordinates[:,2][mask]*deltaX,label='x')
        #ax011.plot(time,motionCoordinates[:,2]*deltaX,label='y')

        self.layoutOfPanel(ax011,xLabel='x displacement ($\mu$m)',yLabel='y displacement ($\mu$m)',Leg=[1,10])


        # sub-panel 1 #############################################
        ax10 = plt.subplot(gs[2])
        if nFigs > 1:
            ax11= plt.subplot(gs[3])
        if nFigs > 2:
            ax12= plt.subplot(gs[4])
        if nFigs > 3:
            ax13= plt.subplot(gs[5])
        
        for n in range(len(rois)):
            fff = int(n/(plotsPerFig))
            colors = plt.cm.jet(float(n)/float(nRois-1))
            print(n, fff, nFigs)
            if fff == 0:
                ax10.plot(ttime,raw_signals[0][n],c=colors,label=str(rois[n].label + ' %.3f' % pearsonr(motionCoordinates[:,2],raw_signals[0][n])[0]))
            elif fff == 1:
                ax11.plot(ttime,raw_signals[0][n],c=colors,label=str(rois[n].label + ' %.3f' % pearsonr(motionCoordinates[:,2],raw_signals[0][n])[0]))
            elif fff == 2:
                ax12.plot(ttime,raw_signals[0][n],c=colors,label=str(rois[n].label + ' %3f' % pearsonr(motionCoordinates[:,2],raw_signals[0][n])[0]))
            elif fff == 2:
                ax13.plot(ttime,raw_signals[0][n],c=colors,label=str(rois[n].label + ' %3f' % pearsonr(motionCoordinates[:,2],raw_signals[0][n])[0]))
        
        # and moves left and bottom axes away
        ax10.spines['top'].set_visible(False)
        ax10.spines['right'].set_visible(False)
        ax10.spines['bottom'].set_position(('outward', 10))
        ax10.spines['left'].set_position(('outward', 10))
        ax10.yaxis.set_ticks_position('left')
        ax10.xaxis.set_ticks_position('bottom')
        ax10.legend(loc=1,frameon=False)
        if nFigs > 1:
            ax11.spines['top'].set_visible(False)
            ax11.spines['right'].set_visible(False)
            ax11.spines['bottom'].set_position(('outward', 10))
            ax11.spines['left'].set_position(('outward', 10))
            ax11.yaxis.set_ticks_position('left')
            ax11.xaxis.set_ticks_position('bottom')
            ax11.legend(loc=1,frameon=False)
        if nFigs > 2:
            ax12.spines['top'].set_visible(False)
            ax12.spines['right'].set_visible(False)
            ax12.spines['bottom'].set_position(('outward', 10))
            ax12.spines['left'].set_position(('outward', 10))
            ax12.yaxis.set_ticks_position('left')
            ax12.xaxis.set_ticks_position('bottom')
            ax12.legend(loc=1,frameon=False)
        if nFigs > 3:
            ax13.spines['top'].set_visible(False)
            ax13.spines['right'].set_visible(False)
            ax13.spines['bottom'].set_position(('outward', 10))
            ax13.spines['left'].set_position(('outward', 10))
            ax13.yaxis.set_ticks_position('left')
            ax13.xaxis.set_ticks_position('bottom')
            ax13.legend(loc=1,frameon=False)
        
        # legends and labels
        #plt.legend(loc=3,frameon=False)
        
        plt.xlabel('time (sec)')
        plt.ylabel('Fluorescence')
        
        # change tick spacing 
        #majorLocator_x = MultipleLocator(10)
        #ax1.xaxis.set_major_locator(majorLocator_x)
        
        # change legend text size 
        #leg = plt.gca().get_legend()
        #ltext  = leg.get_texts()
        #plt.setp(ltext, fontsize=11)
        
        ## save figure ############################################################
        fname = self.determineFileName(date,'roi_traces',reco=rec)
        
        plt.savefig(fname+'.png')
        plt.savefig(fname+'.pdf')

    ##########################################################################################
    def generateWalkCaImage(self, date, rec, img, ttime, rois, raw_signals, imageMetaInfo, motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor):
        '''
            test
        '''
        # img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        # ttime = data['raw_data/caImagingTime'].value

        # dataSet = sima.ImagingDataset.load(self.sima_path)
        # rois = dataSet.ROIs['stica_ROIs']

        nRois = len(rois)

        plotsPerFig = 5
        nFigs = int(nRois / (plotsPerFig + 1) + 1.)
        # Extract the signals.
        # dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')

        # raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        deltaX = imageMetaInfo[2]
        # deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ', deltaX)

        # figure #################################
        fig_width = 7  # width in inches
        fig_height = 4 * nFigs + 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3 + nFigs, 1  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        # plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gssub[0])

        # title
        # ax0.set_title('sub-plot 1')

        ax0.imshow(np.transpose(img),origin='lower',cmap=plt.cm.gray, extent=[0, np.shape(img)[0] * deltaX, 0, np.shape(img)[1] * deltaX])
        # pdb.set_trace()
        for n in range(len(rois)):
            x, y = rois[n].polygons[0].exterior.xy
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            ax0.plot( np.array(y) * deltaX, np.array(x) * deltaX, '-', c=colors, zorder=1)

        # removes upper and right axes
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')

        ax0.set_xlim(0, np.shape(img)[0] * deltaX)
        ax0.set_ylim(0, np.shape(img)[1] * deltaX)
        # ax0.set_xlim()
        # ax0.set_ylim()
        # legends and labels
        # plt.legend(loc=1,frameon=False)

        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')

        print(nFigs, nRois)
        # third sub-plot #######################################################
        #gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
        # sub-panel 1 #############################################
        ax01 = plt.subplot(gs[1])

        # title
        # ax0.set_title('sub-plot 1')
        #pdb.set_trace()
        walk_interp = interp1d(sTimes,linearSpeed)

        mask = (ttime>sTimes[0]) & (ttime<sTimes[-1])
        newWalking = walk_interp(ttime[mask])
        # sTimes, ttime
        ax01.plot(sTimes,linearSpeed)
        ax01.plot(ttime[mask],newWalking)
        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax01, xLabel='time (sec)', yLabel='speed (cm/s)')

        # sub-panel 1 #############################################
        ax02 = plt.subplot(gs[2])

        # title
        # ax0.set_title('sub-plot 1')


        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax02, xLabel='time (sec)', yLabel='speed (cm/s)')
        ax02.set_xlim(8,12)
        ax02.set_ylim(0,4)

        # sub-panel 1 #############################################
        ax10 = plt.subplot(gs[3])
        if nFigs > 1:
            ax11 = plt.subplot(gs[4])
        if nFigs > 2:
            ax12 = plt.subplot(gs[5])
        if nFigs > 3:
            ax13 = plt.subplot(gs[6])

        for n in range(len(rois)):
            fff = int(n / (plotsPerFig))
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            print(n, fff, nFigs)
            if n >2 and n<5:
                ax02.plot(ttime, raw_signals[0][n], c=colors)
            if fff == 0:
                ax10.plot(ttime, raw_signals[0][n], c=colors, label=str(
                    rois[n].label + ' %.3f' % pearsonr(motionCoordinates[:, 2], raw_signals[0][n])[0]))
            elif fff == 1:
                ax11.plot(ttime, raw_signals[0][n], c=colors, label=str(
                    rois[n].label + ' %.3f' % pearsonr(motionCoordinates[:, 2], raw_signals[0][n])[0]))
            elif fff == 2:
                ax12.plot(ttime, raw_signals[0][n], c=colors, label=str(
                    rois[n].label + ' %3f' % pearsonr(motionCoordinates[:, 2], raw_signals[0][n])[0]))
            elif fff == 2:
                ax13.plot(ttime, raw_signals[0][n], c=colors, label=str(
                    rois[n].label + ' %3f' % pearsonr(motionCoordinates[:, 2], raw_signals[0][n])[0]))
        ax02.plot(sTimes,linearSpeed/8.,c='k')
        # and moves left and bottom axes away
        self.layoutOfPanel(ax10, Leg=[1, 10])
        if nFigs > 1:
            self.layoutOfPanel(ax11, Leg=[1, 10])
        if nFigs > 2:
            self.layoutOfPanel(ax12, Leg=[1, 10])
        if nFigs > 3:
            self.layoutOfPanel(ax13, Leg=[1, 10])

        # legends and labels
        # plt.legend(loc=3,frameon=False)

        plt.xlabel('time (sec)')
        plt.ylabel('Fluorescence')

        # change tick spacing
        # majorLocator_x = MultipleLocator(10)
        # ax1.xaxis.set_major_locator(majorLocator_x)

        # change legend text size
        # leg = plt.gca().get_legend()
        # ltext  = leg.get_texts()
        # plt.setp(ltext, fontsize=11)

        ## save figure ############################################################
        fname = self.determineFileName(date, 'ca-walk_traces', reco=rec)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    # cV.generateWalkEphysFigure(foldersRecordings[f][0], foldersRecordings[f][1][r], currentHP, ephysTimes , angluarSpeed, linearSpeed, sTimes, timeStamp, monitor)  # plot fluorescent traces of rois

    def generateWalkEphysFigure(self, date, rec, currentHP, ephysTimes,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,spikesconv, binnedspikes, binWidth):
        '''
            test
        '''
        startE = 2.5
        endE = 4
        # img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        # ttime = data['raw_data/caImagingTime'].value

        # dataSet = sima.ImagingDataset.load(self.sima_path)
        # rois = dataSet.ROIs['stica_ROIs']

        #nRois = len(rois)

        #plotsPerFig = 5
        #nFigs = int(nRois / (plotsPerFig + 1) + 1.)
        # Extract the signals.
        # dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')

        # raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        #deltaX = imageMetaInfo[2]
        # deltaX = (data['raw_data/caImagingField'].value)[2]
        #print 'deltaX ', deltaX

        # figure #################################
        fig_width = 11  # width in inches
        fig_height = 12  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 2  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        # plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)


        # third sub-plot #######################################################
        #gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
        # sub-panel 1 #############################################
        ax00 = plt.subplot(gs[0])

        # title
        # ax0.set_title('sub-plot 1')
        #pdb.set_trace()
        #walk_interp = interp1d(sTimes,linearSpeed)

        #mask = (ttime>sTimes[0]) & (ttime<sTimes[-1])
        #newWalking = walk_interp(ttime[mask])
        # sTimes, ttime
        ax00.plot(sTimes,linearSpeed)
        #ax01.plot(ttime[mask],newWalking)
        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax00, xLabel='time (sec)', yLabel='speed (cm/s)')
        #ax01.set_xlim(2,5)


        # sub-panel 1 #############################################
        ax01 = plt.subplot(gs[1])

        # title
        # ax0.set_title('sub-plot 1')
        #pdb.set_trace()
        #walk_interp = interp1d(sTimes,linearSpeed)

        #mask = (ttime>sTimes[0]) & (ttime<sTimes[-1])
        #newWalking = walk_interp(ttime[mask])
        # sTimes, ttime
        ax01.plot(sTimes,linearSpeed)
        ax01.plot(np.arange(len(binnedspikes))*binWidth, spikesconv/10)
        #ax01.plot(ttime[mask],newWalking)
        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax01, xLabel='time (sec)', yLabel='speed (cm/s)')
        ax01.set_xlim(startE, endE)

        # sub-panel 1 #############################################
        ax10 = plt.subplot(gs[2])

        # title
        # ax0.set_title('sub-plot 1')
        ax10.plot(ephysTimes,currentHP)




        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax10, xLabel='time (sec)', yLabel='current (A)')

        # sub-panel 1 #############################################
        ax11 = plt.subplot(gs[3])

        # title
        # ax0.set_title('sub-plot 1')
        ax11.plot(ephysTimes,currentHP)




        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax11, xLabel='time (sec)', yLabel='current (A)')

        ax11.set_xlim(startE, endE)

        # change tick spacing
        # majorLocator_x = MultipleLocator(10)
        # ax1.xaxis.set_major_locator(majorLocator_x)

        # change legend text size
        # leg = plt.gca().get_legend()
        # ltext  = leg.get_texts()
        # plt.setp(ltext, fontsize=11)

        # sub-panel 1 #############################################
        ax20 = plt.subplot(gs[4])

        # title
        # ax0.set_title('sub-plot 1')
        ax20.plot(np.arange(len(binnedspikes))*binWidth, spikesconv)

        # ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax20, xLabel='time (sec)', yLabel='rate (spk/sec)')

        # sub-panel 1 #############################################
        ax21 = plt.subplot(gs[5])

        # title
        # ax0.set_title('sub-plot 1')
        ax21.plot(np.arange(len(binnedspikes))*binWidth, spikesconv)

        # ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        self.layoutOfPanel(ax21, xLabel='time (sec)', yLabel='rate (spk/sec)')
        ax21.set_xlim(startE, endE)
        # change tick spacing
        # majorLocator_x = MultipleLocator(10)
        # ax1.xaxis.set_major_locator(majorLocator_x)

        # change legend text size
        # leg = plt.gca().get_legend()
        # ltext  = leg.get_texts()
        # plt.setp(ltext, fontsize=11)

        ## save figure ############################################################
        fname = self.determineFileName('ephys-walk_traces',None,reco=rec)

        #plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def generateCaWheelPawImage(self, mouse, allCorrDataPerSession):

        # exclude aborted recordings
        #allCorrDataPerSession[2][1][1][4] = True # 2019.03.08_000, second trial
        #pawTracesInclude = np.ones((len(allCorrDataPerSession),5))
        #allCorrDataPerSession[8][1][1][4] = True # 2019.03.18_000, second trial


        for nSess in range(len(allCorrDataPerSession)):

            # calcium related data
            fTraces = allCorrDataPerSession[nSess][3][0][0]
            timeStamps = allCorrDataPerSession[nSess][3][0][3]
            trials = np.unique(timeStamps[:, 1])

            # wheel speed
            wheelTracks = allCorrDataPerSession[nSess][1]

            # for each session, find pre-motorization period during which there is the leas amount of movement = locomotion
            minMovement = 1000000.
            for n in range(len(trials)):

                mask = (timeStamps[:, 1] == trials[n])
                triggerStart = timeStamps[:, 5][mask]
                trialStartUnixTimes.append(timeStamps[:, 3][mask][0])

                #wheelTracks[n][2],wheelTracks[n][1]
                mask = (wheelTracks[n][2] < 5.)
                meanMovement = np.mean(np.abs(wheelTracks[n][1][mask]))
                if meanMovement < minMovement:
                    minMovement = np.copy(meanMovement)
                    minTrial = n

            print('trial with smallest movement :',n)

            ##########################################
            trialStartUnixTimes = []
            # figure #################################
            fig_width = 30  # width in inches
            fig_height = 30  # height in inches
            fig_size = [fig_width, fig_height]
            params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                      'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                      'xtick.major.size': 4  # major tick size in points
                      # 'edgecolor' : None
                      # 'xtick.major.size' : 2,
                      # 'ytick.major.size' : 2,
                      }
            rcParams.update(params)

            # set sans-serif font to Arial
            rcParams['font.sans-serif'] = 'Arial'

            # create figure instance
            fig = plt.figure()

            # define sub-panel grid and possibly width and height ratios
            gs = gridspec.GridSpec(3, 1,  # ,
                                   # width_ratios=[1.2,1]
                                   height_ratios=[10,1,3]
                                   )

            # define vertical and horizontal spacing between panels
            gs.update(wspace=0.3, hspace=0.2)

            # possibly change outer margins of the figure
            plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.05)

            # sub-panel enumerations
            plt.figtext(0.12, 0.97, 'mouse : %s, session : %s' % (mouse,allCorrDataPerSession[nSess][0]),clip_on=False,color='black', weight='bold',size=22)

            # ca-traces #######################################################


            #triggerStarts = np.unique(timeStamps[:, 5])

            gssub0 = gridspec.GridSpecFromSubplotSpec(1, len(trials), subplot_spec=gs[0], hspace=0.1,wspace=0.1)
            #print(trials)
            #xSeparation = 35.
            ySpacing = 30.
            for n in range(len(trials)):
                ax0 = plt.subplot(gssub0[n])
                mask = (timeStamps[:, 1] == trials[n])
                triggerStart = timeStamps[:, 5][mask]
                trialStartUnixTimes.append(timeStamps[:,3][mask][0])
                if n>0:
                    if oldTriggerStart>triggerStart[0]:
                        print('problem in trial order')
                        sys.exit(1)
                for i in range(len(fTraces)):
                    #colors = plt.cm.jet(float(i) / float(len(fTraces) - 1))
                    ax0.plot(timeStamps[:,4][mask]-triggerStart,fTraces[i][mask]+i*ySpacing)#,color=np.random.rand())

                if n == 0:
                    self.layoutOfPanel(ax0, xLabel=r'time (s)', yLabel=r'fluorescence')
                else:
                    self.layoutOfPanel(ax0, xLabel=r'time (s)', yLabel=None,xyInvisible=[False,True])
                ax0.set_xlim(0,30)
                oldTriggerStart=triggerStart[0]

            # wheel speed  ######################################################
            gssub1 = gridspec.GridSpecFromSubplotSpec(1, len(trials), subplot_spec=gs[1], hspace=0.1,wspace=0.1)
            #pdb.set_trace()

            nFig = 0
            #print(len(wheelTracks))
            for n in range(len(wheelTracks)):
                wheelRecStartTime = wheelTracks[n][3]
                if (trialStartUnixTimes[nFig]-wheelRecStartTime)<1.:
                    #if not wheelTracks[n][4]:
                    #recStartTime = wheelTracks[0][3]
                    if nFig>0:
                        if oldRecStartTime>wheelRecStartTime:
                            print('problem in trial order')
                            sys.exit(1)
                    ax1 = plt.subplot(gssub1[nFig])
                    ax1.axhline(y=0,c='0.7',ls='--')
                    ax1.plot(wheelTracks[n][2],wheelTracks[n][1],c='0.4')

                    if nFig == 0:
                        self.layoutOfPanel(ax1, xLabel=r'time (s)', yLabel=r'wheel speed (cm/s)')
                    else:
                        self.layoutOfPanel(ax1, xLabel=r'time (s)', yLabel=None, xyInvisible=[False,True])
                    ax1.set_xlim(0,30)
                    ax1.set_ylim(-10,30)
                    nFig+=1
                    oldRecStartTime = wheelRecStartTime
            # paw speed  ######################################################
            gssub2 = gridspec.GridSpecFromSubplotSpec(4, len(trials), subplot_spec=gs[2], hspace=0.1,wspace=0.1)
            cc = ['C0','C1','C2','C3']
            pawTracks = allCorrDataPerSession[nSess][2]
            #print(len(pawTracks))
            #pdb.set_trace()
            nFig = 0
            for n in range(len(pawTracks)):
                #if not wheelTracks[n][4]:
                pawRecStartTime = pawTracks[n][4]
                if (trialStartUnixTimes[nFig]-pawRecStartTime)<1.:
                    if nFig>0:
                        if oldRecStartTime>pawRecStartTime:
                            print('problem in trial order')
                            sys.exit(1)
                    ax2 = plt.subplot(gssub2[nFig])
                    ax3 = plt.subplot(gssub2[nFig+len(trials)])
                    ax4 = plt.subplot(gssub2[nFig+2*len(trials)])
                    ax5 = plt.subplot(gssub2[nFig+3*len(trials)])
                    for i in range(4):
                        #pdb.set_trace()
                        pawSpeed = pawTracks[n][3][i]
                        #print(n,i)
                        #frDisplOrig = np.sqrt((np.diff(pawTracks[n][0][:,(i*3+1)][pawMask])) ** 2 + (np.diff(pawTracks[n][0][:,(i*3+2)][pawMask])) ** 2) / np.diff(pawTracks[n][0][:,0][pawMask]) # pawTracks[n][0][:,1]
                        if i==0:
                            ax2.plot(pawSpeed[:,0],pawSpeed[:,1],label='%s' % pawTracks[n][2][i][0],c=cc[i])
                        elif i==1:
                            ax3.plot(pawSpeed[:,0],pawSpeed[:,1],label='%s'% pawTracks[n][2][i][0],c=cc[i])
                        elif i==2:
                            ax4.plot(pawSpeed[:,0],pawSpeed[:,1],label='%s'% pawTracks[n][2][i][0],c=cc[i])
                        elif i==3:
                            ax5.plot(pawSpeed[:,0],pawSpeed[:,1],label='%s'% pawTracks[n][2][i][0],c=cc[i])
                    #ax0.axhline(y=0, c='0.6', ls='--')
                    #ax1.plot(wheelTracks[n][2], wheelTracks[n][1], c='0.3')

                    if nFig == 0:
                        self.layoutOfPanel(ax2, xLabel=None, yLabel=None,Leg=[1,9],xyInvisible=[True, False])
                        self.layoutOfPanel(ax3, xLabel=None, yLabel=r'paw speed (px/s)',Leg=[1,9],xyInvisible=[True, False])
                        self.layoutOfPanel(ax4, xLabel=None, yLabel=None,Leg=[1,9],xyInvisible=[True, False])
                        self.layoutOfPanel(ax5, xLabel=r'time (s)', yLabel=None,Leg=[1,9])
                    else:
                        self.layoutOfPanel(ax2, xLabel=None, yLabel=None, xyInvisible=[True, True])
                        self.layoutOfPanel(ax3, xLabel=None, yLabel=None, xyInvisible=[True, True])
                        self.layoutOfPanel(ax4, xLabel=None, yLabel=None, xyInvisible=[True, True])
                        self.layoutOfPanel(ax5, xLabel=r'time (s)', yLabel=None,xyInvisible=[False,True])
                    ax2.set_xlim(0,30)
                    ax3.set_xlim(0,30)
                    ax4.set_xlim(0,30)
                    ax5.set_xlim(0,30)
                    oldRecStartTime = pawRecStartTime
                    nFig+=1

            ## save figure ############################################################
            date = '2019.06.04'
            fname = self.determineFileName(allCorrDataPerSession[nSess][0], 'ca-walk-paw_dynamics', date=None)

            plt.savefig(fname + '.png')
            plt.savefig(fname + '.pdf')

            #pdb.set_trace()

    ##########################################################################################
    def generateCorrelationPlotsCaWheelPaw(self,mouse,correlationData, allCorrDataPerSession):

        minMaxCa    = [1,-1]
        minMaxWheel = [1,-1]
        minMaxPaw   = [1,-1]
        # find maxima and minima
        for nSess in range(len(correlationData)):
            if np.min(correlationData[nSess][1][:,3])<minMaxCa[0]:
                minMaxCa[0]=np.min(correlationData[nSess][1][:,3])
            if np.max(correlationData[nSess][1][:,3])>minMaxCa[1]:
                minMaxCa[1]=np.max(correlationData[nSess][1][:,3])
            if np.min(correlationData[nSess][2][:,1])<minMaxWheel[0]:
                minMaxWheel[0]=np.min(correlationData[nSess][2][:,1])
            if np.max(correlationData[nSess][2][:,1])>minMaxWheel[1]:
                minMaxWheel[1]=np.max(correlationData[nSess][2][:,1])
            for i in range(4):
                if np.min(correlationData[nSess][3][:,2*i+1])<minMaxPaw[0]:
                    minMaxPaw[0] =np.min(correlationData[nSess][3][:,2*i+1])
                if np.max(correlationData[nSess][3][:,2*i+1])>minMaxPaw[1]:
                    minMaxPaw[1]=np.max(correlationData[nSess][3][:,2*i+1])



        # figure #################################
        fig_width = 33  # width in inches
        fig_height = 30  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(len(correlationData), 7,  # ,
                               width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               #height_ratios=[10, 1, 3])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.2)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.06, right=0.94, top=0.97, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.985, 'Correlations mouse : %s' % (mouse), clip_on=False, color='black', weight='bold', size=22)

        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])

        for nSess in range(len(correlationData)):
            axList = []
            for i in range(7):
                ax=plt.subplot(gs[nSess*7+i])
                axList.append(ax)
            # ax0 = plt.subplot(gs[nSess*6])
            # ax1 = plt.subplot(gs[nSess*6+1])
            # ax2 = plt.subplot(gs[nSess*6+2])
            # ax3 = plt.subplot(gs[nSess*6+3])
            # ax4 = plt.subplot(gs[nSess*6+4])
            # ax5 = plt.subplot(gs[nSess*6+5])
            #trialStartUnixTimes = []

            # inter ca-trace correlations #######################################################
            axList[0].axvline(x=0,ls='--',c='0.6')
            axList[0].hist(correlationData[nSess][1][:,3],bins=100,range=[minMaxCa[0],minMaxCa[1]])

            # ca-trace wheel correlations #######################################################
            axList[1].axvline(x=0,ls='--',c='0.6')
            axList[1].hist(correlationData[nSess][2][:,1],bins=20,range=[minMaxWheel[0],minMaxWheel[1]])

            # ca-trace paw correlations #######################################################
            for i in range(4):
                axList[i+2].axvline(x=0, ls='--', c='0.6')
                axList[i+2].hist(correlationData[nSess][3][:,2*i+1],bins=20,range=[minMaxPaw[0],minMaxPaw[1]])

            sortedPawCorrs = []
            for j in range(len(correlationData[nSess][3])):
                soso = np.sort(correlationData[nSess][3][j,[1,3,5,7]])
                axList[6].plot(np.arange(4),soso-soso[0],'o-',lw=0.5)
                sortedPawCorrs.append(soso)
            sortPawCorrs = np.asarray(sortedPawCorrs)
            axList[6].plot(np.arange(4),np.mean(sortPawCorrs,axis=0)-np.mean(sortPawCorrs,axis=0)[0],'o-',c='black',lw=2)
            for i in range(3):
                print('paired t-test [%s,%s] : ' % (i,(i+1)), scipy.stats.ttest_rel(sortPawCorrs[:,i],sortPawCorrs[:,(i+1)]))



            # layout settings
            if nSess==0:
                axList[0].set_title('pairwise ca-traces')
                axList[1].set_title('wheel speed - ca-traces')
                for i in range(4):
                    axList[i+2].set_title('paw speed - ca-traces')
                # ax2.set_title('paw speed - ca-traces')
                # ax3.set_title('paw speed - ca-traces')
                # ax4.set_title('paw speed - ca-traces')
                # ax5.set_title('paw speed - ca-traces')
                axList[6].set_title('wheel speed - ca-traces : ordered')

            if nSess < (len(correlationData)-1):
                for i in range(6):
                    self.layoutOfPanel(axList[i], xLabel=None, yLabel=('%s\nnumber of pairs' % allCorrDataPerSession[nSess][0] if i==0 else None), xyInvisible=[False, False])
                self.layoutOfPanel(axList[6], xLabel=None, yLabel='diff in correlation', xyInvisible=[False, False])
            else:
                for i in range(6):
                    self.layoutOfPanel(axList[i], xLabel=r'correlation', yLabel=('%s\nnumber of pairs' % allCorrDataPerSession[nSess][0] if i==0 else None))
                self.layoutOfPanel(axList[6], xLabel='ordered paws', yLabel='diff in correlation', xyInvisible=[False, False])



        ## save figure ############################################################
        date = '2019.06.16'
        fname = self.determineFileName(date, 'correlations_ca-walk-paw', date=None)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

            #pdb.set_trace()

    ##########################################################################################
    def generateCorrelationPlotCaTraces(self,mouse,correlationData, allCorrDataPerSession):

        minMaxCa    = [1,-1]
        minMaxWheel = [1,-1]
        minMaxPaw   = [1,-1]
        # find maxima and minima
        allCorrEffs = []
        allEuclDist = []
        allXYDist   = []
        for nSess in range(len(correlationData)):
            allCorrEffs.extend(correlationData[nSess][1][:,3])
            allEuclDist.extend(correlationData[nSess][1][:,5])
            allXYDist.extend(correlationData[nSess][1][:,6])
        #pdb.set_trace()
        allCorrEffs = np.asarray(allCorrEffs)
        allEuclDist = np.asarray(allEuclDist)
        allXYDist = np.asarray(allXYDist)

        # figure #################################
        fig_width = 6  # width in inches
        fig_height = 25  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(6,1,  # ,
                               #width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               #height_ratios=[10, 1, 3])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.1)

        # sub-panel enumerations
        plt.figtext(0.06, 0.985, 'Correlations mouse : %s' % (mouse), clip_on=False, color='black', weight='bold', size=12)

        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])
        ##################################
        ax0=plt.subplot(gs[0])

        ax0.hist(allCorrEffs,bins=100,histtype='stepfilled')
        ax0.axvline(x=0,ls='--',c='0.7')
        ax0.axvline(x=np.mean(allCorrEffs),lw=2,color='C1')
        print(np.mean(allCorrEffs))
        self.layoutOfPanel(ax0, xLabel='correlation', yLabel='number of pairs')

        ##################################
        ax1=plt.subplot(gs[1])

        ax1.hist(allEuclDist,bins=100,histtype='stepfilled')
        self.layoutOfPanel(ax1, xLabel=u'distance (\u03BCm)', yLabel='number of pairs')

        ##################################
        ax2=plt.subplot(gs[2])

        bb = np.linspace(np.min(allEuclDist),np.max(allEuclDist),30)
        bbMean = np.zeros((4,len(bb)-1))
        for i in range(len(bb)-1):
            mask = (allEuclDist>= bb[i]) & (allEuclDist<=bb[i+1])
            bbMean[0,i] = (bb[i]+bb[i+1])/2.
            if sum(mask)>0:
                bbMean[1,i] = np.mean(allCorrEffs[mask])
                bbMean[2,i] = np.std(allCorrEffs[mask])
                #bbMean[2,i] = np.percentile(allCorrEffs[mask],5)
                bbMean[3,i] = np.percentile(allCorrEffs[mask], 95)

        ax2.axhline(y=0,c='0.5',ls='--')
        #ax2.plot(allEuclDist,allCorrEffs,'.',ms=0.6,alpha=0.5)
        #ax2.errorbar(bbMean[0],bbMean[1],yerr=np.row_stack((bbMean[1]-bbMean[2],bbMean[3]-bbMean[1])))
        ax2.fill_between(bbMean[0],bbMean[1]+bbMean[2],bbMean[1]-bbMean[2],color='0.6')
        #ax2.errorbar(bbMean[0],bbMean[1],yerr=bbMean[2])
        ax2.plot(bbMean[0],bbMean[1],color='black')

        self.layoutOfPanel(ax2, xLabel=u'distance (\u03BCm)', yLabel='correlation')

        ##################################
        ax3=plt.subplot(gs[3])

        #ax3.scatter(allXYDist[:,0],allXYDist[:,1],s=1,c=allCorrEffs,alpha=0.5)

        self.layoutOfPanel(ax3, xLabel=u'x distance (\u03BCm)', yLabel=u'y distance (\u03BCm)')

        ##################################
        ax4=plt.subplot(gs[4])

        xTile = np.linspace(np.min(allXYDist[:,0]),np.max(allXYDist[:,0]),20)
        yTile = np.linspace(np.min(allXYDist[:,1]),np.max(allXYDist[:,1]),20)
        tiledValues = np.zeros((len(xTile)-1,len(yTile)-1))
        for i in range(len(xTile)-1):
            for j in range(len(yTile)-1):
                maskX = (allXYDist[:,0]>=xTile[i]) & (allXYDist[:,0]<=xTile[i+1])
                maskY = (allXYDist[:,1]>=yTile[j]) & (allXYDist[:,1]<=yTile[j+1])
                mask = maskX & maskY
                tiledValues[i,j] = np.mean(allCorrEffs[mask])



        imgplot = ax4.imshow(tiledValues,origin='lower',aspect=1,extent=[xTile[0],xTile[-1],yTile[0],yTile[-1]])
        cb = plt.colorbar(imgplot)

        self.layoutOfPanel(ax4, xLabel=u'x distance (\u03BCm)', yLabel=u'y distance (\u03BCm)')

        ##################################
        ax5 = plt.subplot(gs[5])
        dx = xTile[1]-xTile[0]
        dy = xTile[1]-yTile[0]
        ax5.plot(xTile[2:-3]+dx/2.,np.mean(tiledValues[2:-2,2:-2],axis=0))
        ax5.plot(yTile[2:-3]+dy/2.,np.mean(tiledValues[2:-2,2:-2],axis=1))

        self.layoutOfPanel(ax5, xLabel=u'x/y distance (\u03BCm)', yLabel=u'mean correlation')

        ## save figure ############################################################
        date = '2019.07.03'
        fname = self.determineFileName(date, 'correlations-ca', date=None)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

            #pdb.set_trace()

    ##########################################################################################
    def generatePCACorrelationPlot(self,mouse,correlationData, allCorrDataPerSession,varExplained):

        minMaxCa    = [1,-1]
        minMaxWheel = [1,-1]
        minMaxPaw   = [1,-1]
        # find maxima and minima
        allCorrEffs = []
        allEuclDist = []
        allXYDist   = []
        for nSess in range(len(correlationData)):
            allCorrEffs.extend(correlationData[nSess][1][:,3])
            allEuclDist.extend(correlationData[nSess][1][:,5])
            allXYDist.extend(correlationData[nSess][1][:,6])
        #pdb.set_trace()
        allCorrEffs = np.asarray(allCorrEffs)
        allEuclDist = np.asarray(allEuclDist)
        allXYDist = np.asarray(allXYDist)

        # figure #################################
        fig_width = 6  # width in inches
        fig_height = 25  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(6,1,  # ,
                               #width_ratios=[1,1,0.8,0.8,0.8,0.8,0.8])
                               #height_ratios=[10, 1, 3])
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.94, top=0.95, bottom=0.1)

        # sub-panel enumerations
        plt.figtext(0.06, 0.985, 'PCA correlations mouse : %s' % (mouse), clip_on=False, color='black', weight='bold', size=12)

        # sessionCorrelations.append([nSess, ppCaTraces, corrWheel, corrPaws])
        ##################################
        ax0=plt.subplot(gs[0])
        varTemp = np.asarray(varExplained)
        varExp = np.transpose(varTemp)
        ax0.plot(np.arange(len(varExp))+1,varExp,'o',c='0.5',ms=4)
        ax0.plot(np.arange(len(varExp))+1,np.average(varExp,axis=1),'D-',lw=3,ms=10,color='purple')
        ax0.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.layoutOfPanel(ax0, xLabel='PCA component #', yLabel='fraction of variance explained')

        ##################################
        ax1=plt.subplot(gs[1])
        #pdb.set_trace()
        barWidth=0.1
        ddd = 0.8
        corrs = np.zeros((5,len(varExp),len(correlationData)))
        for nSess in range(len(correlationData)):
            pcaCorrs = np.asarray(correlationData[nSess][4])
            for i in range(len(varExp)):
                corrs[:,i,nSess] = pcaCorrs[i,1::2]
                #ax1.plot(i+1+np.arange(5)/10.,pcaCorrs[i,1::2],'o',c='0.6')
        ax1.axhline(y=0,ls='--',color='0.8')
        ax1.bar(np.arange(len(varExp))+ddd, np.mean(corrs[0],axis=1),color='0.3', width=barWidth, edgecolor='white',label='wheel')
        ax1.bar(np.arange(len(varExp))+ddd+barWidth, np.mean(corrs[1],axis=1),color='C0',width=barWidth, edgecolor='white',label='FL paw')
        ax1.bar(np.arange(len(varExp))+ddd+2*barWidth, np.mean(corrs[2],axis=1),color='C1', width=barWidth, edgecolor='white',label='FR paw')
        ax1.bar(np.arange(len(varExp))+ddd+3*barWidth, np.mean(corrs[3],axis=1),color='C2', width=barWidth, edgecolor='white',label='HR paw')
        ax1.bar(np.arange(len(varExp))+ddd+4*barWidth, np.mean(corrs[4],axis=1),color='C3', width=barWidth, edgecolor='white',label='HL paw')
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        self.layoutOfPanel(ax1, xLabel=u'PCA component #', yLabel='mean correlation',Leg=[1,9])

        ##################################
        ax2=plt.subplot(gs[2])



        self.layoutOfPanel(ax2, xLabel=u'distance (\u03BCm)', yLabel='correlation')

        ##################################
        ax3=plt.subplot(gs[3])



        self.layoutOfPanel(ax3, xLabel=u'x distance (\u03BCm)', yLabel=u'y distance (\u03BCm)')

        ##################################
        ax4=plt.subplot(gs[4])




        self.layoutOfPanel(ax4, xLabel=u'x distance (\u03BCm)', yLabel=u'y distance (\u03BCm)')

        ##################################
        ax5 = plt.subplot(gs[5])


        ## save figure ############################################################
        date = '2019.07.03'
        fname = self.determineFileName(date, 'pca-correlations', date=None)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

            #pdb.set_trace()


    ##########################################################################################
    def generateWalkCaCorrelationsImage(self, date, rec, img, ttime, rois, raw_signals, imageMetaInfo, motionCoordinates,angluarSpeed, linearSpeed, sTimes, timeStamp, monitor):
        '''
            test
        '''
        # img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        # ttime = data['raw_data/caImagingTime'].value

        # dataSet = sima.ImagingDataset.load(self.sima_path)
        # rois = dataSet.ROIs['stica_ROIs']

        dt = np.mean(np.diff(ttime))

        nRois = len(rois)

        plotsPerFig = 5
        nFigs = nRois + 2
        # Extract the signals.
        # dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')

        # raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        deltaX = imageMetaInfo[2]
        # deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ', deltaX)

        # figure #################################
        fig_width = 7  # width in inches
        fig_height = 4 * nFigs + 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(nFigs, 1  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.14, right=0.92, top=0.98, bottom=0.05)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)


        # third sub-plot #######################################################
        # gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
        # sub-panel 1 #############################################
        gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gssub[0])

        # title
        # ax0.set_title('sub-plot 1')

        ax0.imshow(np.transpose(img), origin='lower', cmap=plt.cm.gray,
                   extent=[0, np.shape(img)[0] * deltaX, 0, np.shape(img)[1] * deltaX])
        # pdb.set_trace()
        for n in range(len(rois)):
            x, y = rois[n].polygons[0].exterior.xy
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            ax0.plot(np.array(y) * deltaX, np.array(x) * deltaX, '-', c=colors, zorder=1)

        # removes upper and right axes
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')

        ax0.set_xlim(0, np.shape(img)[0] * deltaX)
        ax0.set_ylim(0, np.shape(img)[1] * deltaX)
        # ax0.set_xlim()
        # ax0.set_ylim()
        # legends and labels
        # plt.legend(loc=1,frameon=False)

        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')

        #######################################################
        ax01 = plt.subplot(gs[1])

        # title
        # ax0.set_title('sub-plot 1')
        walk_interp = interp1d(sTimes,linearSpeed)

        mask = (ttime>sTimes[0]) & (ttime<sTimes[-1])
        newWalking = walk_interp(ttime[mask])


        for n in range(len(rois)):
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            #print n, fff, nFigs
            ccW = dataAnalysis.crosscorr(dt,raw_signals[0][n][mask],newWalking)
            ax01.plot(ccW[:,0],ccW[:,1], c=colors, label=str(n)+ ' , ' + str(rois[n].label))
        ccW = dataAnalysis.crosscorr(dt,raw_signals[0][n][mask],motionCoordinates[:,2][mask]*deltaX)
        ax01.plot(ccW[:,0],ccW[:,1], c='k', label='motion')
        self.layoutOfPanel(ax01, xLabel='time (sec)', yLabel='correlation', Leg=[1, 10])

        # sub-panel 1 #############################################
        for n in range(len(rois)):
            ax2 = plt.subplot(gs[n+2])
            ax2.set_title('roi %s ID %s' % (n,rois[n].label))
            for m in range(len(rois)):
                colors = plt.cm.jet(float(m) / float(nRois - 1))
                ccW = dataAnalysis.crosscorr(dt,raw_signals[0][n],raw_signals[0][m])
                ax2.plot(ccW[:,0],ccW[:,1], c=colors, label=str(m)+' , '+str(rois[m].label))
            self.layoutOfPanel(ax2,xLabel='time (sec)', yLabel='correlation', Leg=[1, 10])

        ## save figure ############################################################
        fname = self.determineFileName(date, 'ca-walk_correlations', reco=rec)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def generateWalkCaSpectralAnalysis(self, date, rec, img, ttime, rois, raw_signals, imageMetaInfo,
                                        motionCoordinates, angluarSpeed, linearSpeed, sTimes, timeStamp, monitor):
        '''
            test
        '''
        # img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        # ttime = data['raw_data/caImagingTime'].value

        # dataSet = sima.ImagingDataset.load(self.sima_path)
        # rois = dataSet.ROIs['stica_ROIs']

        dt = np.mean(np.diff(ttime))

        nRois = len(rois)

        plotsPerFig = 5
        nFigs = nRois + 2
        # Extract the signals.
        # dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')

        # raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        deltaX = imageMetaInfo[2]
        # deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ', deltaX)

        # figure #################################
        fig_width = 6  # width in inches
        fig_height = 20 # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(6, 1  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.14, right=0.92, top=0.98, bottom=0.05)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # third sub-plot #######################################################
        # gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
        # sub-panel 1 #############################################
        gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gssub[0])

        # title
        # ax0.set_title('sub-plot 1')

        ax0.imshow(np.transpose(img), origin='lower', cmap=plt.cm.gray,
                   extent=[0, np.shape(img)[0] * deltaX, 0, np.shape(img)[1] * deltaX])
        # pdb.set_trace()
        for n in range(len(rois)):
            x, y = rois[n].polygons[0].exterior.xy
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            ax0.plot(np.array(y) * deltaX, np.array(x) * deltaX, '-', c=colors, zorder=1)

        self.layoutOfPanel(ax0, xLabel=r'x ($\mu$m)', yLabel=r'y ($\mu$m)')

        ax0.set_xlim(0, np.shape(img)[0] * deltaX)
        ax0.set_ylim(0, np.shape(img)[1] * deltaX)

        #######################################################
        ax11 = plt.subplot(gs[1])
        ax12 = plt.subplot(gs[2])
        ax02 = plt.subplot(gs[3])
        ax03 = plt.subplot(gs[4])
        ax04 = plt.subplot(gs[5])
        # title
        # ax0.set_title('sub-plot 1')
        walk_interp = interp1d(sTimes, linearSpeed)

        mask = (ttime > sTimes[0]) & (ttime < sTimes[-1])
        newWalking = walk_interp(ttime[mask])
        # float Time-bandwidth product. Common values are 2, 3, 4 and numbers in between.
        tbp = 2
        # integer, optional Number of tapers to use. Defaults to int(2*time_bandwidth) - 1. This is maximum senseful amount. More tapers will have no great influence on the final spectrum but increase the calculation time. Use fewer tapers for a faster calculation.
        kspec = 7
        # float; confidence for null hypothesis test, e.g. .95
        p = 0.95

        for n in range(len(rois)):
            colors = plt.cm.jet(float(n) / float(nRois - 1))
            # print n, fff, nFigs
            nf = len(raw_signals[0][n][mask])/2 + 1
            out = mt_coherence(dt, raw_signals[0][n][mask], newWalking, tbp, kspec, nf, p, freq=True,
                                        cohe=True, phase=True, speci=True, specj=True, iadapt=1)
            max_freq = 100.
            f_mask = out['freq'] <= max_freq
            frequency = out['freq'][f_mask]
            if n<4:
                ax11.plot(frequency[1:-1], np.convolve(10.*np.log10(out['speci'][f_mask]),np.ones((3,))/3, mode='valid'), c=colors, label=str(n) + ' , ' + str(rois[n].label))
                ax03.plot(frequency[1:-1], np.convolve(out['cohe'][f_mask],np.ones((3,))/3, mode='valid'), c=colors, label=str(n) + ' , ' + str(rois[n].label))
            else:
                ax12.plot(frequency[1:-1], np.convolve(10.*np.log10(out['speci'][f_mask]),np.ones((3,))/3, mode='valid'), c=colors, label=str(n) + ' , ' + str(rois[n].label))
                ax04.plot(frequency[1:-1], np.convolve(out['cohe'][f_mask],np.ones((3,))/3, mode='valid'), c=colors, label=str(n) + ' , ' + str(rois[n].label))
            ax02.plot(frequency[1:-1], np.convolve(10.*np.log10(out['specj'][f_mask]),np.ones((3,))/3, mode='valid'), c=colors, label=str(n) + ' , ' + str(rois[n].label))


        self.layoutOfPanel(ax11, xLabel='frequency (Hz)', yLabel='PSD (dB/Hz)', Leg=[1, 10])
        self.layoutOfPanel(ax12, xLabel='frequency (Hz)', yLabel='PSD (dB/Hz)', Leg=[1, 10])
        self.layoutOfPanel(ax02, xLabel='frequency (Hz)', yLabel='PSD (dB/Hz)', Leg=[1, 10])
        self.layoutOfPanel(ax03, xLabel='frequency (Hz)', yLabel='PSD (dB/Hz)', Leg=[1, 10])
        self.layoutOfPanel(ax04, xLabel='frequency (Hz)', yLabel='PSD (dB/Hz)', Leg=[1, 10])


        ## save figure ############################################################
        fname = self.determineFileName(date, 'ca-walk_spectral', reco=rec)

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def generateWalkingFigure(self,mouse, date,tracks):

        nRecs = len(tracks)

        tracksN = []
        highResRecs = 0
        maxSpeed = 0.
        minSpeed = 0.
        for i in range(nRecs):
            if tracks[i][4]:
                tracksN.append([i,tracks[i][3],True])
            else:
                highResRecs+=1
                tracksN.append([i,tracks[i][3],False])
                if max(tracks[i][1])>maxSpeed:
                    maxSpeed =max(tracks[i][1])
                if min(tracks[i][1])<minSpeed:
                    minSpeed = min(tracks[i][1])
        plotsPerFig = 3
        nFigs = int(highResRecs / (plotsPerFig))

        # sort according to earliest experiment
        tracksN.sort(key=lambda x: x[1])
        #pdb.set_trace()
        startTime = tracksN[0][1]
        print('monitor N : ', tracksN)

        # figure #################################
        fig_width = 12  # width in inches
        fig_height = 4 * (nFigs+2)  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2 + nFigs, 1 # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.1, 0.97, 'mouse : %s, recording : %s' % (mouse,date),clip_on=False,color='black',size=18)

        # first overview plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gs[0])
        ax0.axhline(y=0,ls='--',c='0.5')
        for i in range(nRecs):
            colors = plt.cm.jet(float(i) / float(nRecs - 1))
            if tracks[i][4]:
                timeDiff = np.diff(tracks[i][2])
                pausesIndex = np.where(timeDiff>30.)[0]
                #print np.shape(pausesIndex)
                #pdb.set_trace()
                pausesIndex = np.concatenate((np.array([-1]),pausesIndex))
                for n in range(len(pausesIndex)-1):
                    start = pausesIndex[n]+1
                    end   = pausesIndex[n+1]
                    print(start, end, pausesIndex)
                    ax0.plot((tracks[i][2][start:end]+(tracks[i][3]-startTime))/60.,tracks[i][1][start:end],color=colors)
            else:
                timeDiff = np.diff(tracks[i][2])
                ax0.plot((tracks[i][2]+(tracks[i][3]-startTime))/60.,tracks[i][1],color=colors)

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax0,xLabel='time (min)',yLabel='speed (cm/s)')

        print(nFigs, nRecs)

        # high-res panels #############################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
        ax10 = plt.subplot(gssub1[0])
        ax11 = plt.subplot(gssub1[1])
        ax10.axhline(y=0,ls='--',c='0.5')
        if nFigs >= 1:
            gssub2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[2], hspace=0.2)
            ax20 = plt.subplot(gssub2[0])
            ax21 = plt.subplot(gssub2[1])
            #ax11 = plt.subplot(gs[2])
            ax20.axhline(y=0,ls='--',c='0.5')
        if nFigs >= 2:
            gssub3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[3], hspace=0.2)
            ax30 = plt.subplot(gssub3[0])
            ax31 = plt.subplot(gssub3[1])
            ax30.axhline(y=0,ls='--',c='0.5')
        if nFigs >= 3:
            gssub4 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], hspace=0.2)
            ax40 = plt.subplot(gssub4[0])
            ax41 = plt.subplot(gssub4[1])
            ax40.axhline(y=0,ls='--',c='0.5')
        if nFigs >= 4:
            gssub5 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[5], hspace=0.2)
            ax50 = plt.subplot(gssub5[0])
            ax51 = plt.subplot(gssub5[1])
            ax50.axhline(y=0,ls='--',c='0.5')
        if nFigs > 5:
            gssub6 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[6], hspace=0.2)
            ax60 = plt.subplot(gssub6[0])
            ax61 = plt.subplot(gssub6[1])
            ax60.axhline(y=0,ls='--',c='0.5')
        for n in range(nRecs):
            if not tracks[n][4]:
                fff = int(n / (plotsPerFig))
                colors = plt.cm.jet(float(n) / float(nRecs - 1))
                print(n, fff, nFigs)
                w = np.concatenate((np.array([tracks[n][2][0]]), np.diff(tracks[n][2])))
                if fff == 0:
                    ax10.plot(tracks[n][2],tracks[n][1],label='%s' % tracks[n][5])
                    ax11.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
                elif fff == 1:
                    ax20.plot(tracks[n][2], tracks[n][1],label='%s' % tracks[n][5])
                    ax21.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
                elif fff == 2:
                    ax30.plot(tracks[n][2], tracks[n][1], label='%s' % tracks[n][5])
                    ax31.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
                elif fff == 3:
                    ax40.plot(tracks[n][2], tracks[n][1], label='%s' % tracks[n][5])
                    ax41.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
                elif fff == 4:
                    ax50.plot(tracks[n][2], tracks[n][1], label='%s' % tracks[n][5])
                    ax51.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
                elif fff == 5:
                    ax60.plot(tracks[n][2], tracks[n][1], label='%s' % tracks[n][5])
                    ax61.hist(tracks[n][1],bins=40,range=[minSpeed,maxSpeed],histtype='step',weights=w)
        # and moves left and bottom axes away
        self.layoutOfPanel(ax10,yLabel='speed (cm/s)',Leg=[1,8])
        self.layoutOfPanel(ax11, xLabel='speed (cm/s)')
        if nFigs > 1:
            self.layoutOfPanel(ax20, yLabel='speed (cm/s)', Leg=[1, 8])
            self.layoutOfPanel(ax21, xLabel='speed (cm/s)')
        if nFigs > 2:
            self.layoutOfPanel(ax30, yLabel='speed (cm/s)', Leg=[1, 8])
            self.layoutOfPanel(ax31, xLabel='speed (cm/s)')
        if nFigs > 3:
            self.layoutOfPanel(ax40, yLabel='speed (cm/s)', Leg=[1, 8])
            self.layoutOfPanel(ax41, xLabel='speed (cm/s)')
        if nFigs > 4:
            self.layoutOfPanel(ax50, yLabel='speed (cm/s)', Leg=[1, 8])
            self.layoutOfPanel(ax51, xLabel='speed (cm/s)')
        if nFigs > 5:
            self.layoutOfPanel(ax60, yLabel='speed (cm/s)')#, Leg=[1, 8])
            self.layoutOfPanel(ax61, xLabel='speed (cm/s)')
        ## summaray ############################################################
        gssubSum = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[nFigs+1], hspace=0.2)

        axS0 = plt.subplot(gssubSum[0])
        axS1 = plt.subplot(gssubSum[1])
        width = 0.35
        ax0.axhline(y=0, ls='--', c='0.5')
        for i in range(nRecs):
            colors = plt.cm.jet(float(i) / float(nRecs - 1))
            if not tracks[i][4]:
                w = np.concatenate((np.array([tracks[i][2][0]]), np.diff(tracks[i][2])))
                restingWalkingMask = tracks[i][1] > 5.
                resting = np.sum(w[np.invert(restingWalkingMask)])
                walking = np.sum(w[restingWalkingMask])
                if sum(w[restingWalkingMask])!=0.:
                    avgSpeed = np.average(tracks[i][1][restingWalkingMask],weights=w[restingWalkingMask])
                else:
                    avgSpeed = 0.
                axS0.bar(i,resting/walking,color=colors)
                #axS0.bar(i+width,walking,width,color=colors,hatch='/')
                #(tracks[i][2] + (tracks[i][3] - startTime)) / 60., tracks[i][1], color=colors)
                axS1.bar(i,avgSpeed, color=colors)

        #axS1.set_ylim(8,13)
        self.layoutOfPanel(axS0,xLabel='recording',yLabel='fraction of time resting/running')#, yLabel='speed (cm/s)', Leg=[1, 8])
        self.layoutOfPanel(axS1,xLabel='recording',yLabel='average max. speed (cm/s)')#, xLabel='speed (cm/s)')
        #plt.xlabel('time (s)')
        #plt.ylabel('speed (cm/s)')

        # change tick spacing
        # majorLocator_x = MultipleLocator(10)
        # ax1.xaxis.set_major_locator(majorLocator_x)

        ## save figure ############################################################
        fname = self.determineFileName(date,'walking_activity')

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        plt.close()

    ##########################################################################################
    def generateOverviewFigure(self,mouse,allDataPerSession,wheelCircumshpere):

        # number of recordings sessions
        nSessions = len(allDataPerSession)

        conversionFactor = wheelCircumshpere/(360.*1000.)

        xScalingFactor = 1.56 # um/px with zoomFactor 1
        yScalingFactor = 1.59 # um/px with zoomFactor 1

        # extract
        # tracksN = []
        # highResRecs = 0
        # maxSpeed = 0.
        # minSpeed = 0.
        # for i in range(nRecs):
        #     if tracks[i][4]:
        #         tracksN.append([i,tracks[i][3],True])
        #     else:
        #         highResRecs+=1
        #         tracksN.append([i,tracks[i][3],False])
        #         if max(tracks[i][1])>maxSpeed:
        #             maxSpeed =max(tracks[i][1])
        #         if min(tracks[i][1])<minSpeed:
        #             minSpeed = min(tracks[i][1])
        # plotsPerFig = 3
        # nFigs = int(highResRecs / (plotsPerFig))
        #
        # # sort according to earliest experiment
        # tracksN.sort(key=lambda x: x[1])
        # #pdb.set_trace()
        # startTime = tracksN[0][1]
        # print('monitor N : ', tracksN)

        # figure #################################
        fig_width = 22  # width in inches
        fig_height = (4*nSessions)+4  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(nSessions, 1 # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.9+(fig_height*0.001), bottom=1.1/fig_height)

        # sub-panel enumerations
        plt.figtext(0.05, 0.975, 'mouse : %s with %s recording sessions' % (mouse, nSessions),clip_on=False,color='red',size=18)


        for nSess in range(len(allDataPerSession)):
            gssub = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[nSess], width_ratios=[1.5,1.5,1,0.8], hspace=0.2)

            # angle progression plot #######################################################
            ax0 = plt.subplot(gssub[0])

            ax0.axhline(y=0, ls='--', c='0.5')
            tracks = allDataPerSession[nSess][1]
            nTracks = len(tracks)
            startTime = tracks[0][3]
            #pdb.set_trace()
            endAngles = []
            highResTrials = 0
            for i in range(nTracks):
                colors = plt.cm.jet(float(i) / float(nTracks - 1))
                if tracks[i][4]:
                    timeDiff = np.diff(tracks[i][5][:,0])
                    pausesIndex = np.where(timeDiff > 30.)[0]
                    # print np.shape(pausesIndex)
                    #pdb.set_trace()
                    pausesIndex = np.concatenate((np.array([-1]), pausesIndex))
                    #pdb.set_trace()
                    for n in range(len(pausesIndex)):
                        start = pausesIndex[n] + 1
                        if n==(len(pausesIndex)-1):
                            end = -1
                        else:
                            end = pausesIndex[n + 1]
                        #print(start, end, pausesIndex)
                        endAngles.append(tracks[i][5][start:end][:,1][-1])
                        ax0.plot((tracks[i][5][start:end][:,0] + (tracks[i][3] - startTime)) / 60., tracks[i][5][start:end][:,1]*conversionFactor, color='0.3')
                else:
                    highResTrials+=1
                    #timeDiff = np.diff(tracks[i][2])
                    ax0.plot((tracks[i][5][:,0] + (tracks[i][3] - startTime)) / 60., (endAngles[i-1]+tracks[i][5][:,1])*conversionFactor, color=colors)

            ax0.set_title('%s. session, %s, %s trials' % ((nSess + 1), allDataPerSession[nSess][0], highResTrials), loc='left', fontweight='bold')
            # removes upper and right axes
            # and moves left and bottom axes away
            self.layoutOfPanel(ax0, xLabel='time (min)', yLabel='distance covered (m)')


            # speed plot #######################################################
            #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
            ax1 = plt.subplot(gssub[1])
            ax1.axhline(y=0,ls='--',c='0.5')
            tracks = allDataPerSession[nSess][1]
            nTracks = len(tracks)
            startTime = tracks[0][3]
            for i in range(nTracks):
                colors = plt.cm.jet(float(i) / float(nTracks - 1))
                if tracks[i][4]:
                    timeDiff = np.diff(tracks[i][2])
                    pausesIndex = np.where(timeDiff>30.)[0]
                    #print np.shape(pausesIndex)
                    #pdb.set_trace()
                    pausesIndex = np.concatenate((np.array([-1]),pausesIndex))
                    for n in range(len(pausesIndex)):
                        start = pausesIndex[n]+1
                        if n==(len(pausesIndex)-1):
                            end = -1
                        else:
                            end = pausesIndex[n + 1]
                        #print(start, end, pausesIndex)
                        ax1.plot((tracks[i][2][start:end]+(tracks[i][3]-startTime))/60.,tracks[i][1][start:end],color='0.3')
                else:
                    #timeDiff = np.diff(tracks[i][2])
                    ax1.plot((tracks[i][2]+(tracks[i][3]-startTime))/60.,tracks[i][1],color=colors)
            #pdb.set_trace()
            VideoTimeStamps = allDataPerSession[nSess][2]
            for i in range(len(VideoTimeStamps)):
                ax1.plot([(VideoTimeStamps[i][3]-startTime)/60.,(VideoTimeStamps[i][3]-startTime+30.)/60],[-10,-10],lw=5,c='C1',label=('Video Rec.' if i==0 else None))
            # show when calcium imaging was performed
            CaImgTimeStamps = allDataPerSession[nSess][3][0][4]
            # pdb.set_trace()
            for i in range(len(CaImgTimeStamps)):
                # pdb.set_trace()
                ax1.plot([(CaImgTimeStamps[i] - startTime) / 60., (CaImgTimeStamps[i] - startTime + 30.) / 60], [-15, -15], lw=5, c='C0', label=('Ca Imaging' if i == 0 else None))

            # removes upper and right axes
            # and moves left and bottom axes away
            self.layoutOfPanel(ax1,xLabel='time (min)',yLabel='speed (cm/s)',Leg=[2,9])


            # animal plot #######################################################
            ax2 = plt.subplot(gssub[2])
            #pdb.set_trace()
            ax2.imshow(np.transpose(allDataPerSession[nSess][2][0][0][0]))

            ax2.set_xlabel('pix')
            ax2.set_ylabel('pix')

            # average Ca imaging plot #######################################################
            ax3 = plt.subplot(gssub[3])
            scaleFactor = allDataPerSession[nSess][3][0][3]
            dimensions = np.shape(allDataPerSession[nSess][3][0][1])
            ax3.imshow(np.log(allDataPerSession[nSess][3][0][1]),extent=(0,dimensions[0]*xScalingFactor/scaleFactor,0,dimensions[1]*yScalingFactor/scaleFactor))
            ax3.set_xlabel(u'μm')
            ax3.set_ylabel(u'μm')
            #self.layoutOfPanel(ax3,yLabel='speed (cm/s)') #,Leg=[1,8])


            if nSess == 0:
                #ax0.set_title('distance covered',size=14)
                ax1.set_title('speed',size=14)
                ax2.set_title('first frame from high-speed camera',size=14)
                ax3.set_title('average of Ca-imaging FOV',size=14)


        ## save figure ############################################################
        date = '2019.05.24'
        fname = self.determineFileName(date,'OverviewFigure')

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        plt.close()


    ##########################################################################################
    def generateROIAndEphysImage(self,data,fileName,stim=False):
        '''
            Overview image of an experiment with:
            - average flurescent image
            - outline of rois
            - temporal evolution of fluorescent signal of rois
            - extracellular recording trace, after high-pass filtering
            - extracted spikes and firing rate evolution (after convolution with Gaussian kernel)
            
        '''
        
        img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        caTime = data['raw_data/caImagingTime'].value
        deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ' , deltaX)
        
        ephysHP = data['analyzed_data/spiking_data/ephys_data_high-pass'].value
        ephysTime = data['raw_data/ephysTime'].value
        
        
        leaveOut = self.config['caEphysParameters']['leaveOut'] # in sec 
        leaveOutN = int(leaveOut/mean(ephysTime[1:]-ephysTime[:-1])+0.5)
        
        ephysMask = arange(len(ephysTime)) > leaveOutN
        
        #pdb.set_trace()
        
        firingRate= data['analyzed_data/spiking_data/firing_rate_evolution'].value
        fRDt = data['analyzed_data/spiking_data/firing_rate_evolution'].attrs['dt']
        firingRateTime = linspace(0,(len(firingRate)-1)*fRDt,len(firingRate))
        
        spikeTimes = data['analyzed_data/spiking_data/spikes'].value
        
        firingRateMask = arange(len(firingRateTime)) > int(leaveOut/fRDt + 0.5)
        
        fr = data['analyzed_data/spiking_data/firing_rate'].value[0]
        CV = data['analyzed_data/spiking_data/CV'].value[0]
        
        dataSet = sima.ImagingDataset.load(self.sima_path)
        rois = dataSet.ROIs['stica_ROIs']
        
        nRois = len(rois)
        
        # Extract the signals.
        dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')
        
        raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        
        roiLabels = []
        for n in range(nRois):
            roiLabels.append(rois[n].label)
        
        # read in the time of the stimuli in case of external stimulation
        if stim:
            stimRinging = 0.0015
            stimuli = data['analyzed_data/stimulation_data/stimulus_times'].value
            dt = data['analyzed_data/stimulation_data/stimulus_times'].attrs['dt']
            startStim = stimuli[0]
            endStim   = stimuli[-1] + stimRinging
        
        # figure #################################
        fig_width = 7 # width in inches
        fig_height = 15  # height in inches
        fig_size =  [fig_width,fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi' : 600,
                  'axes.linewidth' : 1.3,
                  'ytick.major.size' : 4,      # major tick size in points
                  'xtick.major.size' : 4     # major tick size in points
                  #'edgecolor' : None
                  #'xtick.major.size' : 2,
                  #'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        
        # create figure instance
        fig = plt.figure()
        
        
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(5, 1,
                               #width_ratios=[1.2,1]
                               height_ratios=[1,1,1,0.1,1]
                               )
        
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3,hspace=0.4)
        
        # possibly change outer margins of the figure
        #plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)
        
        # sub-panel enumerations
        #plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)
        
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        
        # title
        #ax0.set_title('sub-plot 1')
        
        ax0.imshow(img,cmap=cm.gray,extent=[0,shape(img)[1]*deltaX,0,shape(img)[0]*deltaX])
        #pdb.set_trace()
        for n in range(len(rois)):
            x,y = rois[n].polygons[0].exterior.xy
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(array(x)*deltaX,(shape(img)[0]-array(y))*deltaX,'-',c=colors,zorder=1)
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        ax0.set_xlim(0,shape(img)[1]*deltaX)
        ax0.set_ylim(0,shape(img)[0]*deltaX)
        #ax0.set_xlim()
        #ax0.set_ylim()
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        
        
        # sub-panel 0 #############################################
        ax10 = plt.subplot(gs[1])

        for n in range(len(rois)):
            colors = cm.jet(float(n)/float(nRois-1))
            if rois[n].label == 'cell':
                ax10.plot(caTime,raw_signals[0][n],c=colors,label=str(rois[n].label))
            else:
                ax10.plot(caTime,raw_signals[0][n],c=colors,alpha=0.5,label=str(rois[n].label))
        
        #ax10.plot(caTime,caTrace,c='k',label='rec trace')
        
        # and moves left and bottom axes away
        ax10.spines['top'].set_visible(False)
        ax10.spines['right'].set_visible(False)
        ax10.spines['bottom'].set_position(('outward', 10))
        ax10.spines['left'].set_position(('outward', 10))
        ax10.yaxis.set_ticks_position('left')
        ax10.xaxis.set_ticks_position('bottom')
        ax10.legend(loc=1,frameon=False)
        
        ax10.set_ylabel(r'$F/F_0$')
        
        # sub-panel 1 #############################################
        ax11 = plt.subplot(gs[2])

        ax11.plot(ephysTime[ephysMask],ephysHP[ephysMask])
        # and moves left and bottom axes away
        ax11.spines['top'].set_visible(False)
        ax11.spines['right'].set_visible(False)
        ax11.spines['bottom'].set_position(('outward', 10))
        ax11.spines['left'].set_position(('outward', 10))
        ax11.yaxis.set_ticks_position('left')
        ax11.xaxis.set_ticks_position('bottom')
        #ax11.legend(loc=1,frameon=False)
        #pdb.set_trace()
        if stim:
            stimMask = (ephysTime[ephysMask] < startStim) | (ephysTime[ephysMask] > endStim) # elementwise or 
            maxY = np.max(ephysHP[ephysMask][stimMask])
            minY = np.min(ephysHP[ephysMask][stimMask])
            ax11.set_ylim(1.1*maxY,1.1*minY)
        
        # legends and labels
        #plt.legend(loc=3,frameon=False)
        
        ax11.set_ylabel('current')
        
        # sub-panel 1 #############################################
        ax12 = plt.subplot(gs[3])

        ax12.vlines(spikeTimes,  0, 1,lw=0.2)
        if stim:
            ax12.vlines(stimuli,1,2,lw=0.2,color='red')
        
        ax12.spines['top'].set_visible(False)
        ax12.spines['right'].set_visible(False)
        ax12.spines['bottom'].set_visible(False)
        ax12.spines['left'].set_visible(False)
        ax12.yaxis.set_visible(False)
        ax12.xaxis.set_visible(False)
        ax12.set_xlim(ephysTime[0],ephysTime[-1])
            
        # sub-panel 1 #############################################
        ax11 = plt.subplot(gs[4])

        ax11.plot(firingRateTime[firingRateMask],firingRate[firingRateMask],label='firing rate= '+str(round(fr,2))+', CV='+str(round(CV,3)))
        
        # and moves left and bottom axes away
        ax11.spines['top'].set_visible(False)
        ax11.spines['right'].set_visible(False)
        ax11.spines['bottom'].set_position(('outward', 10))
        ax11.spines['left'].set_position(('outward', 10))
        ax11.yaxis.set_ticks_position('left')
        ax11.xaxis.set_ticks_position('bottom')
        ax11.legend(loc=1,frameon=False)
        
        
        # legends and labels
        #plt.legend(loc=3,frameon=False)
        
        ax11.set_xlabel('time (sec)')
        ax11.set_ylabel('firing rate (spk/sec)')
        # change tick spacing 
        #majorLocator_x = MultipleLocator(10)
        #ax1.xaxis.set_major_locator(majorLocator_x)
        
        # change legend text size 
        #leg = plt.gca().get_legend()
        #ltext  = leg.get_texts()
        #plt.setp(ltext, fontsize=11)
        #pdb.set_trace()
        ## save figure ############################################################
        fname = fileName + '_roi_ephys_traces'
        
        #savefig(fname+'.png')
        savefig(fname+'.pdf')
        #pdb.set_trace()
    ##########################################################################################
    def generateCaSpikesImage(self,data,fileName,stim=False):
        '''
            Connection between calcium signal and spikes
            - average flurescent image
            - outline of rois
            - temporal evolution of fluorescent signal of rois
            - extracellular recording trace, after high-pass filtering
            - extracted spikes 
            - spikes from deconvolution 
            
        '''
        
        img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        caTime = data['raw_data/caImagingTime'].value
        deltaX = (data['raw_data/caImagingField'].value)[2]
        print('deltaX ' , deltaX)
        
        ephysHP = data['analyzed_data/spiking_data/ephys_data_high-pass'].value
        ephysTime = data['raw_data/ephysTime'].value
        
        
        leaveOut = self.config['caEphysParameters']['leaveOut'] # in sec 
        leaveOutN = int(leaveOut/mean(ephysTime[1:]-ephysTime[:-1])+0.5)
        
        ephysMask = arange(len(ephysTime)) > leaveOutN
        
        #pdb.set_trace()
        
        firingRate= data['analyzed_data/spiking_data/firing_rate_evolution'].value
        fRDt = data['analyzed_data/spiking_data/firing_rate_evolution'].attrs['dt']
        firingRateTime = linspace(0,(len(firingRate)-1)*fRDt,len(firingRate))
        
        spikeTimes = data['analyzed_data/spiking_data/spikes'].value
        
        firingRateMask = arange(len(firingRateTime)) > int(leaveOut/fRDt + 0.5)
        
        fr = data['analyzed_data/spiking_data/firing_rate'].value[0]
        CV = data['analyzed_data/spiking_data/CV'].value[0]
        
        dataSet = sima.ImagingDataset.load(self.sima_path)
        rois = dataSet.ROIs['stica_ROIs']
        
        nRois = len(rois)
        
        # Extract the signals.
        dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')
        
        raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        
        roiLabels = []
        for n in range(nRois):
            roiLabels.append(rois[n].label)
        
        # read in the time of the stimuli in case of external stimulation
        if stim:
            stimRinging = 0.0015
            stimuli = data['analyzed_data/stimulation_data/stimulus_times'].value
            dt = data['analyzed_data/stimulation_data/stimulus_times'].attrs['dt']
            startStim = stimuli[0]
            endStim   = stimuli[-1] + stimRinging
        
        # extract data from deconvolution
        n_best = data['analyzed_data/deconvolution/spike_train'].value
        tDeconv = data['analyzed_data/roi_fluorescence_traces/rawTracesTime']
        
        
        # figure #################################
        fig_width = 7 # width in inches
        fig_height = 15  # height in inches
        fig_size =  [fig_width,fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi' : 600,
                  'axes.linewidth' : 1.3,
                  'ytick.major.size' : 4,      # major tick size in points
                  'xtick.major.size' : 4     # major tick size in points
                  #'edgecolor' : None
                  #'xtick.major.size' : 2,
                  #'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        
        # create figure instance
        fig = plt.figure()
        
        
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(5, 1,
                               #width_ratios=[1.2,1]
                               height_ratios=[1,1,1,1,0.1]
                               )
        
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3,hspace=0.4)
        
        # possibly change outer margins of the figure
        #plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)
        
        # sub-panel enumerations
        #plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)
        
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        
        # title
        #ax0.set_title('sub-plot 1')
        
        ax0.imshow(img,cmap=cm.gray,extent=[0,shape(img)[1]*deltaX,0,shape(img)[0]*deltaX])
        #pdb.set_trace()
        for n in range(len(rois)):
            x,y = rois[n].polygons[0].exterior.xy
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(array(x)*deltaX,(shape(img)[0]-array(y))*deltaX,'-',c=colors,zorder=1)
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        ax0.set_xlim(0,shape(img)[1]*deltaX)
        ax0.set_ylim(0,shape(img)[0]*deltaX)
        #ax0.set_xlim()
        #ax0.set_ylim()
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'y ($\mu$m)')
        
        
        # sub-panel 0 #############################################
        ax10 = plt.subplot(gs[1])

        for n in range(len(rois)):
            colors = cm.jet(float(n)/float(nRois-1))
            ax10.plot(caTime,raw_signals[0][n],c=colors,label=str(rois[n].label))
        
        #ax10.plot(caTime,caTrace,c='k',label='rec trace')
        
        # and moves left and bottom axes away
        ax10.spines['top'].set_visible(False)
        ax10.spines['right'].set_visible(False)
        ax10.spines['bottom'].set_position(('outward', 10))
        ax10.spines['left'].set_position(('outward', 10))
        ax10.yaxis.set_ticks_position('left')
        ax10.xaxis.set_ticks_position('bottom')
        ax10.legend(loc=1,frameon=False)
        
        ax10.set_ylabel(r'$\Delta F/F$')
        
        # sub-panel 1 #############################################
        ax11 = plt.subplot(gs[2])

        ax11.plot(ephysTime[ephysMask],ephysHP[ephysMask])
        # and moves left and bottom axes away
        ax11.spines['top'].set_visible(False)
        ax11.spines['right'].set_visible(False)
        ax11.spines['bottom'].set_position(('outward', 10))
        ax11.spines['left'].set_position(('outward', 10))
        ax11.yaxis.set_ticks_position('left')
        ax11.xaxis.set_ticks_position('bottom')
        #ax11.legend(loc=1,frameon=False)
        #pdb.set_trace()
        if stim:
            stimMask = (ephysTime[ephysMask] < startStim) | (ephysTime[ephysMask] > endStim) # elementwise or 
            maxY = np.max(ephysHP[ephysMask][stimMask])
            minY = np.min(ephysHP[ephysMask][stimMask])
            ax11.set_ylim(1.1*maxY,1.1*minY)
        
        # legends and labels
        #plt.legend(loc=3,frameon=False)
        
        ax11.set_ylabel('current')
        
        # sub-panel 1 #############################################
        ax11 = plt.subplot(gs[3])
    
        for n in range(len(rois)):
            colors = cm.jet(float(n)/float(nRois-1))
            if n==1:
                ax11.plot(tDeconv,n_best[n],c=colors,label=str(rois[n].label))
        
        # and moves left and bottom axes away
        ax11.spines['top'].set_visible(False)
        ax11.spines['right'].set_visible(False)
        ax11.spines['bottom'].set_position(('outward', 10))
        ax11.spines['left'].set_position(('outward', 10))
        ax11.yaxis.set_ticks_position('left')
        ax11.xaxis.set_ticks_position('bottom')
        ax11.legend(loc=1,frameon=False)
        
        
        # legends and labels
        #plt.legend(loc=3,frameon=False)
        
        ax11.set_xlabel('time (sec)')
        ax11.set_ylabel(r'$\hat{n}$')
        # change tick spacing 
        #majorLocator_x = MultipleLocator(10)
        #ax1.xaxis.set_major_locator(majorLocator_x)
        
        # change legend text size 
        #leg = plt.gca().get_legend()
        #ltext  = leg.get_texts()
        #plt.setp(ltext, fontsize=11)
        
        # sub-panel 1 #############################################
        ax12 = plt.subplot(gs[4])

        ax12.vlines(spikeTimes,  0, 1,lw=0.2)
        if stim:
            ax12.vlines(stimuli,1,2,lw=0.2,color='red')
        
        ax12.spines['top'].set_visible(False)
        ax12.spines['right'].set_visible(False)
        ax12.spines['bottom'].set_visible(False)
        ax12.spines['left'].set_visible(False)
        ax12.yaxis.set_visible(False)
        ax12.xaxis.set_visible(False)
        ax12.set_xlim(ephysTime[0],ephysTime[-1])
            
        
        
        ## save figure ############################################################
        fname = fileName + '_ca-spikes'
        
        #savefig(fname+'.png')
        savefig(fname+'.pdf')
        
    ##########################################################################################
    def checkConnectionBtwCaAndEphys(self,data,fileName):
        
        leaveOut = self.config['caEphysParameters']['leaveOut'] # in sec 
        
        caTime = data['raw_data/caImagingTime'].value
        
        firingRate= data['analyzed_data/spiking_data/firing_rate_evolution'].value
        fRDt = data['analyzed_data/spiking_data/firing_rate_evolution'].attrs['dt']
        firingRateTime = linspace(0,(len(firingRate)-1)*fRDt,len(firingRate))
        
        spikeTimes = data['analyzed_data/spiking_data/spikes'].value
        
        pdb.set_trace()
        firingRateMask = arange(len(firingRateTime)) > int(leaveOut/fRDt + 0.5)
        
        dataSet = sima.ImagingDataset.load(self.sima_path)
        rois = dataSet.ROIs['stica_ROIs']
        
        nRois = len(rois)
        
        # Extract the signals.
        dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')
        
        raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        
        roiLabels = []
        for n in range(nRois):
            roiLabels.append(rois[n].label)
        
        
        # figure #################################
        fig_width = 7 # width in inches
        fig_height = 15  # height in inches
        fig_size =  [fig_width,fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi' : 600,
                  'axes.linewidth' : 1.3,
                  'ytick.major.size' : 4,      # major tick size in points
                  'xtick.major.size' : 4     # major tick size in points
                  #'edgecolor' : None
                  #'xtick.major.size' : 2,
                  #'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        
        # create figure instance
        fig = plt.figure()
        
        
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(4, 1#,
                               #width_ratios=[1.2,1]
                               #height_ratios=[1,1,1,0.1,1]
                               )
        
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3,hspace=0.4)
        
        # possibly change outer margins of the figure
        #plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)
        
        # sub-panel enumerations
        #plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)
        
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        
        #pdb.set_trace()
        
        for n in range(len(rois)):
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(caTime,(raw_signals[0][n]-min(raw_signals[0][n]))/(max(raw_signals[0][n])-min(raw_signals[0][n])),c=colors,label=str(rois[n].label))
            
        ax0.plot(firingRateTime[firingRateMask],(firingRate[firingRateMask]-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='k')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        #ax0.set_xlim(0,shape(img)[1]*deltaX)
        #ax0.set_ylim(0,shape(img)[0]*deltaX)
        #ax0.set_xlim()
        #ax0.set_ylim()
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        #plt.xlabel(r'x ($\mu$m)')
        plt.ylabel(r'normalized')
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[1])
        
        fInterpol = scipy.interpolate.interp1d(firingRateTime,firingRate)
        #rrr = scipy.signal.resample(firingRate,len(raw_signals[0][n]),t=firingRateTime)
        firingRateInt = fInterpol(caTime)
        
        for n in range(len(rois)):
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(caTime,(raw_signals[0][n]-min(raw_signals[0][n]))/(max(raw_signals[0][n])-min(raw_signals[0][n])),c=colors,label=str(rois[n].label))
            
        ax0.plot(firingRateTime[firingRateMask],(firingRate[firingRateMask]-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='k')
        ax0.plot(caTime,(firingRateInt-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='magenta')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        ax0.set_xlim(22,26)
        ax0.set_ylim(0,1)
        #ax0.set_ylim(0,shape(img)[0]*deltaX)
        #ax0.set_xlim()
        #ax0.set_ylim()
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        plt.xlabel(r'time (sec)')
        plt.ylabel(r'normalized')
        
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[2])
        
        deltaTCa = mean(caTime[1:]-caTime[:-1])
        
        ax0.axhline(y=0,c='0.6',ls='--')
        ax0.axvline(x=0,c='0.6',ls='--')
        
        for n in range(len(rois)):
            #
            pear = scipy.stats.pearsonr(raw_signals[0][n], firingRateInt)
            #
            cc = self.analysisTools.crosscorr(deltaTCa,firingRateInt,raw_signals[0][n],correlationRange=3)
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(cc[:,0],cc[:,1],c=colors,label='pearson = '+str(round(pear[0],4)))
        
        #cc = self.analysisTools.crosscorr(deltaTCa,raw_signals[0][1],raw_signals[0][1],correlationRange=5)
        #ax0.plot(cc[:,0],cc[:,1],c='k')
        #ax0.plot(firingRateTime[firingRateMask],(firingRate[firingRateMask]-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='k')
        #ax0.plot(caTime,(firingRateInt-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='magenta')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        ax0.legend(frameon=False)
        
        ax0.set_xlim(-3,3)
        
        plt.xlabel(r'time lag (sec)')
        plt.ylabel(r'correlation')
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[3])
        
        
        
        ax0.axvline(x=0,c='0.5',ls='--')
        for n in range(len(rois)):
            ##
            ##
            #sta = self.analysisTools.spikeTriggeredAverage(fRDt,spikeTimes,caTime,raw_signals[0][n],0.2,0.5)
            [pta,pauseDuration] = self.analysisTools.pauseTriggeredAverage(fRDt,spikeTimes,caTime,raw_signals[0][n])
            colors = cm.jet(float(n)/float(nRois-1))
            ax0.plot(pta[:,0],pta[:,1],c=colors)
            #ax0.plot(sta[:,0],sta[:,1],c=colors,ls='--')
        
        ax0.axvline(x=pauseDuration,c='0.5',ls='--')
        
        #cc = self.analysisTools.crosscorr(deltaTCa,raw_signals[0][1],raw_signals[0][1],correlationRange=5)
        #ax0.plot(cc[:,0],cc[:,1],c='k')
        #ax0.plot(firingRateTime[firingRateMask],(firingRate[firingRateMask]-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='k')
        #ax0.plot(caTime,(firingRateInt-min(firingRate[firingRateMask]))/(max(firingRate[firingRateMask])-min(firingRate[firingRateMask])),c='magenta')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        ax0.legend(frameon=False)
        
        #ax0.set_xlim(-3,3)
        
        plt.xlabel(r'time lag (sec)')
        plt.ylabel(r'$\Delta F/F$')
        
        ## save figure ############################################################
        fname = fileName + '_ca_ephys'
        
        #savefig(fname+'.png')
        savefig(fname+'.pdf')
        
    ##########################################################################################
    def generateProjectionInTime(self,rawData,fileName):
        '''
            Visualization of beam stimulation :
            - image is projected onto the y-axis and the temperal evolution of the projected 1d image is shown 
            - this visualization allows to examine spatial spread (along the rostal-caudal axis) and time course the parallel fiber stimulation induced fluorescent signal
        '''
        images =  rawData['raw_data/caImaging'].value #rawData['analyzed_data/motion_corrected_images/motionCorrectedStack'].value
        imgTimes = rawData['raw_data/caImagingTime'].value
        
        deltaT = average(imgTimes[1:]-imgTimes[:-1])
        
        deltaX = (rawData['raw_data/caImagingField'].value)[2]
        
        print('delta T , X: ', deltaT, deltaX)
        #pdb.set_trace()
        imgYProj = average(images,axis=2)
        
        baseLine = average(imgYProj[:int(self.config['projectionInTimeParameters']['baseLinePeriod']/deltaT+0.5),:],axis=0)
        
        imgYProjNorm = (imgYProj-baseLine)/baseLine
        
        horizontalCuts = self.config['projectionInTimeParameters']['horizontalCuts']
        verticalCut = self.config['projectionInTimeParameters']['verticalCut']
        verticalCutN = int(verticalCut/deltaX+0.5)
        
        stimStart = self.config['projectionInTimeParameters']['stimStart'] #5
        stimLength = self.config['projectionInTimeParameters']['stimLength'] #0.2
        fitStart = self.config['projectionInTimeParameters']['fitStart'] # 5
        
        # double exponential fit-function
        fitfunc = lambda p, x: p[0]*(exp(-x/p[1]) - exp(-x/p[2]))
        errfunc = lambda p, x, y: fitfunc(p,x)-y
        
        p0 = array([10.,0.3,0.1])
        # fit a gaussian to the correlation function
        y = imgYProjNorm[int(fitStart/deltaT):,verticalCutN]
        x = arange(len(y))*deltaT
        p1, success = scipy.optimize.leastsq(errfunc, p0.copy(),args=(x,y))
        print('fit parameter : ', p1)
        yFit = fitfunc(p1, x)
        
        ################################################################
        # set plot attributes
        
        fig_width = 10 # width in inches
        fig_height = 8  # height in inches
        fig_size =  [fig_width,fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi' : 600,
                  'axes.linewidth' : 1.3,
                  'ytick.major.size' : 4,      # major tick size in points
                  'xtick.major.size' : 4      # major tick size in points
                  #'edgecolor' : None
                  #'xtick.major.size' : 2,
                  #'ytick.major.size' : 2,
                  }
        rcParams.update(params)
        
        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'
        
        # create figure instance
        fig = plt.figure()
        
        
        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 2#,
                               #width_ratios=[1,1.2],
                               #height_ratios=[1,1]
                               )
        
        # define vertical and horizontal spacing between panels
        gs.update(hspace=0.3,wspace=0.4)
        
        # possibly change outer margins of the figure
        #plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)
        
        # sub-panel enumerations
        #plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.92, 'B',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.06, 0.47, 'C',clip_on=False,color='black', weight='bold',size=22)
        #plt.figtext(0.47, 0.47, 'D',clip_on=False,color='black', weight='bold',size=22)
        
        
        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        
        # title
        #ax0.set_title('sub-plot 1')
        
        # diplay of data
        ii = ax0.imshow(imgYProjNorm,aspect=self.config['projectionInTimeParameters']['threeDAspectRatio'],extent=[0,shape(imgYProjNorm)[1]*deltaX,0,shape(imgYProjNorm)[0]*deltaT],origin='lower')
        cb = colorbar(ii)
        ax0.plot([0,0],[stimStart,stimStart+stimLength],'-',c='magenta',lw=20,solid_capstyle='butt')
        for i in range(len(horizontalCuts)):
            ax0.axhline(y=horizontalCuts[i],c='white',ls='--')
        ax0.axvline(x=verticalCut,ls='--',c='white')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')
        ax0.xaxis.set_ticks_position('bottom')
        
        ax0.set_xlim(0,shape(imgYProjNorm)[1]*deltaX)
        ax0.set_ylim(0,shape(imgYProjNorm)[0]*deltaT)
        # legends and labels
        #plt.legend(loc=1,frameon=False)
        
        plt.xlabel(r'location ($\mu$m)')
        plt.ylabel('time (sec)')
        
        
        # third sub-plot #######################################################
        ax1 = plt.subplot(gs[2])
        
        
        #pdb.set_trace()
        # diplay of data
        for i in range(len(horizontalCuts)):
            fwhm = self.analysisTools.calcFWHM(arange(shape(imgYProjNorm)[1])*deltaX,imgYProjNorm[int(horizontalCuts[i]/deltaT)])
            ax1.plot(arange(shape(imgYProjNorm)[1])*deltaX,imgYProjNorm[int(horizontalCuts[i]/deltaT)],label=str(horizontalCuts[i])+' sec, '+str(fwhm[0])+r'$\mu$m')
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['bottom'].set_position(('outward', 10))
        ax1.spines['left'].set_position(('outward', 10))
        ax1.yaxis.set_ticks_position('left')
        ax1.xaxis.set_ticks_position('bottom')
        
        ax1.set_xlim(0,shape(imgYProjNorm)[1]*deltaX)
        # legends and labels
        plt.legend(loc=(0.8,0.5),frameon=False)
        
        plt.xlabel(r'distance ($\mu$m)')
        plt.ylabel(r'$\Delta$F/F')
        
        # change tick spacing 
        #majorLocator_x = MultipleLocator(10)
        #ax1.xaxis.set_major_locator(majorLocator_x)
        leg = plt.gca().get_legend()
        ltext  = leg.get_texts()
        plt.setp(ltext, fontsize=11)
        
        # third sub-plot #######################################################
        ax2 = plt.subplot(gs[1])
        
        ax2.plot(arange(shape(imgYProjNorm)[0])*deltaT,imgYProjNorm[:,verticalCutN],lw=2)
        ax2.plot(x+fitStart,yFit,lw=2,label=r'$\tau_{\rm rise} = %8.2f$ sec, $\tau_{\rm decay} = %8.2f$ sec' % (p1[2],p1[1]))
        
        ax2.plot([stimStart,stimStart+1],[self.config['projectionInTimeParameters']['stimulationBarLocation'],self.config['projectionInTimeParameters']['stimulationBarLocation']],'-',c='red',lw=20,solid_capstyle='butt')
        #ax2.annotate('', xy=(stimStart, 1), xytext=(stimStart, 3), annotation_clip=False,
        #    arrowprops=dict(arrowstyle="->",color='red',lw=2),
        #    )
        
        # removes upper and right axes 
        # and moves left and bottom axes away
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_position(('outward', 10))
        ax2.spines['left'].set_position(('outward', 10))
        ax2.yaxis.set_ticks_position('left')
        ax2.xaxis.set_ticks_position('bottom')
        
        
        # legends and labels
        plt.legend(loc=1,frameon=False)
        
        plt.xlabel('time (sec)')
        plt.ylabel(r'$\Delta$F/F')
        
        # change tick spacing 
        #majorLocator_x = MultipleLocator(10)
        #ax2.xaxis.set_major_locator(majorLocator_x)
        leg = plt.gca().get_legend()
        ltext  = leg.get_texts()
        plt.setp(ltext, fontsize=11)
        
        
        ## save figure ############################################################
        fname = fileName + '_temporal_profile_stimulation'
        
        #pdb.set_trace()
        
        savefig(fname+'.png')
        savefig(fname+'.pdf')
        
        
    ##########################################################################################
    def generateMotionArtefactImage(self, date, motion) :
        #rec, img, ttime, rois, raw_signals, imageMetaInfo, motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor):
        '''
            test
        '''
        # img = data['analyzed_data/motionCorrectedImages/timeAverage'].value
        # ttime = data['raw_data/caImagingTime'].value

        # dataSet = sima.ImagingDataset.load(self.sima_path)
        # rois = dataSet.ROIs['stica_ROIs']

        #nRois = len(rois)
        nImage = 0

        nRecordings = len(motion)

        #nFigs = int(nRois / (plotsPerFig + 1) + 1.)
        # Extract the signals.
        # dataSet.extract(rois,signal_channel='GCaMP6F', label='GCaMP6F_signals')

        # raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        deltaX = motion[nImage][3][2]*1.E6
        # deltaX = (data['raw_data/caImagingField'].value)[2]
        #print 'deltaX ', deltaX

        # figure #################################
        fig_width = 16  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(3, 3  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        # plt.subplots_adjust(left=0.14, right=0.92, top=0.92, bottom=0.18)

        # sub-panel enumerations
        plt.figtext(0.06, 0.92, '%s' % date ,clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gs[0])

        # title
        # ax0.set_title('sub-plot 1')
        img = motion[nImage][1]
        ax0.imshow(np.rot90(img,k=2),origin='lower',cmap=plt.cm.gray, extent=[0, np.shape(img)[1] * deltaX, 0, np.shape(img)[0] * deltaX])
        # pdb.set_trace()
        #for n in range(len(rois)):
        #    x, y = rois[n].polygons[0].exterior.xy
        #    colors = plt.cm.jet(float(n) / float(nRois - 1))
        #    ax0.plot( np.array(y) * deltaX, np.array(x) * deltaX, '-', c=colors, zorder=1)

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax0,xLabel=r'x ($\mu$m)',yLabel=r'y ($\mu$m)')

        #ax0.set_xlim(0, np.shape(img)[1] * deltaX)
        #ax0.set_ylim(0, np.shape(img)[0] * deltaX)
        # ax0.set_xlim()
        # ax0.set_ylim()
        # legends and labels
        # plt.legend(loc=1,frameon=False)

        # third sub-plot #######################################################

        # sub-panel 1 #############################################
        ax20 = plt.subplot(gs[3])
        ax21 = plt.subplot(gs[4])
        ax22= plt.subplot(gs[5])

        ax30 = plt.subplot(gs[6])
        ax31 = plt.subplot(gs[7])
        ax32 = plt.subplot(gs[8])


        #ax01.plot(ttime, motionCoordinates[:, 2] * deltaX, label='y')

        # sub-panel 1 #############################################

        for n in range(nRecordings):
            colors = plt.cm.jet(float(n) / float(nRecordings - 1))
            #print n, fff, nFigs
            if n<6:
                ax20.plot(motion[n][2][:,1]*deltaX, motion[n][2][:,2]*deltaX, c=colors, label=str(motion[n][0]))
                ax21.plot(motion[n][4], motion[n][2][:,1]*deltaX, c=colors)
                ax21.plot(motion[n][4], motion[n][2][:,2]*deltaX,ls='--',c=colors)
                ax22.plot(motion[n][4], np.sqrt((motion[n][2][:, 1] * deltaX)**2 + (motion[n][2][:, 2] * deltaX)**2), c=colors)
            else:
                ax30.plot(motion[n][2][:,1]*deltaX, motion[n][2][:,2]*deltaX, c=colors, label=str(motion[n][0]))
                ax31.plot(motion[n][4], motion[n][2][:, 1] * deltaX, c=colors)
                ax31.plot(motion[n][4], motion[n][2][:, 2] * deltaX, ls='--', c=colors)
                ax32.plot(motion[n][4],np.sqrt((motion[n][2][:, 1] * deltaX) ** 2 + (motion[n][2][:, 2] * deltaX) ** 2), c=colors)

        self.layoutOfPanel(ax20, xLabel='x movement ($\mu$m)', yLabel='y movement ($\mu$m)', Leg=[1, 10])
        self.layoutOfPanel(ax21, xLabel='time (s)', yLabel='movement ($\mu$m)')
        self.layoutOfPanel(ax22, xLabel='time (s)', yLabel='movement RMS ($\mu$m)')
        self.layoutOfPanel(ax30,xLabel='x movement ($\mu$m)',yLabel = 'y movement ($\mu$m)', Leg=[1,10])
        self.layoutOfPanel(ax31, xLabel='time (s)', yLabel='movement ($\mu$m)')
        self.layoutOfPanel(ax32, xLabel='time (s)', yLabel='movement RMS ($\mu$m)')
        # change tick spacing
        # majorLocator_x = MultipleLocator(10)
        # ax1.xaxis.set_major_locator(majorLocator_x)

        ## save figure ############################################################
        fname = self.determineFileName(date, 'motion-corretion')

        plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
    ##########################################################################################
    def generateRungMotionPlot(self,rungMotion):

        nRecordings = len(rungMotion)


        date = rungMotion[0][1]
        recA  = rungMotion[0][2]
        recB  = rungMotion[-1][2]


        # pdb.set_trace()
        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(nRecordings, 1  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.14, right=0.92, top=0.98, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.99, '%s   %s   %s-%s' % (self.mouse, date, recA,recB[-3:]), clip_on=False, color='black', size=14)

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        colorCycle = 10
        for i in range(nRecordings):
            ax0 = plt.subplot(gs[i])
            print('rec #%s' % i)
            for n in range(len(rungMotion[i][3])):
                #pdb.set_trace()
                nRungs = len(rungMotion[i][3][n][3][:,0])
                colors = cm.rainbow((rungMotion[i][3][n][2]%colorCycle)/colorCycle)
                ax0.scatter(np.repeat(rungMotion[i][3][n][0],nRungs),rungMotion[i][3][n][3][:,0],c=colors,s=0.3)

            if i ==(nRecordings-1):
                self.layoutOfPanel(ax0, xLabel=r'frame number', yLabel=r'x location (pixel)')
            else:
                self.layoutOfPanel(ax0, xLabel=None, yLabel=r'x location (pixel)')

        ## save figure ############################################################
        fname = self.determineFileName(date, 'rung-motion', reco=recA[:-4])
        # plt.show()
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def generatePawMovementFigure(self,date,rec, fp, hp,rungs,fTimes,centerR,radius,rungsNumbered,fpLinear,hpLinear,linearSpeed,sTimes,frontpawRungDist,hindpawRungDist,startStopFPStep,startStopHPStep):
        yMax = 816
        xMax = 616

        # ax6.plot(hpLinear[:,1][1:],(np.diff(fpLinear[:,5])/np.diff(fpLinear[:,1]))/5.)
        # #ax6.plot(hpLinear[:,1][1:],np.diff(hpLinear[:,5])/np.diff(hpLinear[:,1]))
        #
        # ax6.plot(sTimes, linearSpeed)

        wheelSpeedInterp = interp1d(hpLinear[:,1][1:],(np.diff(fpLinear[:,5])/np.diff(fpLinear[:,1])))

        mask = (sTimes>hpLinear[:,1][1:][0]) & (sTimes<hpLinear[:,1][1:][-1])
        newWheelSpeed = wheelSpeedInterp(sTimes[mask])

        frac = linearSpeed/newWheelSpeed

        pixToCmScaling = 1./np.mean(frac)
        print(pixToCmScaling)

        #pdb.set_trace()
        # figure #################################
        fig_width = 16  # width in inches
        fig_height = 45 # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14,
                  'axes.titlesize': 13,
                  'font.size': 11,
                  'xtick.labelsize': 11,
                  'ytick.labelsize': 11,
                  'figure.figsize': fig_size,
                  'savefig.dpi': 600,
                  'axes.linewidth': 1.3,
                  'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(10, 2  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.14, right=0.92, top=0.98, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.99, '%s   %s   %s' % (self.mouse,date,rec) ,clip_on=False,color='black',size=14)

        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gs[0])

        ax0.plot(fp[:,2],yMax-fp[:,3])
        ax0.plot(fp[:,2],yMax-fp[:,3],'.')

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax0,xLabel=r'x (pixel)',yLabel=r'y (pixel)')

        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax1 = plt.subplot(gs[1])

        ax1.plot(hp[:,2],yMax-hp[:,3])
        ax1.plot(hp[:,2],yMax-hp[:,3],'.')

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax1,xLabel=r'x (pixel)',yLabel=r'y (pixel)')


        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax2 = plt.subplot(gs[2])

        ax2.plot(fpLinear[:,1], fpLinear[:,2],label='fp')
        ax2.plot(fpLinear[:,1], hpLinear[:,2],label='hp')
        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax2, xLabel=r'time (s)', yLabel=r'x forward position (cm)',Leg=[2,9])
        #ax2.set_xlim(3,8)
        #ax2.set_ylim(8,40)

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax3 = plt.subplot(gs[3])

        ax3.plot(fpLinear[:, 1], fpLinear[:, 3],label='fp')
        ax3.plot(hpLinear[:, 1], hpLinear[:, 3],label='hp')

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax3, xLabel=r'time (s)', yLabel=r'y upward position (pixel)',Leg=[2,9])


        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax4 = plt.subplot(gs[4])
        ax4.set_title('frontpaw swings')
        for i in range(len(startStopFPStep)):
            ax4.plot(fp[int(startStopFPStep[i,0]):int(startStopFPStep[i,1]),2]-fp[int(startStopFPStep[i,0]),2],(yMax-fp[int(startStopFPStep[i,0]):int(startStopFPStep[i,1]),3])-(yMax-fp[int(startStopFPStep[i,0]),3]),c='0.5',alpha=0.5)

            #ax0.plot(fp[:,2],yMax-fp[:,3],'.')


        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax4, xLabel=r'x (pixel)', yLabel=r'y (pixel)')

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax5 = plt.subplot(gs[5])
        ax5.set_title('hindpaw swings')
        for i in range(len(startStopHPStep)):
            ax5.plot(hp[int(startStopHPStep[i,0]):int(startStopHPStep[i,1]),2]-hp[int(startStopHPStep[i,0]),2],(yMax-hp[int(startStopHPStep[i,0]):int(startStopHPStep[i,1]),3])-(yMax-hp[int(startStopHPStep[i,0]),3]),c='0.5',alpha=0.5)
            #ax0.plot(fp[:,2],yMax-fp[:,3],'.')


        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax5, xLabel=r'x (pixel)', yLabel=r'y (pixel)')


        print(len(fp), len(fpLinear), len(hp), len(hpLinear))
        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax6 = plt.subplot(gs[6])
        ax8 = plt.subplot(gs[8])
        ax6.set_title('frontpaw linearized swings')
        ax8.set_title('frontpaw linearized swings speed')
        for i in range(len(startStopFPStep)):
            mask = (fpLinear[:,1]>=startStopFPStep[i,2]) & (fpLinear[:,1]<=startStopFPStep[i,3])
            startIdx = np.where(mask==True)[0][0]
            ax6.plot(fpLinear[mask, 2] - fpLinear[startIdx, 2],fpLinear[mask, 3] - fpLinear[startIdx, 3], c='0.5', alpha=0.5)  # ax0.plot(fp[:,2],yMax-fp[:,3],'.')
            #
            speed = (np.sqrt((np.diff(fpLinear[mask, 2])/np.diff(fpLinear[:,1][mask]))**2 + (np.diff(fpLinear[mask, 3])/np.diff(fpLinear[:,1][mask]))**2))
            ax8.plot(fpLinear[:,1][mask][1:]-fpLinear[:,1][mask][1],speed, c='0.5', alpha=0.5)

        ax8.set_ylim(ymax=100.)
        self.layoutOfPanel(ax6, xLabel=r'x (cm)', yLabel=r'y (cm)')
        self.layoutOfPanel(ax8, xLabel=r'time (s)', yLabel=r'speed (cm/s)')

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax7 = plt.subplot(gs[7])
        ax9 = plt.subplot(gs[9])
        ax7.set_title('hindpaw linearized swings')
        ax9.set_title('hindpaw linearized swings speed')
        trajX2 = []
        trajY2 = []
        trajX3 = []
        trajY3 = []
        for i in range(len(startStopHPStep)):
            mask = (hpLinear[:,1]>=startStopHPStep[i,2]) & (hpLinear[:,1]<=startStopHPStep[i,3])
            startIdx = np.where(mask==True)[0][0]
            endIdx   = np.where(mask==True)[0][-1]
            ax7.plot(hpLinear[mask,2] - hpLinear[startIdx,2],hpLinear[mask,3] - hpLinear[startIdx, 3], c='0.5', alpha=0.5)  # ax0.plot(fp[:,2],yMax-fp[:,3],'.')
            #
            if (hpLinear[endIdx,2] - hpLinear[startIdx,2])> 1. and (hpLinear[endIdx,2] - hpLinear[startIdx,2])<3. and np.abs(hpLinear[endIdx,3] - hpLinear[startIdx, 3])<0.4:
                ttNorm = (hpLinear[:,1][mask]-hpLinear[:,1][mask][0])/(hpLinear[:,1][mask][-1] - hpLinear[:,1][mask][0])
                track_interpx = interp1d(ttNorm, hpLinear[mask,2] - hpLinear[startIdx,2])
                track_interpy = interp1d(ttNorm, hpLinear[mask,3] - hpLinear[startIdx,3])
                trajX2.append(track_interpx(np.linspace(0,1,51)))
                trajY2.append(track_interpy(np.linspace(0,1,51)))
            elif (hpLinear[endIdx,2] - hpLinear[startIdx,2])> 3.:
                ttNorm = (hpLinear[:,1][mask]-hpLinear[:,1][mask][0])/(hpLinear[:,1][mask][-1] - hpLinear[:,1][mask][0])
                track_interpx = interp1d(ttNorm, hpLinear[mask,2] - hpLinear[startIdx,2])
                track_interpy = interp1d(ttNorm, hpLinear[mask,3] - hpLinear[startIdx,3])
                trajX3.append(track_interpx(np.linspace(0,1,51)))
                trajY3.append(track_interpy(np.linspace(0,1,51)))


            speed = (np.sqrt((np.diff(hpLinear[mask, 2])/np.diff(hpLinear[:,1][mask]))**2 + (np.diff(hpLinear[mask, 3])/np.diff(hpLinear[:,1][mask]))**2))
            ax9.plot(hpLinear[:,1][mask][1:]-hpLinear[:,1][mask][1],speed, c='0.5', alpha=0.5)

        trajX2 = np.asarray(trajX2)
        trajY2 = np.asarray(trajY2)
        trajX3 = np.asarray(trajX3)
        trajY3 = np.asarray(trajY3)
        every = 4
        ax7.plot(np.mean(trajX2,axis=0),np.mean(trajY2,axis=0),c='C0',lw=2)
        #ax7.errorbar(np.mean(trajX2,axis=0)[::every],np.mean(trajY2,axis=0)[::every],xerr=np.std(trajX2,axis=0)[::every], yerr=np.std(trajY2,axis=0)[::every],c='C0')
        ax7.plot(np.mean(trajX3,axis=0),np.mean(trajY3,axis=0),c='C1',lw=2)
        #ax7.errorbar(np.mean(trajX3,axis=0)[::every],np.mean(trajY3,axis=0)[::every],xerr=np.std(trajX3,axis=0)[::every], yerr=np.std(trajY3,axis=0)[::every],c='C1')

        # removes upper and right axes
        # and moves left and bottom axes away
        ax9.set_ylim(ymax=100.)
        self.layoutOfPanel(ax7, xLabel=r'x (cm)', yLabel=r'y (cm)')
        self.layoutOfPanel(ax9, xLabel=r'time', yLabel=r'speed (cm/s)')


        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)

        ax6 = plt.subplot(gs[10])
        ax6.set_title('wheel speed')
        ax6.plot(hpLinear[:,1][1:],(np.diff(fpLinear[:,5])/np.diff(fpLinear[:,1]))/pixToCmScaling,label='from rung screws')
        #ax6.plot(hpLinear[:,1][1:],np.diff(hpLinear[:,5])/np.diff(hpLinear[:,1]))

        ax6.plot(sTimes, linearSpeed,label='rotary encoder')
        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax6,xLabel=r'time (s)',yLabel=r'wheel speed (cm/s)',Leg=[1,9])

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)

        ax7 = plt.subplot(gs[11])
        ax6.set_title('wheel and paw speed')
        fpTimes = fTimes[np.array(fp[:,1],dtype=int)]
        hpTimes = fTimes[np.array(hp[:,1],dtype=int)]
        ax7.plot(fpTimes[1:], (np.sqrt((np.diff(fp[:, 2]) / np.diff(fpTimes)) ** 2 + (np.diff(fp[:, 3]) / np.diff(fpTimes)) ** 2)) / 80., label='frontpaw')
        ax7.plot(hpTimes[1:], (np.sqrt((np.diff(hp[:,2])/np.diff(hpTimes))**2 + (np.diff(hp[:,3])/np.diff(hpTimes))**2))/80.,label='hindpaw')

        # ax6.plot(hpLinear[:,1][1:],np.diff(hpLinear[:,5])/np.diff(hpLinear[:,1]))

        ax7.plot(sTimes, np.abs(linearSpeed),label='wheel')
        # removes upper and right axes
        # and moves left and bottom axes away

        ax7.set_ylim(ymax=100)
        self.layoutOfPanel(ax7, xLabel=r'time (s)', yLabel=r'speed (cm/s)',Leg=[1,9])

        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax8 = plt.subplot(gs[12])
        #pdb.set_trace()
        ax8.set_title('frontpaw-rungs distance')
        #for i in range(len(frontpawRungDist)):
        #pdb.set_trace()
        #print np.repeat(fTimes[int(frontpawRungDist[i][1])],len(frontpawRungDist[i][2:])), frontpawRungDist[i][2:]
        ax8.plot(fTimes[np.array(frontpawRungDist[:,1],dtype=int)],frontpawRungDist[:,2:9],'.',c='C0')
        #ax8.plot(fTimes[np.array(frontpawRungDist[:,1],dtype=int)],frontpawRungDist[:,3],'.',c='C1')
        #ax8.plot(fTimes[np.array(frontpawRungDist[:,1],dtype=int)],frontpawRungDist[:,4],'.',c='C2')

        # and moves left and bottom axes away
        self.layoutOfPanel(ax8,xLabel=r'time (s)',yLabel=r'frontpaw-rung dist (pixel)')

        # first sub-plot #######################################################
        #gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax9 = plt.subplot(gs[13])
        ax9.set_title('hindpaw-rungs distance')
        ax9.plot(fTimes[np.array(hindpawRungDist[:,1],dtype=int)],hindpawRungDist[:,2:9],'.',c='C0')
        #ax9.plot(fTimes[np.array(hindpawRungDist[:,1],dtype=int)],hindpawRungDist[:,3],'.',c='C1')
        #ax9.plot(fTimes[np.array(hindpawRungDist[:,1],dtype=int)],hindpawRungDist[:,4],'.',c='C2')


        # and moves left and bottom axes away
        self.layoutOfPanel(ax9,xLabel=r'time (s)',yLabel=r'hindpaw-rung dist (pixel)')


        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax14 = plt.subplot(gs[14])
        ax16 = plt.subplot(gs[16])
        ax14.set_title('minimal frontpaw-rung distance during stance')
        ax16.set_title('rung number during frontpaw stance')
        fpTimes = fTimes[np.array(frontpawRungDist[:,1], dtype=int)]
        #minfpRDist = np.min(np.abs(frontpawRungDist[:,2:9]),axis=1)
        minfpRungN = np.argsort(np.abs(frontpawRungDist[:,2:9]),axis=1)
        fpRungsCrossed =[]
        for i in range(len(startStopFPStep)+1):
            if i == len(startStopFPStep):
                mask = (fpTimes >= startStopFPStep[i-1,3]) & (fpTimes <= fpTimes[-1])
            elif i == 0:
                mask = (fpTimes >= fpTimes[0]) & (fpTimes <= startStopFPStep[i,2])
            else:
                mask = (fpTimes >= startStopFPStep[i-1,3]) & (fpTimes <= startStopFPStep[i,2])
            #mask = (fpTimes >= startStopFPStep[i-1,3]) & (fpTimes <= startStopFPStep[i,2])
            #startIdx = np.where(mask == True)[0][0]
            if sum(mask)> 1 :
                llength = len(frontpawRungDist[mask])
                # generate list of tuples which contain the indices of the first three closest rungs
                # pdb.set_trace()
                idx0 = [(x, y) for x, y in zip(range(llength), minfpRungN[mask][:, 0])]
                idx1 = [(x, y) for x, y in zip(range(llength), minfpRungN[mask][:, 1])]
                idx2 = [(x, y) for x, y in zip(range(llength), minfpRungN[mask][:, 2])]
                # hindpawRungDist[mask][:,9:16][tuple(np.array(idx0).T)]
                ax14.plot(fpTimes[mask], frontpawRungDist[mask][:, 2:9][tuple(np.array(idx0).T)], '.', c='0.5', alpha=0.5)
                ax14.plot(fpTimes[mask], frontpawRungDist[mask][:, 2:9][tuple(np.array(idx1).T)], '.', c='0.7', alpha=0.5)
                ax14.plot(fpTimes[mask], frontpawRungDist[mask][:, 2:9][tuple(np.array(idx2).T)], '.', c='0.9', alpha=0.5)
                # draw a line of the median closest distance during step
                ax14.plot(fpTimes[mask],np.repeat(np.median(frontpawRungDist[mask][:, 2:9][tuple(np.array(idx0).T)]),llength),lw=2)

                ax16.plot(fpTimes[mask], frontpawRungDist[mask][:, 9:16][tuple(np.array(idx0).T)])
                fpRungsCrossed.append([np.where((i - 1) > -1, startStopFPStep[i - 1, 3], fpTimes[0]), numpy.bincount(np.array(frontpawRungDist[mask][:, 9:16][tuple(np.array(idx0).T)], dtype=int)).argmax()])
                #pdb.set_trace()
                #ax14.plot(fpTimes[mask],np.repeat(np.median(minfpRDist[mask]),len(fpTimes[mask])),c='C0',lw=2)
            #
            #pdb.set_trace()
            #ax16.plot(fpTimes[mask],frontpawRungDist[mask][:,5])
            #pdb.set_trace()
            #if sum(mask)> 1 :
            #    fpRungsCrossed.append([np.where((i-1)>-1,startStopFPStep[i-1,3],fpTimes[0]),numpy.bincount(np.array(frontpawRungDist[mask][:,5],dtype=int)).argmax()])
        #ax14.set_ylim(-150,150)
        self.layoutOfPanel(ax14, xLabel=r'time (s)', yLabel=r'min distance (pixel)')
        self.layoutOfPanel(ax16, xLabel=r'time (s)', yLabel=r'rung number')
        #ax10.plot(fTimes[np.array(frontpawRungDist[:, 1], dtype=int)], frontpawRungDist[:, 4], '.', c='C2')

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax15 = plt.subplot(gs[15])
        ax17 = plt.subplot(gs[17])
        ax15.set_title('minimal hindpaw-rung distance during stance')
        ax17.set_title('rung number during hindpaw stance')
        hpTimes = fTimes[np.array(hindpawRungDist[:, 1], dtype=int)]
        # minhpRDist = np.min(np.abs(hindpawRungDist[:,2:5]),axis=1)
        minhpRungN = np.argsort(np.abs(hindpawRungDist[:,2:9]),axis=1)
        hpRungsCrossed = []
        for i in range(len(startStopHPStep)+1):
            if i == len(startStopHPStep):
                mask = (hpTimes >= startStopHPStep[i-1,3]) & (hpTimes <= hpTimes[-1])
            elif i == 0:
                mask = (hpTimes >= hpTimes[0]) & (hpTimes <= startStopHPStep[i,2])
            else:
                mask = (hpTimes >= startStopHPStep[i-1,3]) & (hpTimes <= startStopHPStep[i,2])
            if sum(mask)>1:
                #ax15.plot(hpTimes[mask], hindpawRungDist[mask][:,2:9],'.',c='0.5', alpha=0.5)
                llength = len(hindpawRungDist[mask])
                # generate list of tuples which contain the indices of the first three closest rungs
                #pdb.set_trace()
                idx0 = [(x,y) for x,y in zip(range(llength),minhpRungN[mask][:,0])]
                idx1 = [(x,y) for x,y in zip(range(llength),minhpRungN[mask][:,1])]
                idx2 = [(x,y) for x,y in zip(range(llength),minhpRungN[mask][:,2])]
                #hindpawRungDist[mask][:,9:16][tuple(np.array(idx0).T)]
                ax15.plot(hpTimes[mask], hindpawRungDist[mask][:,2:9][tuple(np.array(idx0).T)],'.',c='0.5', alpha=0.5)
                ax15.plot(hpTimes[mask], hindpawRungDist[mask][:,2:9][tuple(np.array(idx1).T)],'.',c='0.7', alpha=0.5)
                ax15.plot(hpTimes[mask], hindpawRungDist[mask][:,2:9][tuple(np.array(idx2).T)],'.',c='0.9', alpha=0.5)

                ax15.plot(hpTimes[mask],np.repeat(np.median(hindpawRungDist[mask][:,2:9][tuple(np.array(idx0).T)]),llength),lw=2)
                #ax15.plot(hpTimes[mask], minhpRDist[mask],'.',c='0.5', alpha=0.5)  # ax0.plot(fp[:,2],yMax-fp[:,3],'.')
                #np.min(np.abs(hindpawRungDist[mask][:,2:9]),axis=1)
                #ax15.plot(hpTimes[mask],np.repeat(np.median(np.min(minhpRDist[mask]),len(hpTimes[mask])),c='C0',lw=2)
                #
                ax17.plot(hpTimes[mask],hindpawRungDist[mask][:,9:16][tuple(np.array(idx0).T)])
                hpRungsCrossed.append([np.where((i-1)>-1,startStopHPStep[i-1,3],hpTimes[0]),numpy.bincount(np.array(hindpawRungDist[mask][:,9:16][tuple(np.array(idx0).T)],dtype=int)).argmax()])
        #ax15.set_ylim(-150,150)
        self.layoutOfPanel(ax15, xLabel=r'time (s)', yLabel=r'min distance (pixel)')
        self.layoutOfPanel(ax17, xLabel=r'time (s)', yLabel=r'rung number')
        #ax10.plot(fTimes[np.array(frontpawRungDist[:, 1], dtype=int)], frontpawRungDist[:, 4], '.', c='C2')

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax18 = plt.subplot(gs[18])
        ax18.set_title('frontpaw : rungs crossed during swing')
        # ax16.set_title('frontpaw linearized swings speed')
        #fpTimes = fTimes[np.array(frontpawRungDist[:, 1], dtype=int)]
        #minfpRDist = np.min(np.abs(frontpawRungDist[:, 2:5]), axis=1)
        #minfpRungN = np.argmin(np.abs(frontpawRungDist[:, 2:5]), axis=1)
        # fpRungsCrossed
        fpRungsCrossed = np.asarray(fpRungsCrossed)
        #for i in range(1,len(fpRungsCrossed)):
        ax18.plot(fpRungsCrossed[:,0][1:],np.diff(fpRungsCrossed[:,1]),'o-')


        self.layoutOfPanel(ax18, xLabel=r'time (s)', yLabel=r'rungs crossed')
        # ax10.plot(fTimes[np.array(frontpawRungDist[:, 1], dtype=int)], frontpawRungDist[:, 4], '.', c='C2')

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax19 = plt.subplot(gs[19])
        ax19.set_title('hindpaw : rungs crossed during swing')
        # ax16.set_title('frontpaw linearized swings speed')
        # hpTimes = fTimes[np.array(hindpawRungDist[:, 1], dtype=int)]
        # minhpRDist = np.min(np.abs(hindpawRungDist[:, 2:5]), axis=1)
        # minhpRungN = np.argmin(np.abs(hindpawRungDist[:, 2:5]), axis=1)
        hpRungsCrossed = np.asarray(hpRungsCrossed)
        #for i in range(1,len(hpRungsCrossed)):
        ax19.plot(hpRungsCrossed[:,0][1:],np.diff(hpRungsCrossed[:,1]),'o-')

        self.layoutOfPanel(ax19, xLabel=r'time (s)', yLabel=r'rungs crossed')
        # ax10.plot(fTimes[np.array(frontpawRungDist[:, 1], dtype=int)], frontpawRungDist[:, 4], '.', c='C2')


        ## save figure ############################################################
        fname = self.determineFileName(date, 'paw-movement', reco=rec)
        #plt.show()
        #plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def generateHistogram(self,mouse,date,rec,gData):

        # pdb.set_trace()
        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 8  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 14, 'axes.titlesize': 13, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(1, 1  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.14, right=0.92, top=0.91, bottom=0.15)

        # sub-panel enumerations
        plt.figtext(0.06, 0.96, '%s   %s   %s' % (self.mouse, date, rec), clip_on=False, color='black', size=14)

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
        ax0 = plt.subplot(gs[0])
        ss = ['850','800','750','700','650','600']
        for i in range(len(gData)):
            print(i)
            ax0.hist(gData[i][2][0],bins=100,range=[-1.75,0.],histtype='step',label='%s mV' % ss[i])

        # removes upper and right axes
        # and moves left and bottom axes away
        self.layoutOfPanel(ax0, xLabel=r'value (V)', yLabel=r'occurrence',Leg=[2,9])
        ax0.set_yscale('log')
        ## save figure ############################################################
        fname = self.determineFileName(date, 'preAmpOutput', reco=rec)
        # plt.show()
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')

    ##########################################################################################
    def createPawMovementFigure(self,date,rec,pawTrackingOutliers):

        # fig = plt.figure(figsize=(11, 11))
        # ax0 = fig.add_subplot(3, 2, 1)
        # ax1 = fig.add_subplot(3, 2, 3)
        # figure #################################
        fig_width = 25  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(5, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[1, 1, 1, 1, 4])

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.98, '%s   %s   %s' % (self.mouse, date, rec), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        gsList = []
        axList = []
        for i in range(4):
            gssub = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[i], hspace=0.2)
            gsList.append(gssub)
            ax0 = plt.subplot(gssub[0])
            ax1 = plt.subplot(gssub[1])
            ax2 = plt.subplot(gssub[2])
            ax3 = plt.subplot(gssub[3])

            axList.append([ax0, ax1, ax2, ax3])
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[4], hspace=0.2)
        ax4 = plt.subplot(gssub1[0])

        ccc = ['C0', 'C1', 'C2', 'C3']

        # fig.set_title(jointName)

        for i in range(4):
            jointName = pawTrackingOutliers[i][4]
            onePawData = pawTrackingOutliers[i][5]
            onePawDataTmp = pawTrackingOutliers[i][6]
            frDispl = pawTrackingOutliers[i][7]
            frDisplOrig = pawTrackingOutliers[i][8]

            axList[i][0].plot(onePawData[:, 0], onePawData[:, 1]*0.025, c='0.5')
            axList[i][0].plot(onePawDataTmp[:, 0], onePawDataTmp[:, 1]*0.025, c=ccc[i])
            if i == 3:
                self.layoutOfPanel(axList[i][0], xLabel='frame number', yLabel='x (pixel)')
            else:
                self.layoutOfPanel(axList[i][0], xLabel=None, yLabel='x (cm)', xyInvisible=[True, False])
            # ax0.set_ylabel('x (pixel)')

            axList[i][1].plot(onePawData[:, 0], onePawData[:, 2]*0.025, c='0.5')
            axList[i][1].plot(onePawDataTmp[:, 0], onePawDataTmp[:, 2]*0.025, c=ccc[i])
            if i == 3:
                self.layoutOfPanel(axList[i][1], xLabel='frame number', yLabel='y (pixel)')
            else:
                self.layoutOfPanel(axList[i][1], xLabel=None, yLabel='y (cm)', xyInvisible=[True, False])
            axList[i][1].invert_yaxis()
            # ax1.set_ylabel('y (pixel)')

            axList[i][2].plot(onePawData[:-1, 0], frDisplOrig, c='0.5')
            axList[i][2].plot(onePawDataTmp[:-1, 0], frDispl, c=ccc[i])
            if i == 3:
                self.layoutOfPanel(axList[i][2], xLabel='frame number', yLabel='speed (pix/frame)')
            else:
                self.layoutOfPanel(axList[i][2], xLabel=None, yLabel='speed (pix/frame)', xyInvisible=[True, False])

            axList[i][3].hist(frDisplOrig, bins=200, color='0.5')
            axList[i][3].hist(frDispl, bins=200, range=(min(frDisplOrig), max(frDisplOrig)),color=ccc[i])
            axList[i][3].set_yscale('log')
            if i == 3:
                self.layoutOfPanel(axList[i][3], xLabel='speed (pixel/frame)', yLabel='occurrence')
            else:
                self.layoutOfPanel(axList[i][3], xLabel=None, yLabel='occurrence', xyInvisible=[True, False])

            # trajectories of all four paws in the same panel ###################################
            # ax2 = fig.add_subplot(3, 2, 2)
            ax4.plot(onePawData[:, 1], onePawData[:, 2], c='0.5')
            ax4.plot(onePawDataTmp[:, 1], onePawDataTmp[:, 2], c=ccc[i], label='%s' % jointName)
            self.layoutOfPanel(ax4, xLabel='x (pixel)', yLabel='y (pixel)', Leg=[1, 9])  # ax2.set_xlabel('x (pixel)')  # ax2.set_ylabel('y (pixel)')



            # ax3 = fig.add_subplot(3, 2, 4)
            # ax3.plot(onePawData[:-1, 0], frDisplOrig, c='0.5')
            # ax3.plot(onePawDataTmp[:-1, 0], frDispl, c='C0')
            # ax3.set_xlabel('frame #')
            # ax3.set_ylabel('movement speed (pixel/frame)')
            #
            # ax4 = fig.add_subplot(3, 2, 5)
            # ax4.hist(frDisplOrig, bins=300, color='0.5')
            # ax4.hist(frDispl, bins=300, range=(min(frDisplOrig), max(frDisplOrig)))
            # ax4.set_xlabel('displacement (pixel)')  # ax4.set_ylabel('occurrence')
            # ax4.set_yscale('log')

        ## save figure ############################################################
        ax4.invert_yaxis()
        rec = rec.replace('/','-')
        fname = self.determineFileName(rec, what='paw_trajectory',date=date)
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

    ##########################################################################################
    def createSwingStanceFigure(self,recs):

        stepNumber = []
        stepDuration = []
        nDays = len(recs)
        lpaw = ['FR','FL','HL','HR']
        print('number of days',nDays)
        for n in range(nDays):
            totalSteps = [[],[],[],[]]
            stepDuration.append([[],[],[],[]])
            stepNumber.append([0.,0.,0.,0.])
            # pdb.set_trace()
            print('number of recordings : ',len(recs[n][4]))
            for j in range(len(recs[n][4])):
                for i in range(4):
                    #pdb.set_trace()
                    #print(len(recs[n][4][j][3][i][1]))
                    idxSwings = recs[n][4][j][3][i][1]
                    recTimes  = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    #pdb.set_trace()
                    # only look at steps during motorization period
                    mask = (recTimes[idxSwings[:,0]]>7.) & (recTimes[idxSwings[:,0]]<26.6)
                    idxSwings = np.asarray(idxSwings)
                    stepNumber[-1][i] += len(idxSwings[mask])/len(recs[n][4])
                    stepDuration[-1][i].extend((recTimes[idxSwings[:,1]+1][mask]-recTimes[idxSwings[:,0][mask]]).tolist())

        stepNumber = np.asarray(stepNumber)
        # pdb.set_trace()
        #stepDuration = np.asarray(stepDuration)

        # fig = plt.figure(figsize=(11, 11))
        # ax0 = fig.add_subplot(3, 2, 1)
        # ax1 = fig.add_subplot(3, 2, 3)
        # figure #################################
        fig_width = 6  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(4, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[1,3,1,1]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.25)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.16, right=0.9, top=0.92, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.96, '%s, %s days of recordings' % (self.mouse, len(recs)), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        ax0 = plt.subplot(gs[0])
        sN = []
        for i in range(4):
            ax0.plot(np.arange(nDays)+1, stepNumber[:, i],'o-',label=lpaw[i])
            sN.append(stepNumber[:, i])

        self.layoutOfPanel(ax0, xLabel=None, yLabel='average steps during trial',Leg=[1,9])
        np.save('testScripts/meanStepNumber.npy',np.asarray(sN))
        # second sub-plot #######################################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=gs[1],hspace=0.4)
        #gssub2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[2], hspace=0.4)
        ax20 = plt.subplot(gs[2])
        ax21 = plt.subplot(gs[3])
        axL = []
        for i in range(4):
            ax = plt.subplot(gssub1[i])
            axL.append(ax)

        #for n in range(nDays):
        sD = []
        for i in range(4):
            stepD = nDays*[[]]
            stepDurationMean = nDays*[[]]
            stepDurationMedian = nDays*[[]]
            for n in range(nDays):
                stepD[n]=stepDuration[n][i]
                stepDurationMean[n] = np.mean(stepDuration[n][i])
                stepDurationMedian[n] = np.median(stepDuration[n][i])
            axL[i].boxplot(stepD,positions=np.arange(nDays)+1,showfliers=False)
            ax20.plot(np.arange(nDays)+1,stepDurationMean,'o-')
            ax21.plot(np.arange(nDays)+1,stepDurationMedian,'o-')
            sD.append(stepDurationMean)
        for i in range(4):
            if i < 3:
                self.layoutOfPanel(axL[i], xLabel=None, yLabel=None,xyInvisible=[True,False])
            else:
                self.layoutOfPanel(axL[i], xLabel='recording session', yLabel='step duration (s)',xyInvisible=[False,False])
        np.save('testScripts/meanStepDuration.npy',np.asarray(sD))
        self.layoutOfPanel(ax20, xLabel=None, yLabel='mean \n step duration (s)',xyInvisible=[True,False])
        self.layoutOfPanel(ax21, xLabel='recording session', yLabel='median \n step duration (s)',xyInvisible=[False,False])

        # save figure #######################################################
        #rec = rec.replace('/','-')
        fname = self.determineFileName(self.mouse, what='swing_stance')
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

    ##########################################################################################
    def createSwingTraceFigure(self,recs,linear=True):

        nDays = len(recs)

        # figure #################################
        fig_width = 25  # width in inches
        fig_height = 22  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[10,4]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.96, '%s, %s days of recordings' % (self.mouse, len(recs)), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # create panels #######################################################
        gssub0 = gridspec.GridSpecFromSubplotSpec(nDays, 8, subplot_spec=gs[0], hspace=0.2)
        axL = []
        for n in range(nDays):
            axL.append([[],[],[],[],[],[],[],[]])
            for i in range(8):
                ax = plt.subplot(gssub0[n*8+i])
                axL[-1][i].append(ax)
        # plot all swing phases ###############################################
        stepDistances = []
        stepC = np.zeros((4,nDays,2))
        for n in range(nDays):
            stepDistances.append([[], [], [], []])
            for j in range(len(recs[n][4])):
                for i in range(4):
                    pawPos   = recs[n][2][j][5][i]
                    linearPawPos = recs[n][4][j][4][i][5]
                    #pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    stepCharacter = recs[n][4][j][3][i][3]
                    stepC[i, n, 1] += len(stepCharacter)
                    recTimes = recs[n][4][j][4][i][2]
                    idxSwings = np.asarray(idxSwings)
                    for k in range(len(idxSwings)):
                        # pdb.set_trace()
                        # only look at steps during motorization period
                        if linear :
                            mask = (linearPawPos[:, 0] >= recTimes[idxSwings[k, 0]]) & (linearPawPos[:, 0] <= recTimes[idxSwings[k, 1]])
                            # pdb.set_trace()
                            stepDistances[-1][i].extend([(linearPawPos[:, 1][mask][-1] - linearPawPos[:, 1][mask][0])])
                            axL[n][i][0].plot(linearPawPos[:, 0][mask] - linearPawPos[:, 0][mask][0], (linearPawPos[:, 1][mask] - linearPawPos[:, 1][mask][0]), '0.5', lw=0.2, alpha=0.2)
                            #axL[n][i][0].hist((linearPawPos[:, 1][mask] - linearPawPos[:, 1][mask][0]),bins=20)
                        else:
                            mask = (pawPos[:,0]>=recTimes[idxSwings[k,0]]) & (pawPos[:,0]<=recTimes[idxSwings[k,1]])
                            #pdb.set_trace()
                            stepDistances[-1][i].extend([(pawPos[:,1][mask][-1]-pawPos[:,1][mask][0])*0.025])
                            axL[n][i][0].plot(pawPos[:,0][mask]-pawPos[:,0][mask][0],(pawPos[:,1][mask]-pawPos[:,1][mask][0])*0.025,'0.5',lw=0.2,alpha=0.2)
                        #ttimes = recTimes[idxSwings[k,0]:idxSwings[k,1]] - recTimes[idxSwings[k,0]]
                        if stepCharacter[k][3]:
                            stepC[i,n,0] += 1
                    if linear:
                        if i < 2:
                            axL[n][i][0].set_ylim(-2, 6)
                            axL[n][i][0].set_xlim(0, 0.4)
                        else:
                            axL[n][i][0].set_ylim(-2, 10)
                            axL[n][i][0].set_xlim(0, 0.5)
                    else:
                        if i <2:
                           axL[n][i][0].set_ylim(-1.25,5)
                           axL[n][i][0].set_xlim(0, 0.4)
                        else:
                           axL[n][i][0].set_ylim(-1.25,7.5)
                           axL[n][i][0].set_xlim(0, 0.5)
        for n in range(nDays):
            if n < (nDays-1):
                for i in range(4):
                    if (i ==0):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel='x (cm)', xyInvisible=[True, False])
                    elif (i == 2):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,True])
            else:
                for i in range(4):
                    if i ==0:
                        self.layoutOfPanel(axL[n][i][0], xLabel='time (s)', yLabel='x (cm)',xyInvisible=[False,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel='time (s)', yLabel=None,xyInvisible=[False,True])

        # plot distance histograms ################################################################
        meanStrideLengths = np.zeros((4, nDays))
        for n in range(nDays):
            for i in range(4):
                meanStrideLengths[i][n] = np.mean(stepDistances[n][i])
                axL[n][i+4][0].axvline(x=0,c='0.7')
                axL[n][i+4][0].hist(stepDistances[n][i],bins=50)
                if linear:
                    if i<2:
                       axL[n][i+4][0].set_ylim(0, 60)
                       axL[n][i+4][0].set_xlim(-2, 6)
                    else:
                       axL[n][i+4][0].set_ylim(0, 40)
                       axL[n][i+4][0].set_xlim(-2, 10)
                else:
                    if i<2:
                       axL[n][i+4][0].set_ylim(0, 30)
                       axL[n][i+4][0].set_xlim(-2, 5)
                    else:
                       axL[n][i+4][0].set_ylim(0, 60)
                       axL[n][i+4][0].set_xlim(-2, 7)

        for n in range(nDays):
            if n < (nDays-1):
                for i in range(4):
                    if (i ==0):
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel='frequency', xyInvisible=[True, False])
                    elif (i == 2):
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel=None,xyInvisible=[True,False])
                    else:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel=None,xyInvisible=[True,True])
            else:
                for i in range(4):
                    if i ==0:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel='distance (cm)', yLabel='frequency',xyInvisible=[False,False])
                    else:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel='distance (cm)', yLabel=None,xyInvisible=[False,True])
        # save figure #######################################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs[1], hspace=0.2)
        ax = plt.subplot(gssub1[0])
        for i in range(4):
            #ax = plt.subplot(gssub1[i])
            ax.plot(np.arange(nDays)+1, meanStrideLengths[i],'o-')
            if i == 0:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel='mean stride length (cm)', xyInvisible=[False, False])
            elif i > 0:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=None, xyInvisible=[False, False])
            ax.set_ylim(1.6,2.9)

            ax1 = plt.subplot(gssub1[4+i])
            ax1.plot(range(nDays), stepC[i,:,0]/stepC[i,:,1])
            if i == 0:
                self.layoutOfPanel(ax1, xLabel='recording day', yLabel='fraction of indecisive steps', xyInvisible=[False, False])
            elif i > 0:
                self.layoutOfPanel(ax1, xLabel='recording day', yLabel=None, xyInvisible=[False, False])


        # save figure #######################################################
        #rec = rec.replace('/','-')
        if linear:
            fname = self.determineFileName(self.mouse, what='swing_traces-linear')
        else:
            fname = self.determineFileName(self.mouse, what='swing_traces')
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

    ##########################################################################################
    def createSwingSpeedProfileFigure(self,recs,linear=True):

        nDays = len(recs)

        # figure #################################
        fig_width = 25  # width in inches
        fig_height = 22  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[10,2]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.96, '%s, %s days of recordings' % (self.mouse, len(recs)), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # create panels #######################################################
        gssub0 = gridspec.GridSpecFromSubplotSpec(nDays, 8, subplot_spec=gs[0], hspace=0.2)
        axL = []
        for n in range(nDays):
            axL.append([[],[],[],[],[],[],[],[]])
            for i in range(8):
                ax = plt.subplot(gssub0[n*8+i])
                axL[-1][i].append(ax)
        # plot all swing phases ###############################################
        speedProfile = []
        for n in range(nDays):
            speedProfile.append([[], [], [], []])
            for j in range(len(recs[n][4])):
                for i in range(4):
                    #print(n,i)
                    #pawPos   = recs[n][2][j][5][i]
                    #linearPawPos = recs[n][4][j][4][i][5]
                    pawSpeed = recs[n][2][j][3][i]
                    idxSwings = recs[n][4][j][3][i][1]
                    recTimes = recs[n][4][j][4][i][2]
                    xSpeed = recs[n][4][j][4][i][1]
                    wSpeed = recs[n][4][j][4][i][0]
                    idxSwings = np.asarray(idxSwings)
                    for k in range(len(idxSwings)):
                        # pdb.set_trace()
                        # only look at steps during motorization period
                        #mask = (pawSpeed[:,0]>=recTimes[idxSwings[k,0]]) & (pawSpeed[:,0]<=recTimes[idxSwings[k,1]])
                        #pdb.set_trace()
                        if (recTimes[idxSwings[k,0]]>7.) and  (recTimes[idxSwings[k,0]]>22.6): # only look at steps during motorization period
                            speedDiff = xSpeed[idxSwings[k,0]:idxSwings[k,1]]*0.025 - wSpeed[idxSwings[k,0]:idxSwings[k,1]]
                            #pdb.set_trace()
                            speedProfile[-1][i].extend(speedDiff)
                            axL[n][i][0].plot(recTimes[idxSwings[k,0]:idxSwings[k,1]]-recTimes[idxSwings[k,0]],speedDiff,'0.5',lw=0.2,alpha=0.2)
                        #ttimes = recTimes[idxSwings[k,0]:idxSwings[k,1]] - recTimes[idxSwings[k,0]]
                    # if linear:
                    #     if i < 2:
                    #         axL[n][i][0].set_ylim(-2, 6)
                    #         axL[n][i][0].set_xlim(0, 0.4)
                    #     else:
                    #         axL[n][i][0].set_ylim(-2, 10)
                    #         axL[n][i][0].set_xlim(0, 0.5)
                    # else:
                    if i <2:
                        axL[n][i][0].set_ylim(-100,220)
                        axL[n][i][0].set_xlim(0, 0.4)
                    else:
                        axL[n][i][0].set_ylim(-100,300)
                        axL[n][i][0].set_xlim(0, 0.5)
        for n in range(nDays):
            if n < (nDays-1):
                for i in range(4):
                    if (i ==0):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel='speed (cm/s)', xyInvisible=[True, False])
                    elif (i == 2):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,True])
            else:
                for i in range(4):
                    if i ==0:
                        self.layoutOfPanel(axL[n][i][0], xLabel='time (s)', yLabel='speed (cm/s)',xyInvisible=[False,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel='time (s)', yLabel=None,xyInvisible=[False,True])

        # plot distance histograms
        #pdb.set_trace()
        speedRanges = [[-100,-10],[-10,30],[30,70],[70,300]]
        histIntegrals = np.zeros((4, len(speedRanges), nDays))
        for n in range(nDays):
            for i in range(4):
                #print('hist',n,i)
                #if i == 2:
                #    print(np.percentile(speedProfile[n][i],95.))
                axL[n][i+4][0].axvline(x=0,c='0.7')
                axL[n][i+4][0].hist(speedProfile[n][i],bins=100,normed=True,cumulative=True,histtype='step')
                (hh,be) = np.histogram(speedProfile[n][i],bins=100,range=[-100,300],normed=True)
                for j in range(len(speedRanges)):
                    mask = (be[:-1]>=speedRanges[j][0]) & (be[1:]<=speedRanges[j][1])
                    histIntegrals[i,j,n] = sum(hh[mask])*(be[1]-be[0])
                # if linear:
                #     if i<2:
                #        axL[n][i+4][0].set_ylim(0, 60)
                #        axL[n][i+4][0].set_xlim(-2, 6)
                #     else:
                #        axL[n][i+4][0].set_ylim(0, 40)
                #        axL[n][i+4][0].set_xlim(-2, 10)
                # else:
                if i<2:
                    #axL[n][i+4][0].set_ylim(0, 30)
                    axL[n][i+4][0].set_xlim(-100,220)
                    #axL[n][i+4][0].set_yscale('log')
                else:
                    #axL[n][i+4][0].set_ylim(0, 60)
                    axL[n][i+4][0].set_xlim(-100, 300)
                    #axL[n][i+4][0].set_yscale('log')
        for i in range(4):
            s1 = np.asarray(speedProfile[0][i])
            s2 = np.asarray(speedProfile[nDays-1][i])
            mask1 = (s1>-150) & (s1<150)
            mask2 = (s2>-150) & (s2<150)
            print('%s ' %i,stats.ks_2samp(s1[mask1], s2[mask2]))
        for n in range(nDays):
            if n < (nDays-1):
                for i in range(4):
                    if (i ==0):
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel='frequency', xyInvisible=[True, False])
                    elif (i == 2):
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel=None,xyInvisible=[True,False])
                    else:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel=None, yLabel=None,xyInvisible=[True,True])
            else:
                for i in range(4):
                    if i ==0:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel='speed (cm/s)', yLabel='frequency',xyInvisible=[False,False])
                    else:
                        self.layoutOfPanel(axL[n][i+4][0], xLabel='speed (cm/s)', yLabel=None,xyInvisible=[False,True])

        # summary panels #######################################################
        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], hspace=0.2)
        for i in range(4):
            ax = plt.subplot(gssub1[i])
            ax.stackplot(np.arange(nDays)+1,  histIntegrals[i],labels=['%s' % i for i in speedRanges])
            if i == 0:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel='% time in speed ranges', xyInvisible=[False, False])
            elif i>0 and i<3:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=None, xyInvisible=[False, True])
            elif i==3:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=None, Leg=[1,9],xyInvisible=[False, True])
        # save figure #######################################################
        #rec = rec.replace('/','-')
        if linear:
            fname = self.determineFileName(self.mouse, what='swing_speed-profile-linear')
        else:
            fname = self.determineFileName(self.mouse, what='swing_speed-profile')
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

    ##########################################################################################
    def createRungCrossingFigure(self,recs):

        nDays = len(recs)
        stepNumber = []
        rungID = []
        nDays = len(recs)

        print('number of days',nDays)
        for n in range(nDays):
            #totalSteps = [[],[],[],[]]
            #stepDuration.append([[],[],[],[]])
            rungID.append([[],[],[],[]])
            # pdb.set_trace()
            print('number of recordings : ',len(recs[n][4]))
            for j in range(len(recs[n][4])):
                for i in range(4):
                    #pdb.set_trace()
                    #print(len(recs[n][4][j][3][i][1]))
                    rungNumbers = recs[n][4][j][3][i][2]
                    #recTimes  = recs[n][4][j][4][i][2]
                    #idxSwings = np.asarray(idxSwings)
                    #pdb.set_trace()
                    # only look at steps during motorization period
                    #mask = (recTimes[idxSwings[:,0]]>7.) & (recTimes[idxSwings[:,0]]<26.6)
                    #idxSwings = np.asarray(idxSwings)
                    #stepNumber[-1][i] += len(idxSwings[mask])/len(recs[n][4])
                    rungID[-1][i].extend(rungNumbers)

        #pdb.set_trace()
        #stepNumber = np.asarray(stepNumber)
        # figure #################################
        fig_width = 25  # width in inches
        fig_height = 20  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(2, 1,  # ,
                               # width_ratios=[1.2,1]
                               height_ratios=[10,2]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.1)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.96, top=0.95, bottom=0.05)

        # sub-panel enumerations
        plt.figtext(0.06, 0.96, '%s, %s days of recordings' % (self.mouse, len(recs)), clip_on=False, color='black', size=14)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # create panels #######################################################
        gssub0 = gridspec.GridSpecFromSubplotSpec(nDays, 4, subplot_spec=gs[0],hspace=0.2)
        axL = []
        for n in range(nDays):
            axL.append([[],[],[],[],[]])
            for i in range(4):
                ax = plt.subplot(gssub0[n*4+i])
                axL[-1][i].append(ax)
        # plot all swing phases ###############################################
        #speedProfile = []
        #percentSteps =
        stepDist = np.array([-1,0,1,2,3,4,5])
        stepsPercentages = np.zeros((4,len(stepDist),nDays))
        for n in range(nDays):
            for i in range(4):
                rungsCrossed = np.diff(rungID[n][i])
                rungsCrossed = rungsCrossed[rungsCrossed>-20]
                axL[n][i][0].plot(np.arange(len(rungsCrossed)),rungsCrossed)
                for j in range(len(stepDist)):
                    stepsPercentages[i,j,n] = np.sum(rungsCrossed==stepDist[j])/len(rungsCrossed)
                if i < 2:
                   axL[n][i][0].set_ylim(-1, 4)
                   #axL[n][i][0].set_xlim(0, 0.4)
                else:
                   axL[n][i][0].set_ylim(-1, 6)

        for n in range(nDays):
            if n < (nDays-1):
                for i in range(4):
                    if (i ==0):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel='rungs crossed', xyInvisible=[True, False])
                    elif (i == 2):
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel=None, yLabel=None,xyInvisible=[True,True])
            else:
                for i in range(4):
                    if i ==0:
                        self.layoutOfPanel(axL[n][i][0], xLabel='step (#)', yLabel='rungs crossed',xyInvisible=[False,False])
                    else:
                        self.layoutOfPanel(axL[n][i][0], xLabel='step (#)', yLabel=None,xyInvisible=[False,True])

        gssub1 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], hspace=0.2)
        for i in range(4):
            ax = plt.subplot(gssub1[i])
            ax.stackplot(np.arange(nDays)+1,  stepsPercentages[i],labels=['%s' % i for i in stepDist])
            if i == 0:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel='fraction of rungs crossed', xyInvisible=[False, False])
            elif i>0 and i<3:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=None, xyInvisible=[False, True])
            elif i==3:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=None, Leg=[1,9],xyInvisible=[False, True])

        np.save('testScripts/rungsCrossedPercentages.npy',stepsPercentages)

        # save figure #######################################################
        #rec = rec.replace('/','-')
        fname = self.determineFileName(self.mouse, what='rungs-crossed')
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

    ##########################################################################################
    def generateR2ValueFigure(self,mouse,Rvalues,figName):

        RA = np.asarray(Rvalues)
        nDays = len(RA)

        #stepNumber = np.asarray(stepNumber)
        # figure #################################
        fig_width = 6  # width in inches
        fig_height = 15  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 12, 'axes.titlesize': 12, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
                  'axes.linewidth': 1.3, 'ytick.major.size': 4,  # major tick size in points
                  'xtick.major.size': 4  # major tick size in points
                  # 'edgecolor' : None
                  # 'xtick.major.size' : 2,
                  # 'ytick.major.size' : 2,
                  }
        rcParams.update(params)

        # set sans-serif font to Arial
        rcParams['font.sans-serif'] = 'Arial'

        # create figure instance
        fig = plt.figure()

        # define sub-panel grid and possibly width and height ratios
        gs = gridspec.GridSpec(5, 1,  # ,
                               # width_ratios=[1.2,1]
                               #height_ratios=[10,2]
                               )
        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.1, hspace=0.4)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

        # sub-panel enumerations
        plt.figtext(0.06, 0.98, '%s, %s days of recordings' % (self.mouse, nDays), clip_on=False, color='black', size=12)
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)
        forecasted = ['wheel speed','FR speed','FL speed','HL speed','HR speed']
        for n in range(5):
            ax = plt.subplot(gs[n])
            ax.set_title(forecasted[n])

            ax.axhline(y=0,ls='--',c='0.6')
            ax.axhline(y=1,ls='-',c='0.6')
            ax.plot(np.arange(1,nDays+1),RA[:,2*n],'o-',label='training score')
            ax.plot(np.arange(1,nDays+1),RA[:,2*n+1],'o-',label='testing score')

            majorLocator_x = ticker.MultipleLocator(1)
            ax.xaxis.set_major_locator(majorLocator_x)
            if n == 4:
                self.layoutOfPanel(ax, xLabel='recording day', yLabel=r'R$^2$')
            elif n==0:
                self.layoutOfPanel(ax, xLabel=None, yLabel=r'R$^2$', Leg=[1, 9])
            else:
                self.layoutOfPanel(ax, xLabel=None, yLabel=r'R$^2$')

        # save figure #######################################################
        #rec = rec.replace('/','-')
        fname = self.determineFileName(self.mouse, what=figName)
        # plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')
        #plt.show()

