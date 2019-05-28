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


    ##########################################################################################
    def determineFileName(self,date,what,reco=None):
        if reco is None:
            ff = self.figureDirectory + '%s_%s' % (date,what)
        else:
            ff = self.figureDirectory + '%s_%s_%s' % (date,reco,what)
        return ff

    ##########################################################################################
    def layoutOfPanel(self, ax,xLabel=None,yLabel=None,Leg=None):

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position(('outward', 10))
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

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
        fname = self.determineFileName(date, 'ephys-walk_traces', reco=rec)

        #plt.savefig(fname + '.png')
        plt.savefig(fname + '.pdf')



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
        plt.subplots_adjust(left=0.05, right=0.96, top=0.96, bottom=0.02)

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
                    timeDiff = np.diff(tracks[i][2])
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
                    timeDiff = np.diff(tracks[i][2])
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