import time
import numpy as np
import sys
import scipy, scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import pdb
import scipy.ndimage
import itertools
from scipy.interpolate import interp1d

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm


def getSpeed(angles,times,circumsphere):
    angleJumps = angles[np.concatenate((([True]),np.diff(angles)!=0.))]
    timePoints = times[np.concatenate((([True]),np.diff(angles)!=0.))]
    angularSpeed = np.diff(angleJumps[::2])/np.diff(timePoints[::2])
    linearSpeed = angularSpeed*circumsphere/360.
    speedTimes = (timePoints[::2][1:] + timePoints[::2][:-1])/2.
    #pdb.set_trace()
    return (angularSpeed,linearSpeed,speedTimes)


def crosscorr(deltat, y0, y1, correlationRange=1.5, fast=False):
    """
            home-written routine to calcualte cross-correlation between two contiuous traces
            new version from February 9th, 2011
    """

    if len(y0) != len(y1):
        print('Data to be correlated has different dimensions!')
        sys.exit(1)

    y0mean = y0.mean()
    y1mean = y1.mean()
    y0sd = y0.std()
    y1sd = y1.std()

    if y0sd != 0 and y1sd != 0:
        y0norm = (y0 - y0mean) / y0sd
        y1norm = (y1 - y1mean) / y1sd
    else:
        y0norm = y0 - y0mean
        y1norm = y1 - y1mean

    # defined range calculation of cross-correlation
    # value is specified in main routine
    # deltat = 0.9

    pointnumber1 = len(y0)

    ncorrrange = np.ceil(correlationRange / deltat)
    corrrange = np.arange(2 * ncorrrange + 1) - ncorrrange
    ycorr = np.zeros(len(corrrange))

    # print corrrange
    if fast:
        pass
    else:
        for n in corrrange:
            corrpairs = pointnumber1 - abs(n)
            # ccc = arange(corrpairs)
            # print n
            if n < 0:
                y1mod = np.hstack((y1norm[int(-abs(n)):], y1norm[:-int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n > -10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1)
            elif n == 0:
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1norm)) / (float(pointnumber1))
                # print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], (float(pointnumber1-1))
            elif n > 0:
                y1mod = np.hstack((y1norm[int(abs(n)):], y1norm[:int(abs(n))]))
                ycorr[int(n + ncorrrange)] = (np.add.reduce(y0norm * y1mod)) / (float(pointnumber1))
                # if n < 10 :
                #       print n, ncorrrange, n+ncorrrange, ycorr[n+ncorrrange], float(pointnumber1-1)
            else:
                print('Problem!')
                exit(1)
            # print n , ycorr[n+ncorrrange]

    float_corrrange = np.array([float(i) for i in corrrange])

    xcorr = float_corrrange * deltat

    normcorr = np.column_stack((xcorr, ycorr))
    return normcorr

    ############################################################
    ## high-pass filter from http://nullege.com/codes/show/src@obspy.signal-0.3.3@obspy@signal@filter.py
    ############################################################

def highpass(data, freq, df=200, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.

    Filter data removing data below certain frequency freq using corners.

    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz; Default 200.
    :param corners: Filter corners. Note: This is twice the value of PITSA's
            filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
            This results in twice the number of corners but zero phase shift in
            the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='highpass', ftype='butter', output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)

############################################################
## high-pass filter from http://stackoverflow.com/questions/12093594/how-to-implement-band-pass-butterworth-filter-with-scipy-signal-butter
############################################################
def butter_highpass(interval, cutoff, sampling_rate, order=5):
    nyq = sampling_rate * 0.5

    stopfreq = float(cutoff)
    cornerfreq = 0.4 * stopfreq  # (?)

    ws = cornerfreq / nyq
    wp = stopfreq / nyq

    # for bandpass:
    # wp = [0.2, 0.5], ws = [0.1, 0.6]

    N, wn = scipy.signal.buttord(wp, ws, 3, 16)  # (?)

    # for hardcoded order:
    # N = order

    b, a = scipy.signal.butter(N, wn, btype='high')  # should 'high' be here for bandpass?
    sf = scipy.signal.lfilter(b, a, interval)
    return sf

#################################################################################
# detect spikes in ephys trace
#################################################################################
def detectSpikeTimes(tresh,eDataHP,ephysTimes,positive=True):
    #global detectionTreshold
    if positive :
        excursion = eDataHP > tresh  # threshold ephys trace
    else:
        excursion = eDataHP < tresh
    excursionInt = np.array(excursion, dtype=int)  # convert boolean array into array of zeros and ones
    diff = excursionInt[1:] - excursionInt[:-1]  # calculate difference
    spikeStart = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == 1))]  # a difference of one is the start of a spike
    spikeEnd = np.arange(len(eDataHP))[np.concatenate((np.array([False]), diff == -1))]  # a difference of -1 is the spike end
    if (spikeEnd[0] - spikeStart[0]) < 0.:  # if trace starts below threshold
        spikeEnd = spikeEnd[1:]
    if (spikeEnd[-1] - spikeStart[-1]) < 0.:  # if trace ends below threshold
        spikeStart = spikeStart[:-1]
    if len(spikeStart) != len(spikeEnd):  # unequal lenght of starts and ends is a problem of course
        print('problem in length of spikeStart and spikeEnd')
        sys.exit(1)
    spikeT = []
    for i in range(len(spikeStart)):
        if (spikeEnd[i] - spikeStart[i]) > 10:  # ignore if difference between end and start is smaller than 15 points
            nMin = spikeStart[i] #np.argmin(eDataHP[spikeStart[i]:spikeEnd[i]]) + spikeStart[i]
            spikeT.append(nMin)
    # detectionTreshold = tresh
    spikeTimes = ephysTimes[np.array(spikeT,dtype=int)]
    return spikeTimes

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
def mapToXbit(inputArray,xBitEncoding):
    oldMin = np.min(inputArray)
    oldMax = np.max(inputArray)
    newMin = 0.
    newMax = 2**xBitEncoding-1.
    normXBit = newMin + (inputArray - oldMin) * newMax / (oldMax - oldMin)
    normXBitInt = np.array(normXBit, dtype=int)
    return normXBitInt

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# dataAnalysis.determineFrameTimes(exposureArray[0],arrayTimes,frames)
def determineFrameTimes(exposureArray,arrayTimes,frames,rec=None):
    display = False
    #pdb.set_trace()
    numberOfFrames = len(frames)
    exposure = exposureArray > 20                # threshold trace
    exposureInt = np.array(exposure, dtype=int)  # convert boolean array into array of zeros and ones
    difference = np.diff(exposureInt)            # calculate difference
    expStart = np.arange(len(exposureArray))[np.concatenate((np.array([False]), difference == 1))]  # a difference of one is the start of a spike
    expEnd = np.arange(len(exposureArray))[np.concatenate((np.array([False]), difference == -1))]  # a difference of -1 is the spike end
    if (expEnd[0] - expStart[0]) < 0.:  # if trace starts above threshold
        expEnd = expEnd[1:]
    if (expEnd[-1] - expStart[-1]) < 0.:  # if trace ends above threshold
        expStart = expStart[:-1]
    frameDuration = expEnd - expStart
    midExposure = (expStart + expEnd)/2
    expStartTime = arrayTimes[expStart.astype(int)]
    expEndTime   = arrayTimes[expEnd.astype(int)]
    #framesIdxDuringRec = np.array(len(softFrameTimes))[(arrayTimes[expEnd[0]]+0.002) < softFrameTimes]
    #framesIdxDuringRec = framesIdxDuringRec[:len(expStart)]

    if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
        recordedFrames = frames[3:(len(midExposure) + 3)]
    elif arrayTimes[int(midExposure[0])]<0.003:
        recordedFrames = frames[2:(len(midExposure) + 2)]
    else:
        recordedFrames = frames[:len(midExposure)]
    print('number of tot. frames, recorded frames, exposures start, end :',numberOfFrames,len(recordedFrames), len(expStart), len(expEnd))
    if display:
        ledON = np.zeros(len(exposureArray))
        for i in range(11):
            ledON[((i*1.)<=arrayTimes) & ((i*1.+0.2)>arrayTimes)] = 1.
        ledON[29.<=arrayTimes] = 1.
        data = np.loadtxt('/home/mgraupe/2019.04.01_000-%s.csv' % (rec[-3:]),delimiter=',',skiprows=1,usecols=(0,1))
        print(len(data))
        plt.plot(arrayTimes,exposureArray/32.)
        plt.plot(arrayTimes,ledON)
        print('fist frame at %s sec' % arrayTimes[int(midExposure[0])],end='')
        if arrayTimes[int(midExposure[0])]<0.015 and arrayTimes[int(midExposure[0])]>=0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[3:(len(midExposure) + 3), 1] - 148.6) / 105.4, 'o-')
            print(3)
        elif arrayTimes[int(midExposure[0])]<0.003:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[2:(len(midExposure) + 2), 1] - 148.6) / 105.4, 'o-')
            print(2)
        else:
            plt.plot(arrayTimes[midExposure.astype(int)], (data[:len(midExposure), 1] - 148.6) / 105.4, 'o-')
            print(0)
        #plt.plot(softFrameTimes[data[:-6,0].astype(np.int)+6],(data[:-6,1]-148.6)/105.4)
        #plt.plot(softFrameTimes,np.ones(len(softFrameTimes)),'|')
        plt.show()

        pdb.set_trace()

    return (expStartTime,expEndTime,recordedFrames)


#################################################################################
# detect spikes in ephys trace
#################################################################################
def applyImageNormalizationMask(frames,imageMetaInfo,normFrame,normImageMetaInfo,mouse, date, rec):
    print(imageMetaInfo, normImageMetaInfo)

    pixelRange = 10
    print('small, large frame : ', np.shape(frames), np.shape(normFrame))
    print('pixel-ratio, x-ratio, y-ratio',  imageMetaInfo[4]/normImageMetaInfo[4],end='')
    fig = plt.figure()
    rect1 = patches.Rectangle(normImageMetaInfo[:2], normImageMetaInfo[2], normImageMetaInfo[3],linewidth=1,edgecolor='C0',facecolor='none')
    rect2 = patches.Rectangle(imageMetaInfo[:2],imageMetaInfo[2],imageMetaInfo[3],linewidth=1,edgecolor='C1',facecolor='none')

    framesF = np.array(frames,dtype=float)
    avgFrame = np.average(frames[:,:,:,0],axis=0)
    # rescale image stack to the resolution of the normalization image
    framesRescaled = scipy.ndimage.zoom(framesF, [1,imageMetaInfo[4]/normImageMetaInfo[4],imageMetaInfo[4]/normImageMetaInfo[4],1], order=3)

    # average across all time points of image stack
    #avgFrameZ = np.average(framesRescaled[:,:,:,0],axis=0)
    # rescale the average image to match pixel-size of normalization image, the re-scaling factor of the ratio of the pixel-sizes : stack/norm
    avgFrameZ = scipy.ndimage.zoom(avgFrame, imageMetaInfo[4]/normImageMetaInfo[4], order=3)
    # x,y location in pixel indices of the stack in the normalization image
    xLoc = int(np.round((imageMetaInfo[0] - normImageMetaInfo[0]) / normImageMetaInfo[4]))
    yLoc = int(np.round((imageMetaInfo[1] - normImageMetaInfo[1]) / normImageMetaInfo[4]))

    # dimensions of the rescaled image
    xDim = np.shape(avgFrameZ)[0]
    yDim = np.shape(avgFrameZ)[1]

    #####################
    ax0 = fig.add_subplot(2,3,4)
    ax0.set_title('avg. of image stack',size=7)
    ax0.imshow(np.transpose(avgFrame))

    ax1 = fig.add_subplot(2,3,5)
    ax1.set_title('avg. of image stack : rescaled to norm. image pixel size',size=7)
    ax1.imshow(np.transpose(avgFrameZ))

    print('image stack : ', np.shape(framesRescaled))
    scipy.io.savemat('%s_%s_%s_imageStackBeforeRescaling.mat' % (mouse, date, rec), mdict={'dataArray': framesF})
    scipy.io.savemat('%s_%s_%s_imageStack.mat' % (mouse, date, rec), mdict={'dataArray': framesRescaled})

    #img_stack_uint8 = mapToXbit(avgFrameZ,8)
    #pdb.set_trace()
    #tiff.imsave('avg_imageStack_scaled.tif', np.array(img_stack_uint8, dtype=np.uint8))

    ax2 = fig.add_subplot(2,3,1)
    ax2.set_title('normalization image with image stack rectangle',size=7)
    ret = patches.Rectangle([(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4],(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4]],imageMetaInfo[2]/normImageMetaInfo[4],imageMetaInfo[3]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(normFrame[0,:,:,0]))
    ax2.add_patch(ret)
    scipy.io.savemat('%s_%s_%s_registrationImage.mat' % (mouse, date, rec), mdict={'dataArray': normFrame[0,:,:,0]})


    ax2 = fig.add_subplot(2,3,6)
    ax2.set_title('area of image stack from normalization image',size=7)
    #ret = patches.Rectangle([(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4],(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4]],imageMetaInfo[3]/normImageMetaInfo[4],imageMetaInfo[2]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]))
    #ax2.add_patch(ret)
    print('norm. image : ', np.shape(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]))
    scipy.io.savemat('%s_%s_%s_normalizationImage.mat' % (mouse, date, rec), mdict={'dataArray': normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0]})




    normFrameF = np.array(normFrame, dtype=float)
    test1 = scipy.ndimage.gaussian_filter1d(normFrameF[:,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),:], 2, axis=1)
    test2 = scipy.ndimage.gaussian_filter1d(test1, 2, axis=2)

    #test1 = scipy.ndimage.gaussian_filter1d(framesF, 2, axis=1)
    #test2 = scipy.ndimage.gaussian_filter1d(test1, 2, axis=2)

    #norm8bit = mapToXbit(test2,8)

    filter1D = scipy.ndimage.gaussian_filter1d(framesRescaled, 2, axis=1)
    filter2D = scipy.ndimage.gaussian_filter1d(filter1D, 2, axis=2)

    norm = filter2D / test2
    norm8bit = mapToXbit(norm,8)

    ax2 = fig.add_subplot(2,3,2)
    ax2.set_title('normalized average image',size=7)
    #ret = patches.Rectangle([(imageMetaInfo[1]-normImageMetaInfo[1])/normImageMetaInfo[4],(imageMetaInfo[0]-normImageMetaInfo[0])/normImageMetaInfo[4]],imageMetaInfo[3]/normImageMetaInfo[4],imageMetaInfo[2]/normImageMetaInfo[4],linewidth=1,edgecolor='r',facecolor='none')
    ax2.imshow(np.transpose(np.average(norm[:,:,:,0],axis=0)))

    plt.show()
    #pdb.set_trace()

    errMatrix = np.zeros((pixelRange*2+1,pixelRange*2+1))
    # #row, col = np.indices(err)
    xRange = np.arange(pixelRange*2+1)
    yRange = np.copy(xRange)
    # #avgFrameZ = np.copy(normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0])
    for xy in itertools.product(xRange, yRange):

         #err = np.abs((normFrame[0,xLoc:(xLoc+xDim),yLoc:(yLoc+yDim),0] - avgFrameZ) ** 2).sum() / (xDim*yDim)
         xStart = xLoc + xy[0] - pixelRange
         yStart = yLoc + xy[1] - pixelRange
         normImg = normFrameF[0,xStart:(xStart+xDim),yStart:(yStart+yDim),0]
         normImgNorm = mapToXbit(normImg,8) #normImg - np.average(normImg)
         avgFrameZNorm = mapToXbit(np.average(framesRescaled[:,:,:,0],axis=0),8) #avgFrameZ - np.average(avgFrameZ)
         errMatrix[xy[0],xy[1]] = ((normImgNorm - avgFrameZNorm) ** 2).sum() / (xDim*yDim)

    minimumIndices = np.argwhere(errMatrix == np.min(errMatrix))

    print('MI :', minimumIndices)
    #pdb.set_trace()


    #xNorm = np.linspace(normImageMetaInfo[0],normImageMetaInfo[0]+
    #ax.add_patch(rect1)
    #ax.add_patch(rect2)
    #ax.set_ylim(normImageMetaInfo[1]-10,normImageMetaInfo[1]+normImageMetaInfo[3]+10)
    #ax.set_xlim(normImageMetaInfo[0]-10,normImageMetaInfo[0]+normImageMetaInfo[2]+10)
    #plt.patches.Rectangle(normImageMetaInfo[:2],normImageMetaInfo[2],normImageMetaInfo[3])
    #plt.patches.Rectangle(imageMetaInfo[:2],imageMetaInfo[2],imageMetaInfo[3])
    #plt.show()
    #pdb.set_trace()
    return norm8bit

##########################################################################################
def layoutOfPanel(ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):


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

#################################################################################
# detect spikes in ephys trace
#################################################################################
def detectPawTrackingOutlies(pawTraces,pawMetaData,showFig=True):
    jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
    threshold = 60

    #fig = plt.figure(figsize=(11, 11))
    #ax0 = fig.add_subplot(3, 2, 1)
    #ax1 = fig.add_subplot(3, 2, 3)
    # figure #################################
    fig_width = 11  # width in inches
    fig_height = 16  # height in inches
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
    gs = gridspec.GridSpec(5,1,  # ,
                           # width_ratios=[1.2,1]
                           height_ratios=[1,1,1,1,4]
                           )

    # define vertical and horizontal spacing between panels
    gs.update(wspace=0.3, hspace=0.2)

    # possibly change outer margins of the figure
    plt.subplots_adjust(left=0.1, right=0.95, top=0.96, bottom=0.05)

    # sub-panel enumerations
    # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

    # first sub-plot #######################################################
    gsList = []
    axList = []
    for i in range(4):
        gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[i], hspace=0.2)
        gsList.append(gssub)
        ax0 = plt.subplot(gssub[0])
        ax1 = plt.subplot(gssub[1])
        axList.append([ax0,ax1])
    ax4 = plt.subplot(gs[4])

    ccc = ['C0','C1','C2','C3']
    #pdb.set_trace()
    def findOutliersBasedOnMaxSpeed(onePawData,jointName,i): # should be an 3 column array frame#, x, y
        frDisplOrig = np.sqrt((np.diff(onePawData[:, 1])) ** 2 + (np.diff(onePawData[:, 2])) ** 2) / np.diff(onePawData[:, 0])
        onePawDataTmp = np.copy(onePawData)
        onePawIndicies = np.arange(len(onePawData))
        # excursionsBoolOld = np.zeros(len(pawDataTmp)-1,dtype=bool)
        nIt = 0
        while True:  # cycle as long as there are large displacements

            frDispl = np.sqrt((np.diff(onePawDataTmp[:,1])) ** 2 + (np.diff(onePawDataTmp[:,2])) ** 2) / np.diff(onePawDataTmp[:, 0])  # calcuate displacement
            excursionsBoolTmp = frDispl > threshold  # threshold displacement
            print(nIt, sum(excursionsBoolTmp))
            nIt += 1
            if sum(excursionsBoolTmp) == 0:  # no excursions above threshold are found anymore -> exit loop
                break
            else:
                onePawDataTmp = onePawDataTmp[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]
                onePawIndicies = onePawIndicies[np.concatenate((np.array([True]), np.invert(excursionsBoolTmp)))]

        print('%s # of positions, # of detected mis-trackings, fraction : ' % (jointName), len(onePawData), len(onePawData) - len(onePawDataTmp), (len(onePawData) - len(onePawDataTmp)) / len(onePawData))
        #pdb.set_trace()
        if showFig:

            #fig.set_title(jointName)

            axList[i][0].plot(onePawData[:, 0], onePawData[:,1], c='0.5')
            axList[i][0].plot(onePawDataTmp[:, 0], onePawDataTmp[:, 1],c=ccc[i])
            if i==3:
                layoutOfPanel(axList[i][0],xLabel='frame number',yLabel='x (pixel)')
            else:
                layoutOfPanel(axList[i][0], xLabel=None, yLabel='x (pixel)',xyInvisible=[True,False])
            #ax0.set_ylabel('x (pixel)')


            axList[i][1].plot(onePawData[:, 0], onePawData[:, 2], c='0.5')
            axList[i][1].plot(onePawDataTmp[:, 0], onePawDataTmp[:, 2],c=ccc[i])
            if i==3:
                layoutOfPanel(axList[i][1], xLabel='frame number', yLabel='y (pixel)')
            else:
                layoutOfPanel(axList[i][1], xLabel=None, yLabel='y (pixel)',xyInvisible=[True,False])
            #ax1.set_ylabel('y (pixel)')

            #ax2 = fig.add_subplot(3, 2, 2)
            ax4.plot(onePawData[:, 1], onePawData[:,2], c='0.5')
            ax4.plot(onePawDataTmp[:, 1], onePawDataTmp[:,2],c=ccc[i],label='%s' % jointName)
            layoutOfPanel(ax4, xLabel='x (pixel)', yLabel='y (pixel)',Leg=[1,9])
            #ax2.set_xlabel('x (pixel)')
            #ax2.set_ylabel('y (pixel)')

            # ax3 = fig.add_subplot(3, 2, 4)
            # ax3.plot(onePawData[:-1, 0], frDisplOrig, c='0.5')
            # ax3.plot(onePawDataTmp[:-1, 0], frDispl, c='C0')
            # ax3.set_xlabel('frame #')
            # ax3.set_ylabel('movement speed (pixel/frame)')
            #
            # ax4 = fig.add_subplot(3, 2, 5)
            # ax4.hist(frDisplOrig, bins=300, color='0.5')
            # ax4.hist(frDispl, bins=300, range=(min(frDisplOrig), max(frDisplOrig)))
            # ax4.set_xlabel('displacement (pixel)')
            # ax4.set_ylabel('occurrence')
            # ax4.set_yscale('log')





        return (len(onePawData),len(onePawDataTmp),onePawIndicies)

    pawTrackingOutliers = []
    for i in range(4):
        (tot,correct,correctIndicies) = findOutliersBasedOnMaxSpeed(np.column_stack((pawTraces[:,0],pawTraces[:,(i*3+1)],pawTraces[:,(i*3+2)])),jointNames[i],i)
        pawTrackingOutliers.append([i,tot,correct,correctIndicies])
    plt.savefig('paw_trajectories.png')
    plt.savefig('paw_trajectories.pdf')
    plt.show()
    return pawTrackingOutliers
    #pdb.set_trace()


#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doCorrelationAnalysis(mouse,allCorrDataPerSession):
    #
    #itertools.combinations(arr,2)
    sessionCorrelations = []
    for nSess in range(len(allCorrDataPerSession)):
        # calculate correcorrelation between individual calcium traces
        fTraces = allCorrDataPerSession[nSess][3][0][0]
        #frameNumbers = [0] + frameNumbers
        combis = list(itertools.combinations(np.arange(len(fTraces)), 2))
        ppCaTraces = []
        for i in range(len(combis)):
            #pdb.set_trace()
            corrTemp = scipy.stats.pearsonr(fTraces[combis[i][0]],fTraces[combis[i][1]])
            ppCaTraces.append([i,combis[i][0],combis[i][1],corrTemp[0],corrTemp[1]])

        ppCaTraces = np.asarray(ppCaTraces)
        # code taken from figure generation
        #for nSess in range(len(allCorrDataPerSession)):
        trialStartUnixTimes = []

        # ca-traces #######################################################

        fTraces = allCorrDataPerSession[nSess][3][0][0]
        timeStamps = allCorrDataPerSession[nSess][3][0][3]
        trials = np.unique(timeStamps[:, 1])
        caTracesDict = {}
        for n in range(len(trials)):

            mask = (timeStamps[:, 1] == trials[n])
            triggerStart = timeStamps[:, 5][mask]
            trialStartUnixTimes.append(timeStamps[:,3][mask][0])
            if n>0:
                if oldTriggerStart>triggerStart[0]:
                    print('problem in trial order')
                    sys.exit(1)
            for i in range(len(fTraces)):
                caTracesTime = (timeStamps[:, 4][mask] - triggerStart)
                caTracesFluo = fTraces[:,mask]
                #pdb.set_trace()
                #caTraces.append(np.column_stack((caTracesTime,caTracesFluo)))
                caTracesDict[n] = np.row_stack((caTracesTime,caTracesFluo))

            oldTriggerStart=triggerStart[0]

        # wheel speed  ######################################################
        wheelTracks = allCorrDataPerSession[nSess][1]
        nFig = 0
        #print(len(wheelTracks))
        wheelSpeedDict = {}
        for n in range(len(wheelTracks)):
            wheelRecStartTime = wheelTracks[n][3]
            if (trialStartUnixTimes[nFig]-wheelRecStartTime)<1.:
                #if not wheelTracks[n][4]:
                #recStartTime = wheelTracks[0][3]
                if nFig>0:
                    if oldRecStartTime>wheelRecStartTime:
                        print('problem in trial order')
                        sys.exit(1)
                wheelTime  = wheelTracks[n][2]
                wheelSpeed = wheelTracks[n][1]
                wheelSpeedDict[nFig] = np.row_stack((wheelTime,wheelSpeed))

                nFig+=1
                oldRecStartTime = wheelRecStartTime
        # paw speed  ######################################################
        pawTracks = allCorrDataPerSession[nSess][2]
        nFig = 0
        pawTracksDict = {}
        pawID = []
        for n in range(len(pawTracks)):
            #if not wheelTracks[n][4]:
            pawRecStartTime = pawTracks[n][4]
            if (trialStartUnixTimes[nFig]-pawRecStartTime)<1.:
                if nFig>0:
                    if oldRecStartTime>pawRecStartTime:
                        print('problem in trial order')
                        sys.exit(1)
                pawTracksDict[nFig] = {}
                for i in range(4):
                    #pdb.set_trace()
                    if nFig==0:
                        pawID.append(pawTracks[n][2][i][0])
                    #pawTracksDict[nFig][i]['pawID'] = pawTracks[n][2][i][0]
                    pawSpeedTime = pawTracks[n][3][i][:,0]
                    pawSpeed     = pawTracks[n][3][i][:,1]
                    pawTracksDict[nFig][i] = np.row_stack((pawSpeedTime, pawSpeed))
                    # interp = interp1d(pawSpeedTime, pawSpeed)
                    # newPawSpeedAtCaTimes = interp(caTracesTime[nFig])
                    # pawTracksDict[i]['pawSpeed'].extend(newPawSpeedAtCaTimes)

                oldRecStartTime = pawRecStartTime
                nFig+=1

        ###################################################################
        # interp = interp1d(wheelTime, wheelSpeed)
        # interpMask = (caTracesTime[nFig] >= wheelTime[0]) & (caTracesTime[nFig] <= wheelTime[-1])
        # newWheelSpeedAtCaTimes = interp(caTracesTime[nFig][interpMask])
        # wheelSpeedAll.extend(newWheelSpeedAtCaTimes)
        wheelAll = []
        pawAll = {}
        for i in range(4):
            pawAll[i] = []
        for n in range(len(pawTracksDict)):
            startInterpTime = np.max((caTracesDict[n][0, 0], wheelSpeedDict[n][0, 0], pawTracksDict[n][0][0, 0], pawTracksDict[n][1][0, 0], pawTracksDict[n][2][0, 0], pawTracksDict[n][3][0, 0]))
            endInterpTime   = np.min((caTracesDict[n][0, -1], wheelSpeedDict[n][0, -1], pawTracksDict[n][0][0, -1], pawTracksDict[n][1][0, -1], pawTracksDict[n][2][0, -1], pawTracksDict[n][3][0, -1]))

            interpMask = (caTracesDict[n][0] >= startInterpTime) & (caTracesDict[n][0] <= endInterpTime)

            interpW = interp1d(wheelSpeedDict[n][0], wheelSpeedDict[n][1])
            newWheelSpeedAtCaTimes = interpW(caTracesDict[n][0][interpMask])
            wheelAll.extend(newWheelSpeedAtCaTimes)

            #aa = np.copy(newWheelSpeedAtCaTimes)
            for i in range(4):
                interpP = interp1d(pawTracksDict[n][i][0], pawTracksDict[n][i][1])
                newPawSpeedAtCaTimes = interpP(caTracesDict[n][0][interpMask])
                pawAll[i].extend(newPawSpeedAtCaTimes)

            #
            #pdb.set_trace()
            if n==0:
                caAll = caTracesDict[n][1:][:,interpMask]
            else:
                caAll = np.column_stack((caAll,caTracesDict[n][1:][:,interpMask]))

        #pdb.set_trace()
        corrWheel = []
        corrPaws  = []
        for i in range(len(caAll)):
            corrWheelTemp = scipy.stats.pearsonr(caAll[i], wheelAll)
            corrWheel.append([i,corrWheelTemp[0],corrWheelTemp[1]])

            corrPaw0Temp = scipy.stats.pearsonr(caAll[i], pawAll[0])
            corrPaw1Temp = scipy.stats.pearsonr(caAll[i], pawAll[1])
            corrPaw2Temp = scipy.stats.pearsonr(caAll[i], pawAll[2])
            corrPaw3Temp = scipy.stats.pearsonr(caAll[i], pawAll[3])
            corrPaws.append([i,corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])


        corrWheel = np.asarray(corrWheel)
        corrPaws = np.asarray(corrPaws)
        sessionCorrelations.append([nSess,ppCaTraces,corrWheel,corrPaws])

    return (sessionCorrelations)

##########################################################
# Get absolute speed of paws
def getPawSpeed(recordingsM, mouse_tracks, showFig=True):
    mouse_speed = []
    for d in range(len(recordingsM)):
        day_speed = []
        for s in range(len(recordingsM[d])):
            sess_speed = []
            for p in range(len(recordingsM[d][s])):
                paw_speed = np.diff(recordingsM[d][s][p][:, 0]) / np.diff(recordingsM[d][s][p][:, 2])
                sess_speed.append(paw_speed)
            day_speed.append(sess_speed)
        mouse_speed.append(day_speed)
    mouse_time = []
    for d in range(len(recordingsM)):
        day_time = []
        for s in range(len(recordingsM[d])):
            sess_time = []
            for p in range(len(recordingsM[d][s])):
                mean_time = np.mean([mouse_tracks[d][s][2], mouse_tracks[d][s][3]], axis=0)
                frames_time = mean_time[recordingsM[d][s][p][:, 2].astype(int)]
                sess_time.append(np.delete(frames_time, 0))
            day_time.append(sess_time)
        mouse_time.append(day_time)

    return mouse_speed, mouse_time
##########################################################
# Get average stride length and min/max values

##########################################################
#
def findStancePhases(speedDiff,speedDiffThresh,thStance, thSwing, trailingStart, trailingEnd, bounds):
    # determine regions during which the speed is different for more than xLength values
    thresholded = (speedDiff > -speedDiffThresh) & (speedDiff < speedDiffThresh)
    startStop = np.diff(np.arange(len(speedDiff))[thresholded]) > 1
    mmmStart = np.hstack((([True]), startStop))  # np.logical_or(np.hstack((([True]),startStop)),np.hstack((startStop,([True]))))
    mmmStop = np.hstack((startStop, ([True])))
    startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
    stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
    minStanceLength = (stopIdx - startIdx) > thStance
    startStance = startIdx[minStanceLength]
    endStance = stopIdx[minStanceLength]
    stanceIndices = np.column_stack((startStance, endStance))
    stanceIndices = stanceIndices[np.any(stanceIndices >= bounds[0], axis=1)]
    stanceIndices = stanceIndices[~np.any(stanceIndices >= bounds[1], axis=1)]
    stanceIndices[:, 0] = stanceIndices[:, 0] + trailingEnd
    stanceIndices[:, 1] = stanceIndices[:, 1] - trailingStart
    swingIndices = np.empty((0,2), int)
    for i in range(len(startStance) - 1):
        swingIndices = np.vstack((swingIndices, [endStance[i], startStance[i + 1]]))

    minSwingLength = (swingIndices[:, 1] - swingIndices[:, 0]) > thSwing
    swingIndices = swingIndices[minSwingLength]
    swingIndices = swingIndices[np.any(swingIndices>=bounds[0], axis=1)]
    swingIndices = swingIndices[~np.any(swingIndices>=bounds[1], axis=1)]
    swingIndices[:, 0] = swingIndices[:, 0]-trailingStart
    swingIndices[:, 1] = swingIndices[:, 1]+trailingEnd
    swingPhases = np.full(len(speedDiff), np.nan)
    for i in range(len(swingIndices)):
        swingPhases[swingIndices[i, 0]:swingIndices[i, 1]] = speedDiff[swingIndices[i, 0]:swingIndices[i, 1]]
    stancePhases = np.full(len(speedDiff), np.nan)
    for i in range(len(stanceIndices)):
        stancePhases[stanceIndices[i, 0]:stanceIndices[i, 1]] = speedDiff[stanceIndices[i, 0]:stanceIndices[i, 1]]
    return swingIndices, swingPhases, stanceIndices, stancePhases
##########################################################
# Plot and/or save paw vs wheel speed difference figure
def plotSpeedDiff(mouse, recday, session,mouse_time, mouse_speedDiff, mouse_swing, trailingStart, trailingEnd,speedDiffThresh, bounds, saveFig=False, showFig=False):
    plt.ioff()
    plt.figure(figsize=[19.20, 10.80])
    plt.suptitle('Mouse:' + mouse + '; Day:' + recday + '; Session:' + session)
    plt.subplot(2,1,1)
    plt.title('Stance and step phases detection during a recording session')
    plt.plot([0,30], [speedDiffThresh,speedDiffThresh], [0,30], [-speedDiffThresh,-speedDiffThresh], linestyle='--', c='0.5')
    # plt.text(0, 20, 'Threshold=%s' % th,fontsize=6 )
    # plt.text(0, -20, 'Threshold=%s' % -th,fontsize=6 )
    FR_paw, = plt.plot(mouse_time[0][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)], mouse_speedDiff[0][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)] , c='b', label='Front right paw')
    plt.plot(mouse_time[0][bounds[0]:bounds[1]], mouse_swing[0][1][bounds[0]:bounds[1]], c='slateblue')
    FL_paw, = plt.plot(mouse_time[1][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)], mouse_speedDiff[1][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)] , c='orange', label='Front left paw')
    plt.plot(mouse_time[1][bounds[0]:bounds[1]], mouse_swing[1][1][bounds[0]:bounds[1]], c='moccasin')

    plt.xlabel('Time during recording session(s)')
    plt.ylabel('X speed difference between a paw and the wheel (a.u.)')
    plt.legend(handles=[FR_paw, FL_paw])

    plt.subplot(2,1,2)
    plt.plot([0,30], [speedDiffThresh,speedDiffThresh], [0,30], [-speedDiffThresh,-speedDiffThresh], linestyle='--', c='0.5')
    # plt.text(0, 20, 'Threshold=%s' % th,fontsize=6 )
    # plt.text(0, -20, 'Threshold=%s' % -th,fontsize=6 )
    HR_paw, = plt.plot(mouse_time[3][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)], mouse_speedDiff[3][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)] , c='b', label='Hind right paw')
    plt.plot(mouse_time[3][bounds[0]:bounds[1]], mouse_swing[3][1][bounds[0]:bounds[1]], c='slateblue')
    HL_paw, = plt.plot(mouse_time[2][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)], mouse_speedDiff[2][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)] , c='orange', label='Hind left paw')
    plt.plot(mouse_time[2][bounds[0]:bounds[1]], mouse_swing[2][1][bounds[0]:bounds[1]], c='moccasin')
    plt.xlabel('Time during recording session(s)')
    plt.ylabel('X speed difference between a paw and the wheel (a.u.)')
    plt.legend(handles=[HR_paw, HL_paw])
    if saveFig:
        plt.savefig('/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/%s/PawWheelSpeedDiff_fig/%s_%s.pdf' % (mouse, recday, session[:-4] + '_' + session[-3:]))
    if showFig:
        plt.show()
    else:
        plt.close()
    plt.ion()
##########################################################
def plotHist(mouse_speedDiff, bounds, showHist=False, showStats=False):
    speedProfile = []
    if showHist:
        plt.figure(figsize=[10.80,19.20])
    a = [0,5,11]
    for d in range(len(mouse_speedDiff)):
        FR_profile = np.array([])
        FL_profile = np.array([])
        HL_profile = np.array([])
        HR_profile = np.array([])
        for s in range(len(mouse_speedDiff[d])):
            FR_profile = np.concatenate((FR_profile, mouse_speedDiff[d][s][0][bounds[0]:bounds[1]]))
            FL_profile = np.concatenate((FR_profile, mouse_speedDiff[d][s][1][bounds[0]:bounds[1]]))
            HL_profile = np.concatenate((FR_profile, mouse_speedDiff[d][s][2][bounds[0]:bounds[1]]))
            HR_profile = np.concatenate((FR_profile, mouse_speedDiff[d][s][3][bounds[0]:bounds[1]]))
        speedProfile.append((FR_profile, FL_profile, HL_profile, HR_profile))
        if showHist:
            plt.subplot(len(mouse_speedDiff), 4, 4 * d + 1) #(3,1,a.index(d)+1)
            plt.ylim((1, 24000))
            plt.yscale('log')
            plt.xlim(-100, 200)
            plt.hist(FR_profile, bins=100)
            plt.xlabel('Paw and wheel speed difference')
            plt.ylabel('Frequency for 5 videos of sqme day')
            plt.title('Day : %s' % (d + 1))
            plt.axvline(FR_profile.mean(), color='k', linestyle='dashed')
            plt.axvline(np.percentile(FR_profile, 5), color='0.5', linestyle=':')
            plt.axvline(np.percentile(FR_profile, 95), color='0.5', linestyle=':')
            if d == 0:
                plt.title('Front right paw')

            plt.subplot(len(mouse_speedDiff), 4, 4 * d + 2)
            plt.ylim((1, 24000))
            plt.yscale('log')
            plt.xlim(-100, 200)
            plt.hist(FL_profile, bins=100)
            plt.axvline(FL_profile.mean(), color='k', linestyle='dashed')
            plt.axvline(np.percentile(FL_profile, 5), color='0.5', linestyle=':')
            plt.axvline(np.percentile(FL_profile, 95), color='0.5', linestyle=':')
            if d == 0:
                plt.title('Front left paw')

            plt.subplot(len(mouse_speedDiff), 4, 4 * d + 3)
            plt.hist(HL_profile, bins=100)
            plt.ylim((1, 24000))
            plt.yscale('log')
            plt.xlim(-100, 200)
            plt.axvline(HL_profile.mean(), color='k', linestyle='dashed')
            plt.axvline(np.percentile(HL_profile, 5), color='0.5', linestyle=':')
            plt.axvline(np.percentile(HL_profile, 95), color='0.5', linestyle=':')
            if d == 0:
                plt.title('Hind left paw')

            plt.subplot(len(mouse_speedDiff), 4, 4 * d + 4)
            plt.hist(HR_profile, bins=100)
            plt.axvline(HR_profile.mean(), color='k', linestyle='dashed')
            plt.axvline(np.percentile(HR_profile, 5), color='0.5', linestyle=':')
            plt.axvline(np.percentile(HR_profile, 95), color='0.5', linestyle=':')
            plt.ylim((1, 24000))
            plt.yscale('log')
            plt.xlim(-100, 200)
            if d == 0:
                plt.title('Hind right paw')

        stats = np.empty((12,4,4))
        for d in range(len(speedProfile)):
            for p in range(len(speedProfile[d])):
                stats[d, p, 0] = speedProfile[d][p].mean()
                stats[d, p, 1] = np.median(speedProfile[d][p])
                stats[d, p, 2] = np.percentile(speedProfile[d][p], 5)
                stats[d, p, 3] = np.percentile(speedProfile[d][p], 95)
    return stats

