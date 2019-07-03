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

import matplotlib.pyplot as plt
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


#################################################################################
# detect spikes in ephys trace
#################################################################################
def detectPawTrackingOutlies(pawTraces,pawMetaData):
    jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
    threshold = 60

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

        return (len(onePawData),len(onePawDataTmp),onePawIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig)

    pawTrackingOutliers = []
    for i in range(4):
        (tot,correct,correctIndicies,onePawData,onePawDataTmp,frDispl,frDisplOrig) = findOutliersBasedOnMaxSpeed(np.column_stack((pawTraces[:,0],pawTraces[:,(i*3+1)],pawTraces[:,(i*3+2)])),jointNames[i],i)
        pawTrackingOutliers.append([i,tot,correct,correctIndicies,jointNames[i],onePawData,onePawDataTmp,frDispl,frDisplOrig])
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
def calculateDistanceBtwLineAndPoint(x1,y1,x2,y2,x0,y0):
    nenner   = np.sqrt((y2-y1)**2 + (x2-x1)**2)
    zaehler  = (y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1
    dist = zaehler/nenner
    return dist

##########################################################
# (tracks,pawTracks,stanceSwingsParams)
def findStancePhases(tracks, pawTracks,rungMotion,showFigFit=False,showFigPaw=False) :

    speedDiffThresh = 5  # cm/s Speed threshold, determine with variance
    minimalLengthOfSwing = 3 # number of frames @ 200 Hz
    thStance = 10
    thSwing = 2
    trailingStart = 1
    trailingEnd = 1
    bounds = [1400, 5320]

    # error function : difference betweeen paw and wheel speed ; the inverse of the absolute difference is used to emphasize small values which would be the stance phases
    errfunc = lambda p, x1, y1, x2, y2, x3, y3, x4, y4: np.sum(1./np.abs(x1-p*y1)) + np.sum(1./np.abs(x2-p*y2))+ np.sum(1./np.abs(x3-p*y3)) + np.sum(1./np.abs(x4-p*y4))
    # guess some fit parameters
    p0 = 0.025
    # calculate wheel speed at the frame times : requires interpolation of the wheel speed
    interpAngle = interp1d(tracks[5][:,0],tracks[5][:,1])
    interp = interp1d(tracks[2], -tracks[1])
    forFit = []
    for i in range(4):
        mask = ((pawTracks[3][i][:,0])>=min(tracks[2])) & ((pawTracks[3][i][:,0])<=max(tracks[2]))
        newWheelSpeedAtPawTimes = interp(pawTracks[3][i][:,0][mask])
        #
        maskAngle = ((pawTracks[5][i][:,0])>=min(tracks[5][:,0])) & ((pawTracks[5][i][:,0])<=max(tracks[5][:,0]))
        newWheelAngleAtPawTimes = interpAngle(pawTracks[5][i][:,0][maskAngle])
        newX  = (pawTracks[5][i][:,1][maskAngle])*0.025 + (newWheelAngleAtPawTimes*80./360.) - (pawTracks[5][i][:,1][maskAngle][0])*0.025
        forFit.append([newWheelSpeedAtPawTimes, pawTracks[3][i][:,2][mask],pawTracks[3][i][:,0][mask],mask,np.array(pawTracks[3][i][:,4][mask],dtype=int),np.column_stack((pawTracks[5][i][:,0][maskAngle],newX))])

    (p1, success) = scipy.optimize.leastsq(errfunc, p0 ,args=(forFit[0][0],forFit[0][1],forFit[1][0],forFit[1][1],forFit[2][0],forFit[2][1],forFit[3][0],forFit[3][1]))
    print('fit parameter : ', p1)
    if showFigFit :
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,1]*p1)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,2]*p0)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,2]*p1)
        #plt.plot(pawTracks[0][3][1][:,0],pawTracks[0][3][1][:,2]*p1)
        #plt.plot(pawTracks[0][3][2][:,0],pawTracks[0][3][2][:,2]*p1)
        #plt.plot(pawTracks[0][3][3][:,0],pawTracks[0][3][3][:,2]*p1)
        #plt.plot(pawTracks[0][3][0][:,0],pawTracks[0][3][0][:,3]*p1)
        #plt.plot(pawTracks[0][3][0][:,0][mask],newSpeedAtPawTimes)
        #plt.plot(tracks[0][2], -tracks[0][1])
        plt.plot(forFit[0][1]*p1)
        plt.plot(forFit[0][0])
        plt.show()
    #pdb.set_trace()
    ##############################################################################################################
    # calculate paw-rung distance
    pawRungDistances = []
    for i in range(4):
        rungInd = []
        for n in forFit[i][4]:
            rungLocs = rungMotion[3][n][3]
            xPaw = pawTracks[0][n,(i*3+1)]
            yPaw = pawTracks[0][n,(i*3+2)]
            distances = calculateDistanceBtwLineAndPoint(rungLocs[:,0],rungLocs[:,1],rungLocs[:,2],rungLocs[:,3],xPaw,yPaw)
            sortedArguments  = np.argsort(np.abs(distances))
            #closestRungIdx = np.argmin(np.abs(distances))
            closestRungNumber = rungMotion[3][n][2][sortedArguments[0]]
            closestDist = distances[sortedArguments[0]]
            secondClosestRungNumber = rungMotion[3][n][2][sortedArguments[1]]
            secondClosestDist = distances[sortedArguments[1]]
            rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw])
        rungInd = np.asarray(rungInd)
        pawRungDistances.append([i,rungInd])

    ##############################################################################################################
    # determine regions during which the speed is different for more than xLength values #########################
    stanceDistances = [[10, 40],[10,40],[-4,40],[-4,40]]
    swingPhases = []

    for i in range(4):
        #pdb.set_trace()
        print(pawTracks[2][i])
        # thresholded = speedDiff > speedDiffThresh
        # startStop = np.diff(np.arange(len(speedDiff))[thresholded]) > 1
        # mmmStart = np.hstack((([True]), startStop))  # np.logical_or(np.hstack((([True]),startStop)),np.hstack((startStop,([True]))))
        # mmmStop = np.hstack((startStop, ([True])))
        # startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
        # stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
        # minLengthThres = (stopIdx - startIdx) > minLength
        # startStep = startIdx[minLengthThres] - trailingStart
        # endStep = stopIdx[minLengthThres] + trailingEnd
        # return np.column_stack((startStep, endStep))
        ##
        speedDiff = (forFit[i][0] - forFit[i][1]*p1)
        thresholded = np.abs(speedDiff) > speedDiffThresh
        startStop = np.diff(forFit[i][4][thresholded]) > 1 # use indices taking into account missed frames
        mmmStart = np.hstack((([True]), startStop))
        mmmStop = np.hstack((startStop, ([True])))
        startIdx = (np.arange(len(speedDiff))[thresholded])[mmmStart]
        stopIdx = (np.arange(len(speedDiff))[thresholded])[mmmStop]
        swingIndices = np.column_stack((startIdx, stopIdx))
        nIdx = 0
        cleanedSwingIndicies = []
        #pdb.set_trace()
        while True:
            #print(nIdx,forFit[i][4][swingIndices[nIdx,0]],forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]])
            if forFit[i][4][swingIndices[nIdx,1]]-forFit[i][4][swingIndices[nIdx,0]]>0:
                #if swingIndices[nIdx,1]-swingIndices[nIdx,0]>2:
                #cleanedSwingIndicies.append([forFit[i][4][swingIndices[nIdx,0]]-trailingStart,forFit[i][4][swingIndices[nIdx,1]]+trailingEnd])
                sttart = (swingIndices[nIdx,0]-trailingStart) if (swingIndices[nIdx,0]-trailingStart)>0 else 0
                ennd   = (swingIndices[nIdx,1]+trailingEnd) if (swingIndices[nIdx,1]+trailingEnd)<len(forFit[i][4]) else (len(forFit[i][4])-1)
                cleanedSwingIndicies.append([sttart,ennd])
                if (cleanedSwingIndicies[-1][1]-cleanedSwingIndicies[-1][0])< minimalLengthOfSwing :# remove short swing phases
                    del cleanedSwingIndicies[-1]
                if len(cleanedSwingIndicies)>1:
                    if cleanedSwingIndicies[-2][1] > cleanedSwingIndicies[-1][0]: # remove overlapping swing phases
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]
                #startIdx = forFit[i][4][cleanedSwingIndicies[-1][0]]
                #endIdx   = forFit[i][4][cleanedSwingIndicies[-1][1]]
                #meanSpeedDiff = np.mean(p1*forFit[i][1][startIdx:endIdx])
                #if meanSpeedDiff < speedDiffThresh : # remove osciallatory phases
                #    del cleanedSwingIndicies[-1]
                if len(cleanedSwingIndicies) > 1:
                    if (cleanedSwingIndicies[-1][0]-cleanedSwingIndicies[-2][1])<4: # remove very short stance phases
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]

                    #pdb.set_trace()
                    #print(cleanedSwingIndicies)
                    #print(cleanedSwingIndicies[-2][1],cleanedSwingIndicies[-1][0],len(forFit[i][4]))
                if len(cleanedSwingIndicies) > 1: # remove stance phase if distance to rung is too large
                    mask = (pawRungDistances[i][1][:,0]>=forFit[i][4][cleanedSwingIndicies[-2][1]]) & (pawRungDistances[i][1][:,0]<=forFit[i][4][cleanedSwingIndicies[-1][0]])
                    meanDist = np.mean(pawRungDistances[i][1][:,1][mask])
                    stdDist  = np.std(pawRungDistances[i][1][:,1][mask])
                    #print(forFit[i][2][cleanedSwingIndicies[-1][0]],stdDist)
                    if  (meanDist < stanceDistances[i][0]) or (meanDist>stanceDistances[i][1]):
                        cleanedSwingIndicies[-2][1] = cleanedSwingIndicies[-1][1]
                        del cleanedSwingIndicies[-1]

                if len(cleanedSwingIndicies) > 0: # remove swing phases for which there is no change in paw-rung distance
                    mask = (pawRungDistances[i][1][:, 0] >= forFit[i][4][cleanedSwingIndicies[-1][0]]) & (pawRungDistances[i][1][:, 0] <= forFit[i][4][cleanedSwingIndicies[-1][1]])
                    if np.std(pawRungDistances[i][1][:,1][mask])< 2. :
                        del cleanedSwingIndicies[-1]


            nIdx += 1
            if nIdx==(len(swingIndices)-1):
                break
        #pdb.set_trace()
        cSIA = np.asarray(cleanedSwingIndicies)
        # extract rung number of stance phase #####################################
        stanceRungIdentity = []
        for n in range(len(cleanedSwingIndicies)):
            if cleanedSwingIndicies[n][0]>0:
                if n==0:
                    startStanceI = 0
                else:
                    startStanceI = int(cleanedSwingIndicies[n-1][1])
                endStanceI   = int(cleanedSwingIndicies[n][0])
                #print(cleanedSwingIndicies[n],startStanceI,endStanceI)
                (values,counts) = np.unique(pawRungDistances[i][1][:,3][startStanceI:endStanceI],return_counts=True)
                stanceRungIdentity.append(values[np.argmax(counts)])

        # find indecisive steps - steps with bimodal speed profile ################
        stepCharacter = []
        for n in range(len(cleanedSwingIndicies)):
            speedDuringStep = speedDiff[cleanedSwingIndicies[n][0]:cleanedSwingIndicies[n][1]]
            thresholded = np.abs(speedDuringStep) < 10. # use different, larger threshold
            if sum(thresholded)==0:
                indecisiveStep = False
            else:
                startStop = np.diff(np.arange(len(speedDuringStep))[thresholded]) > 2 # use indices taking into account missed frames
                mmmStart = np.hstack((([True]), startStop))
                mmmStop = np.hstack((startStop, ([True])))
                #print(len(speedDuringStep),thresholded,mmmStart,speedDuringStep)
                startIdx = (np.arange(len(speedDuringStep))[thresholded])[mmmStart]
                stopIdx = (np.arange(len(speedDuringStep))[thresholded])[mmmStop]
                closeIndicies = np.column_stack((startIdx, stopIdx))
                # do not consider periods at the beginning or the end of the step
                mask = (startIdx > 3) & (stopIdx < (len(speedDuringStep)-3))
                closeIndicies = closeIndicies[mask]
                if any((closeIndicies[:,1]-closeIndicies[:,0])>=3): # if there are long periods below threshold
                    indecisiveStep = True
                elif len(closeIndicies)>=3: # if there are many values below threshold
                    indecisiveStep = True
                else:
                    indecisiveStep = False
            stepCharacter.append([n, cleanedSwingIndicies[n][0], cleanedSwingIndicies[n][1], indecisiveStep, closeIndicies])
        #pdb.set_trace()
        ###########################################################################
        #if i ==0:
        #    print(forFit[i][2][cSIA])
        #pdb.set_trace()
        #swingIndices[:, 0] = swingIndices[:, 0] - 0
        #swingIndices[:, 1] = swingIndices[:, 1] + 0
        if showFigPaw :
            fig = plt.figure(figsize=(22,7)) 
            ax = fig.add_subplot(1,2,1)
            ax.axvline(x=stanceDistances[i][0],color='0.6')
            ax.axvline(x=stanceDistances[i][1],color='0.6')
            ax.hist(pawRungDistances[i][1][:, 1],bins=100)
            plt.xlabel('paw rung distance (cm/s)')
            plt.ylabel('occurrence')
            #plt.show()

            ax = fig.add_subplot(1,2,2)
            ax.fill_between(forFit[i][2],stanceDistances[i][0],stanceDistances[i][1],color='0.8')
            ax.plot(forFit[i][2], forFit[i][0])
            ax.plot(forFit[i][2], forFit[i][1] * p1)

            ax.plot(forFit[0][5][:,0],forFit[0][5][:,1])
            ax.plot(forFit[1][5][:,0],forFit[1][5][:,1])
            ax.plot(forFit[2][5][:,0],forFit[2][5][:,1])
            ax.plot(forFit[3][5][:,0],forFit[3][5][:,1])
            uniqueRungIdx = np.unique(np.concatenate((pawRungDistances[i][1][:,3],pawRungDistances[i][1][:,6])))
            #c0 = np.append(pawRungDistances[i][1][:,3][1:],0)
            #c1 = np.append(pawRungDistances[i][1][:,6][1:],0)
            for j in uniqueRungIdx:
                rMask1 = pawRungDistances[i][1][:,3] == j
                rMask2 = pawRungDistances[i][1][:,6] == j
                #ax.plot(pawRungDistances[i][1][:,0][rMask1],pawRungDistances[i][1][:,1][rMask1],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
                #ax.plot(pawRungDistances[i][1][:,0][rMask2],pawRungDistances[i][1][:,4][rMask2],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
                ax.plot(forFit[i][2][rMask1],pawRungDistances[i][1][:,1][rMask1],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
                ax.plot(forFit[i][2][rMask2],pawRungDistances[i][1][:,4][rMask2],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
            for n in range(len(cleanedSwingIndicies)):
                startI = int(cleanedSwingIndicies[n][0])
                endI   = int(cleanedSwingIndicies[n][1]) + 1
                #print(n,startI,endI,endI-startI,len(cleanedSwingIndicies))
                if stepCharacter[n][3]:
                    plt.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='red')
                else:
                    plt.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='C2')
                #plt.plot(range(startI,endI),forFit[i][1][startI:endI] * p1,c='C2')
                plt.xlabel('time (s)')
                plt.ylabel('speed (cm/s)')

            #plt.xlim(4610, 4720)
            plt.show()

        swingPhases.append([i,cleanedSwingIndicies,stanceRungIdentity,stepCharacter])
    return (swingPhases,forFit)

