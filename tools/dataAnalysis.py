import time
import numpy as np
import sys
import os
import scipy, scipy.io
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tifffile as tiff
import pdb
import scipy.ndimage
import itertools
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
import cv2
from scipy import signal
from scipy.signal import find_peaks
import pickle

from scipy import ndimage
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib
matplotlib.use('WXAgg')


def getSpeed(angles,times,circumsphere,minSpacing):
    angleJumps = angles[np.concatenate((([True]),np.diff(angles)!=0.))] # find angles at the points where the angle value changes
    timePoints = times[np.concatenate((([True]),np.diff(angles)!=0.))]  # find times at the points where the angle value changes
    #
    dt = np.diff(timePoints) # delta t values of the time array
    dtMultipleSpace = dt//minSpacing # how many times does the minSpacing fit in the gaps
    dtMultipleSpace = dtMultipleSpace[dtMultipleSpace>1] # gap must be as big as 2 times the spacing to add a point
    # note that gap must be larger than 2 times the spacing
    startGap = np.arange(len(timePoints))[np.hstack(((dt>2.*minSpacing),np.array(False)))] # index values of the start of the gap
    endGap = np.arange(len(timePoints))[np.hstack((np.array(False),(dt>2.*minSpacing)))] # index values of the end of the gap
    newSpacingValue = (timePoints[endGap] - timePoints[startGap]) / dtMultipleSpace
    newTvalues = []
    newAvalues = []
    for i in range(len(dtMultipleSpace)):
        newTvalues.extend(timePoints[startGap][i] + newSpacingValue[i] * np.arange(1, dtMultipleSpace[i]))
        newAvalues.extend(np.repeat(angleJumps[startGap][i], (dtMultipleSpace[i] - 1)))
    timesNew = np.hstack((timePoints,np.asarray(newTvalues)))
    anglesNew = np.hstack((angleJumps,np.asarray(newAvalues)))
    both = np.row_stack((timesNew,anglesNew))
    bothSorted = both[:,both[0].argsort()]
    angularSpeed = np.diff(bothSorted[1])/np.diff(bothSorted[0])
    angularSpeedM = (angularSpeed[1:]+angularSpeed[:-1])/2.

    linearSpeed = angularSpeedM*circumsphere/360.
    speedTimes = bothSorted[0][1:-1]
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
def generatePlotWithSTD(data,std=[2,3,4],names = None):
    matplotlib.use('TkAgg')
    nData = len(data)
    fig = plt.figure()
    for i in range(nData):
        ax0 = fig.add_subplot(nData,1,i+1)
        if names is not None:
            ax0.set_title('%s' % names[i])
        STD = np.std(data[i])
        MM  = np.mean(data[i])
        for n in range(len(std)):
            ax0.axhline(y=MM+std[n]*STD,ls='--',c=plt.cm.RdYlBu(n/len(std)),label='%s STD' % std[n])
            ax0.axhline(y=MM-std[n]*STD, ls='--', c=plt.cm.RdYlBu(n/len(std)))
        ax0.axhline(y=MM,c='C1',label='mean')
        ax0.plot(data[i],c='C0')
        if i==2:
            ax0.plot(np.abs(data[i]), c='C4')
        ax0.legend()
    plt.show(block=False)

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
def determineFramesToExclude(frames,probIdx):
    listOfFramesToExclude = []
    canBeUsed = True
    # first let's decide on how many LED's (if any) are present in the FOV
    for i in range(len(probIdx)):
        currIdx = probIdx[i]
        continueDetectLoop = True
        while continueDetectLoop:
            print('checking idx, started at :', currIdx, probIdx[i])
            frame8bit = np.array(np.transpose(frames[currIdx]), dtype=np.uint8)
            img = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            # rungs = []
            #imgPure = img.copy()
            cv2.imshow("PureImage", img)
            print('e if to exclude; r to remove from exclude; left right arrows to go back-forward one frame; o to specify another idx; f to move to next; x if recording contains too many errors and cannot be used :')
            PressedKey = cv2.waitKey(0)
            print(PressedKey)
            if PressedKey == 81: # left arrow key
                currIdx -=1
            elif PressedKey == 83: # right arrow key
                currIdx +=1
            elif PressedKey == 101: # y key
                print('%s added to exclude list' %currIdx)
                listOfFramesToExclude.append(currIdx)
            elif PressedKey == 114: # e key
                print('%s removed from exclude list' % currIdx)
                listOfFramesToExclude.remove(currIdx)
            elif PressedKey == 111 : # o key
                nIdx = input('specify a new idx to check :')
                currIdx = int(nIdx)
            elif PressedKey == 102: # f key
                continueDetectLoop = False
            elif PressedKey == 120: # x key
                canBeUsed = False
                break
            else:
                print('Key not recognized, try again.')
            print('current exclude list :',listOfFramesToExclude)
        if not canBeUsed:
            break
    cv2.destroyWindow("PureImage") # only destroy window at the end of the exploration
    lofEx = list(dict.fromkeys(listOfFramesToExclude)) # removes duplicates
    lofEx.sort()
    print('starting list of indexes :', probIdx)
    print('indexes to exclude :', lofEx)
    lofEx = np.asarray(lofEx,dtype=int)
    #pdb.set_trace()
    return (lofEx, canBeUsed)


#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# ([ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],verbose=True)
def determineErroneousFrames(frames):
    # first threshold metrics of the movie to detect and exclude erroneous frames with horizontal lines, flash-back frames #########################
    frameDiff = []
    lineDiff = []
    for i in range(len(frames)):
        if i>0:
            frameDiffAllPix = cv2.absdiff(frames[i],frames[i-1])
            fD = np.average(frameDiffAllPix)
            frameDiff.append(fD)
        lineDiffAllLines = cv2.absdiff(frames[i][:,1:],frames[i][:,:-1])
        lD = np.average(lineDiffAllLines,axis=0)
        lineDiff.append(lD)
        #pdb.set_trace()
    frameDiff = np.asarray(frameDiff)
    lineDiff = np.asarray(lineDiff)
    lineDiffSum = np.sum(lineDiff,axis=1)
    frameDiffDiff = np.diff(frameDiff)
    generatePlotWithSTD([lineDiffSum,frameDiff,frameDiffDiff],std=[3,3.5,4],names=['lineDiffSum','frameDiff','diff of FrameDiff'])
    # trick to display the above image
    frame8bit = np.array(np.transpose(frames[0]), dtype=np.uint8)
    img = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
    cv2.imshow('HoldImage',img)
    cv2.waitKey(0) #cv2.imshow()
    cv2.destroyWindow('HoldImage')
    thresholdingInput = input("Specify which trace to use (lineDiffSum - 1, frameDiff - 2, diff of frameDiff - 3; and which multiple of the STD (e.g. 1 3.5); type '4 0' if recording cannote be used (too many errors) : ")
    threshold = [float(i) for i in thresholdingInput.split()]
    print('choice :', threshold)
    #pdb.set_trace()
    if threshold[0] == 1.:
        thresholded = lineDiffSum > np.mean(lineDiffSum) + np.std(lineDiffSum)*threshold[1]
        outlierIdx = np.arange(len(lineDiffSum))[thresholded]  # use indices taking into account missed frames
    elif threshold[0] == 2.:
        thresholded = frameDiff > np.mean(frameDiff) + np.std(frameDiff) * threshold[1]
        outlierIdx = np.arange(len(frameDiff))[thresholded]  # use indices taking into account missed frames
        outlierIdx += 1 # this is since the difference trace does not start at at the first frame but at the difference between first and second frame
    elif threshold[0] == 3.:
        thresholded = np.abs(frameDiffDiff) > np.mean(frameDiffDiff) + np.std(frameDiffDiff)*threshold[1]
        outlierIdx = np.arange(len(frameDiffDiff))[thresholded]  # use indices taking into account missed frames
        outlierIdx += 2  # this is since the difference trace does not start at at the first frame but at the difference between first and second frame
    elif threshold[0] == 4.:
        canBeUsed = False
        idxToExclude = np.arange([])
        return (idxToExclude, canBeUsed)
    print('length and identity of possible erronous frames :' , len(outlierIdx), outlierIdx)
    (idxExclude,canBeUsed) = determineFramesToExclude(frames,outlierIdx)
    #excludeMask = np.ones(len(ledVideoRoi[2]),dtype=bool)
    # add indicies for equivalent frames
    sameFrames = (frameDiff == 0)
    sameFrameIdx = np.arange(len(frameDiff))[sameFrames]
    sameFrameIdx += 1
    print('same frames were recorded here :', sameFrameIdx)
    idxToExclude = np.sort(np.concatenate((sameFrameIdx, idxExclude)))
    #pdb.set_trace()
    #excludeMask[idxToExclude] = False
    plt.close('all')
    return (idxToExclude,canBeUsed)

#################################################################################
# maps an abritray input array to the entire range of X-bit encoding
#################################################################################
# ([ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],verbose=True)
def determineFrameTimesBasedOnLED(ledVideoRoi, cameraExposure, ledDAQc, pc, verbose=False, tail=False):
    ##############################################################################################################
    # auxiliary function to convert bimodal trace into boolean array
    def traceToBinary(trace,threshold=None):
        rescaledTrace = (trace - np.min(trace)) / (np.max(trace) - np.min(trace))
        if threshold is None:
            rescaledTraceBin = rescaledTrace > 0.3
        else:
            rescaledTraceBin = rescaledTrace > threshold
        return (rescaledTrace,rescaledTraceBin)
    ##############################################################################################################
    def traceToBinaryForChangingMaxMin(trace,threshold=None):
        maxTrace = ndimage.maximum_filter(trace, size=5*2)
        minTrace = ndimage.minimum_filter(trace, size=5*2)
        rescaledTrace = (trace - minTrace) / (maxTrace - minTrace)
        if threshold is None:
            rescaledTraceBin = rescaledTrace > 0.3
        else:
            rescaledTraceBin = rescaledTrace > threshold
        return (rescaledTrace,rescaledTraceBin)
    ##############################################################################################################

    # maps LED daq control trace to boolean array ################################################################
    # TODO this number is zero on the behavior setup and 4 here
    if pc == 'behaviorPC':
        LEDcontrolIdx = 0 # which trace of the DAQ recording is linked to the  !!!
    elif pc == '2photonPC':
        LEDcontrolIdx = 4
    else:
        print('Make sure the computer of the recording is specified.')
    ledDAQcontrolBin = traceToBinary(ledDAQc[0][LEDcontrolIdx])[1]  # here the threshold is not important as the trace is binary to start out with
    # convert LED roi traces from video to boolean arrays
    ledVideoRoiBins = []
    ledVideoRoiRescaled = []
    allLEDVideoRoiValues = []
    # determine threshold   [ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude]
    # tail covering the LEDs for some
    if tail:
        anticipateCorrectValues = True
        for i in range(ledVideoRoi[1][0]):
            plt.plot(ledVideoRoi[0][i],'o-',ms=2,label='%s' % i)
        plt.legend(loc=1)
        plt.show()
        if anticipateCorrectValues:
            inputA = input('Index until which the recording is not affected by the tail (integer; type 0 if recording is ok) :')
            untilOKidx = int(inputA)
            if untilOKidx != 0:
                period = [7, 7, 7, 5]
                for i in range(ledVideoRoi[1][0]):
                    maxVal = np.max(ledVideoRoi[0][i][20:untilOKidx])
                    minVal = np.min(ledVideoRoi[0][i][20:untilOKidx])
                    for n in range(period[i]):
                        # repeatValue(ledVideoRoi[0][i][(untilOKidx+n):],7)
                        isHigh = [True if abs(ledVideoRoi[0][i][(untilOKidx + n)] - maxVal) < abs(ledVideoRoi[0][i][(untilOKidx + n)] - minVal) else False]
                        if isHigh:
                            ledVideoRoi[0][i][(untilOKidx + n):][::period[i]] = ledVideoRoi[0][i][(untilOKidx + n)]
                        else:
                            ledVideoRoi[0][i][(untilOKidx + n):][::period[i]] = ledVideoRoi[0][i][(untilOKidx + n)]  # ledVideoRoi[0][i][]
        else:
            maxV = [254,251,250,213]
            minV = [200,174,217,147]
            idxMaxV = [[8580,8582,8583,8585],
                       [],
                       [8589],
                       []]
            idxMinV = [[8581,8584,8586,8588],
                       [8579,8580],
                       [8590,8591],
                       [8584,8585,8586,8589,8590,8591]]
            for i in range(4):
                for n in idxMaxV[i]:
                    ledVideoRoi[0][i][n] = maxV[i]
                for m in idxMinV[i]:
                    ledVideoRoi[0][i][m] = minV[i]

        #fig = plt.figure()
        for i in range(ledVideoRoi[1][0]):
            #ax = fig.add_subplot(3,1,i)
            plt.plot(ledVideoRoi[0][i],'o-',ms=2,label='%s' % i)
            #ax.set_xlim(8)
        plt.legend(loc=1)
        plt.show()
        #pdb.set_trace()
    ###########
    for i in range(ledVideoRoi[1][0]):
        allLEDVideoRoiValues.extend(traceToBinaryForChangingMaxMin(ledVideoRoi[0][i])[0]) # rescale all values to [0,1] and stack them
    allLEDVideoRoiValues = np.sort(np.asarray(allLEDVideoRoiValues)) # convert to array and sort
    luminocityDifferences = np.diff(allLEDVideoRoiValues)
    idxMaxDiff = np.argmax(luminocityDifferences)
    LEDVideoThreshold =  allLEDVideoRoiValues[idxMaxDiff] + (allLEDVideoRoiValues[idxMaxDiff+1] - allLEDVideoRoiValues[idxMaxDiff])/2.
    if pc == 'behaviorPC':
        illumLEDcontrolThreshold = LEDVideoThreshold**4.49185827 # mapping, i.e. exponent, from tools/fitOfIlluminationValues
    elif pc == '2photonPC':
        illumLEDcontrolThreshold = LEDVideoThreshold**2.61290794 # 2pinvivo

    print('thresholds : ',LEDVideoThreshold, illumLEDcontrolThreshold)
    #LEDVideoThreshold = 0.8
    #illumLEDcontrolThreshold = 0.6
    #print('adjusted thresholds : ', LEDVideoThreshold, illumLEDcontrolThreshold)
    #pdb.set_trace()
    # threshold and convert to binary
    for i in range(ledVideoRoi[1][0]):
        ledVideoRoiBins.append(traceToBinary(ledVideoRoi[0][i],threshold=LEDVideoThreshold)[1])  # 0.6 before 0.4
        ledVideoRoiRescaled.append(traceToBinary(ledVideoRoi[0][i])[0])

    # find start and end of camera exposure period ################################################################
    exposureInt = np.array(cameraExposure[0][0], dtype=int)  # convert boolean array into array of zeros and ones
    difference = np.diff(exposureInt)            # calculate difference
    expStart = np.arange(len(exposureInt))[np.concatenate((np.array([False]), difference > 0))]  # a difference of one is the start of the exposure
    expEnd = np.arange(len(exposureInt))[np.concatenate((np.array([False]), difference < 0))]  # a difference of -1 is the end the exposure period
    #pdb.set_trace()
    if (expEnd[0] - expStart[0]) < 0.:  # if trace starts above threshold
        print('exposure at start of recording')
        expEnd = expEnd[1:]
        exposureAtStart = True
    else:
        exposureAtStart = False
    if (expEnd[-1] - expStart[-1]) < 0.:  # if trace ends above threshold
        print('exposure during end of recording')
        exposureAtEnd = True
        expStart = expStart[:-1]
    else:
        exposureAtEnd = False
    expStart = expStart.astype(int)
    expEnd   = expEnd.astype(int)
    expStartTime = cameraExposure[1][expStart] # everything was based on indicies up to this point : here indicies -> time
    expEndTime   = cameraExposure[1][expEnd]   # everything was based on indicies up to this point : here indicies -> time
    frameDuration = expEndTime - expStartTime
    print('first frame started at ', expStartTime[0]*1000., 'ms' )

    ## based on exposure start-stop, how bright should the DAQ LED signal be ##########################################
    startEndExposureTime = np.column_stack((expStartTime, expEndTime))
    startEndExposurepIdx = np.column_stack((expStart,expEnd)) # create a 2-column array with 1st column containing start and 2nd column containing end index
    illumLEDcontrol = [np.mean(ledDAQcontrolBin[b[0]:b[1]]) for b in startEndExposurepIdx]  # extract MEAN illumination value - from LED control trace - during exposure period
    illumLEDcontrol = np.asarray(illumLEDcontrol)
    (illumLEDcontrolrescaled,illumLEDcontrolBin) = traceToBinary(illumLEDcontrol,threshold=illumLEDcontrolThreshold) # 0.2 and 0.15 before

    ## loop over frame numbers and extract binary number shown by leds ################################################
    nFrames = len(ledVideoRoiBins[0])
    binNumbers = np.array([[False,False,False],[True,False,False],[False,True,False],[True,True,False],[False,False,True],[True,False,True],[False,True,True],[True,True,True]])
    recordedFrames = 0
    frameCount = []
    binNumberInFrame = np.column_stack((ledVideoRoiBins[0],ledVideoRoiBins[1],ledVideoRoiBins[2]))
    frameNBefore = 0
    oldI = -1
    exceptionsInFrameCount = []
    idxToExclude = ledVideoRoi[5]
    for i in range(nFrames):
        if i not in idxToExclude:
            matchBool = np.all(np.equal(binNumberInFrame[i],binNumbers),axis=1) # which of the boolean number corresponds to the current frame pattern : return is a boolean list from 0 to 8 with one TRUE entry
            matchFrameN = np.arange(len(binNumbers))[matchBool][0]   # converts the boolean list into the index corresponding to the match
            frameDiff = matchFrameN - frameNBefore  # difference in count to previous frame
            if frameDiff < 0: # else : negative difference indicates that the counter restarted
                frameDiff+=7
            if (frameDiff != 1) and (frameDiff != -6):
                print(i,oldI,i-oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1])
                exceptionsInFrameCount.append([i,oldI,i-oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1]])
            if matchFrameN == 0: # counter will start at 0 and possibly go back to zero after end of recording
                if (i>10) and (i<(nFrames-10)):
                    print(i,oldI,matchFrameN,frameNBefore,frameDiff,binNumberInFrame[i],binNumberInFrame[i-1])
                    print('strange, zero frame in the middle of recording')
                    pdb.set_trace()
                #frameDiff = 0
            frameCount.append([i,matchFrameN,frameDiff,int(ledVideoRoiBins[3][i]),oldI])
            frameNBefore = matchFrameN
            oldI = i
    frameCount = np.asarray(frameCount,dtype=int) # convert list to integer array
    idxRecordedFrames = np.cumsum(frameCount[:,2]) # use the frame differences to generate new index corresponding to video recording
    idxCounting = np.argwhere(idxRecordedFrames>0) # start and end index with first and last frame recording the counter
    idxFramesDuringRecording = idxRecordedFrames[idxCounting[0,0]:(idxCounting[-1,0]+1)] - 1  # remove leading and trailing zeros, and remove one to have the new index start with zero, cumsum makes first index to be 1
    #pdb.set_trace()
    if exposureAtStart:   # remove first frame if exposure was active during start of recording, i.e., at t = 0 s
        idxFramesDuringRecording = idxFramesDuringRecording[1:] - 1
    if exposureAtEnd:
        idxFramesDuringRecording = idxFramesDuringRecording[:-1]
    #pdb.set_trace()
    idxMissingFrames = np.delete(np.arange(idxFramesDuringRecording[-1]+1),idxFramesDuringRecording)

    #idxTestMask = idxFramesDuringRecording < len(illumLEDcontrolBin) # index should not exceed length of array
    #illum = illumLEDcontrolBin[idxFramesDuringRecording[idxTestMask]]
    ##  the excluded frames - based on distortions - need to be removed from the video sequence
    videoRoi = ledVideoRoiBins[3]
    mask = np.ones(len(videoRoi),dtype=bool)
    mask[idxToExclude]  = False
    #pdb.set_trace()
    ## first loop to align the START of the frame recording - in illumLEDcontrolBin - with the video recording
    if any(idxToExclude < 20):
        print('Early frames to exclude. Problem!')
        pdb.set_trace()
    else:
        tmpIdx = np.where(videoRoi==True) # Index of the first ON frame for the 4th LED
        idxFirstFrameRec = tmpIdx[0][0] # extract index of first frame during recording
        if exposureAtStart:
            idxFirstFrameRec+=1         # increase that
        for j in range(10):
            videoRoiWOEX = videoRoi[mask][j:]
            if np.all(videoRoiWOEX[:20] == illumLEDcontrolBin[:20]): # note that illumLEDcontrolBin already accounts for a recording during stat of rec, this frame is removed
                missedFramesBegin = j
                break
    if (idxFirstFrameRec == missedFramesBegin) or (missedFramesBegin == 0):
        print('Number of frames recorded before first full exposed frame during recording :', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20] )
        videoRoiWOEX = videoRoi[mask][missedFramesBegin:]
    elif idxFirstFrameRec == (missedFramesBegin+1):
        missedFramesBegin+=1
        print('Number of frames recorded before first full exposed frame during recording (increased by one):', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20])
        videoRoiWOEX = videoRoi[mask][missedFramesBegin:]
    else:
        print('Number of frames recorded before first full exposed frame during recording :', missedFramesBegin, idxFirstFrameRec, illumLEDcontrolBin[:20],videoRoi[:20] )
        print('Problem with determining index of first recorded frame.')
        pdb.set_trace()
    #pdb.set_trace()
    ## second loop in order to align the
    shiftDifference = []
    lengthOfIllumLEDcontrol = len(illumLEDcontrolBin)
    lengthOfROIinVideo = len(videoRoiWOEX)
    lengthOfIdxCount = idxFramesDuringRecording[-1] + 1
    print('length of illumLEDcontrolBin and videoRoiWOEX and idxFramesDuringRecording[-1] : ', lengthOfIllumLEDcontrol, lengthOfROIinVideo, lengthOfIdxCount)
    for i in range(-10,11,1): # loop to shift the mask over
        idxTemp = idxFramesDuringRecording + i # shift the array by increasing the indicies by a certain number
        idxIllum = idxTemp[(idxTemp>=0)&(idxTemp<lengthOfIllumLEDcontrol)] # indicies have to be larger than zero and should not be larger than the length of the illumLEDcontrolBin array
        #idxIllum = idx[idx<lengthOfIllumLEDcontrol]  # indicies should not be larger than the length of the illumLEDcontrolBin array
        illum = illumLEDcontrolBin[idxIllum]  # illumination at these indicies

        # idxMissing = np.delete(np.arange(idxIllum[-1]), idxIllum) #[i:]
        # idxMissing = np.delete(np.arange(idxFramesDuringRecording[-1]), idxIllum)  # [i:]
        idxMissing = np.delete(np.arange(lengthOfIllumLEDcontrol), idxIllum)
        NidxRemovedAtExtremities = idxIllum[0] + ((lengthOfIllumLEDcontrol-1) - idxIllum[-1]) # counts number of frames missing in the beginning and end
        NidxRemovedAtExtremities -= np.sum((idxMissingFrames<idxIllum[0]) | (idxMissingFrames>idxIllum[-1])) # reduce if missing frames are in the extrimities
        #if i<0:
        #    frameOverlap = [0 if ((lengthOfIllumLEDcontrol+np.abs(i)+1)<(lengthOfROIinVideo+len(idxMissingFrames))) else ((lengthOfROIinVideo+len(idxMissingFrames)) - (lengthOfIllumLEDcontrol+np.abs(i)+1))]
        #elif i>=0:
        #    frameOverlap = [i if (lengthOfIllumLEDcontrol<(lengthOfROIinVideo+len(idxMissingFrames))) else (lengthOfIllumLEDcontrol-(lengthOfROIinVideo+len(idxMissingFrames)+i+1))]
        #print('overlap :',frameOverlap[0])
        compareIdx = np.sum(idxMissing == idxMissingFrames) - len(idxMissing)  + np.abs(NidxRemovedAtExtremities) # abs(i)

        len0 = len(illum)
        len1 = lengthOfROIinVideo
        if len0 < len1:
            compare = np.sum(np.equal(illum,videoRoiWOEX[:len0]))
            versch = compare - len0
            totLength = len0
            largeLength = len1
        else:
            compare = np.sum(np.equal(illum[:len1],videoRoiWOEX))
            versch = compare - len1
            totLength = len1
            largeLength = len0

        #pdb.set_trace()
        shiftDifference.append([i, versch, totLength, compareIdx,NidxRemovedAtExtremities])
        print(i, versch, totLength, compareIdx, NidxRemovedAtExtremities, idxMissing, idxMissingFrames)
        #if i >=0 :
    #pdb.set_trace()
    #compare = np.equal()
    shiftDifference = np.asarray(shiftDifference)
    if len(idxToExclude)==0:   # without erronous frames ...
        shiftDifference = shiftDifference[shiftDifference[:,0]>=0]   # ... use only the shifts which are larger than zero
    shiftToZero = shiftDifference[:,0][(shiftDifference[:,1]==0) & (shiftDifference[:,3]==0)]
    finalLength = shiftDifference[:,2][(shiftDifference[:,1]==0) & (shiftDifference[:,3]==0)]
    if len(shiftToZero)>1 or len(shiftToZero)==0:
        if len(shiftToZero)==0:
            idxTemp = idxFramesDuringRecording + 0
            idx = idxTemp[idxTemp >= 0]
            idxIllum = idx[idx < len(illumLEDcontrolBin)]
            # pdb.set_trace()
            shortest = [len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) if len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) < len(illumLEDcontrolrescaled[idxIllum]) else len(
                illumLEDcontrolrescaled[idxIllum])]
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]], illumLEDcontrolrescaled[idxIllum][:shortest[0]], 'o', ms=1)
            plt.show()
            pdb.set_trace()
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]], 'o-')
            plt.plot(illumLEDcontrolrescaled[idxIllum][:shortest[0]], 'o-')
            plt.show()
            pdb.set_trace()
        elif shiftToZero[1] == (shiftToZero[0]+5):
            print('Multiple shifts to zero, so multiple perfect overlays exist. First overlay with shift %s will be used.' % shiftToZero[0])
            pass
        else:
            print('Problem! More than one shift led to perfect overlay!')
            #np.arange(np.diff(idxRecordedFrames)>1)
            #
            idxTemp = idxFramesDuringRecording + 0
            idx = idxTemp[idxTemp>=0]
            idxIllum = idx[idx<len(illumLEDcontrolBin)]
            #pdb.set_trace()
            shortest = [len(ledVideoRoiRescaled[3][mask][missedFramesBegin:]) if len(ledVideoRoiRescaled[3][mask][missedFramesBegin:])<len(illumLEDcontrolrescaled[idxIllum]) else len(illumLEDcontrolrescaled[idxIllum])]
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]],illumLEDcontrolrescaled[idxIllum][:shortest[0]],'o',ms=1)
            plt.show()
            pdb.set_trace()
            plt.plot(ledVideoRoiRescaled[3][mask][missedFramesBegin:][:shortest[0]],'o-')
            plt.plot(illumLEDcontrolrescaled[idxIllum][:shortest[0]],'o-')
            plt.show()
            pdb.set_trace()
    finalShiftToZero = shiftToZero[0]
    print('final shift to zero and final length : ', finalShiftToZero,finalLength[0])
    idxTemp = idxFramesDuringRecording + finalShiftToZero
    idx = idxTemp[idxTemp>=0]
    idxIllumFinal = idx[idx<len(illumLEDcontrolBin)][:finalLength[0]]
    compareIllumination = False
    if compareIllumination:
        illum = illumLEDcontrolrescaled[idxIllumFinal]
        videoROI = ledVideoRoiRescaled[3][mask][missedFramesBegin:]
        shortest = [len(videoROI) if len(videoROI) < len(illum) else len(illum)]
        bothCombined = np.column_stack((videoROI[:shortest[0]],illum[:shortest[0]]))
        plt.plot(videoROI[:shortest[0]], illum[:shortest[0]], 'o')
        plt.show()
        pdb.set_trace()
        illumValues = pickle.load( open('illuminatoinValues.p', 'rb' ) )
        illumValues.append(bothCombined)
        pickle.dump(illumValues, open('illuminatoinValues.p', 'wb'))
    frameTimes = startEndExposureTime[idxIllumFinal]
    frameStartStopIdx = startEndExposurepIdx[idxIllumFinal]
    videoIdx = np.arange(len(ledVideoRoiBins[3]))[mask][missedFramesBegin:][:finalLength[0]]
    #recFrames = ledVideoRoi[2][videoIdx]
    ddd = np.diff(idxIllumFinal)
    print('Total number of dropped and excluded frames : ', np.sum(ddd-1), 'out of',len(ledVideoRoi[2]),'frame in total.')
    print('Excluded frames :', len(idxToExclude))
    print('Dropped frames :', np.sum(ddd-1)-len(idxToExclude))
    frameSummary = np.array([len(ledVideoRoi[2]),np.sum(ddd-1),len(idxToExclude), np.sum(ddd-1)-len(idxToExclude)])
    #pdb.set_trace()
    return (idxIllumFinal,frameTimes,frameStartStopIdx,videoIdx,frameSummary)
    ##############################################################################################################################
    #pdb.set_trace()

    for i in range(10):
        #print(i)
        #idxTest = idxRecordedFramesCleaned[1:-3] - 1
        #compare = ledVideoRoiBins[3][2:-3] == illumLEDcontrolBin[idxTest]
        #shortestLength = [len(illum) if (0<(len(ledVideoRoiBins)-(len(illum)+i))) else ]

        videoRoi = ledVideoRoiBins[3][i]
        if len(videoRoi) > len(illum):
            #shortestLength = len(illum)
            #else:
            #shortestLength = len(videoRoi)
            print('problem in length relations')
            pdb.set_trace()
        compare = illum[:len(videoRoi)] == videoRoi
        differences =  np.sum(np.invert(compare))
        print('number of differences :', i, differences,i,len(illum)+i,len(videoRoi))
        shifting.append([i,differences,i,len(illum)+i,len(videoRoi)])
    shifting = np.asarray(shifting)
    correctShift = np.argwhere(shifting[:,1]==0)
    if len(correctShift) == 0:
        print('No perfect overlay has been found')
        print(shifting)
        pdb.set_trace()
    elif len(correctShift)>1:
        print('Multiple corret overlays have been found. Suspicious!')
        print(shifting)
        pdb.set_trace()
    elif len(correctShift) == 1:
        rightShift = shifting[correctShift[0][0]]
        print('The correct shift is ', rightShift)
        print('Number of recorded videos :', len(ledVideoRoi[2][rightShift[2]:rightShift[3]]))
        print('Number of associated time points :', len(startEndExposurepIdx[idxRecordedFramesCleaned[idxTestMask]][:rightShift[4]]))
        ddd = np.diff(idxRecordedFramesCleaned[idxTestMask][:rightShift[4]])
        print('Number of gaps, number of lost frames, and size of gaps :', len(ddd[ddd>1]),np.sum(ddd[ddd>1]) - len(ddd[ddd>1]), ddd[ddd>1])
        idxVideo = np.arange(rightShift[2],rightShift[3])
        idxTimePoints = idxRecordedFramesCleaned[idxTestMask][:rightShift[4]]
        #pdb.set_trace()
        return (idxVideo,idxTimePoints,startEndExposureTime,startEndExposurepIdx,rightShift)


    if compare == 0:
        #plt.plot(ledVideoRoi[0][3][2:-3], 'o-', label='ledVideoRoi')
        ii = 3
        plt.plot(ledVideoRoiRescaled[3][ii:len(illum)+ii], 'o-', label='ledVideoRoiRescaled')
        plt.plot(ledVideoRoiBins[3][ii:len(illum)+ii],'o-',label='ledVideoRoiBins')
        plt.plot(illumLEDcontrol[idxRecordedFramesCleaned[idxTestMask]],'o-',label='illumLEDcontrol')
        plt.plot(illumLEDcontrolBin[idxRecordedFramesCleaned[idxTestMask]], 'o-', label='illumLEDcontrolBin')
        plt.legend()

        plt.show()
        pdb.set_trace()
        #compare = illumLEDcontrol[idxTest] ==
        #idxTest = idxRecordedFramesCleaned[1:-1]-1
        totLength = len(illumLEDcontrol[idxTest])
        ret = np.array_equal(illumLEDcontrol[idxRecordedFramesCleaned],ledVideoRoiBins[3][i:(totLength+i)])
        print(i,ret)
        pdb.set_trace()
    pdb.set_trace()



    if len(illumLEDcontrol) <= len(ledVIDEOroi):
        illuminationLonger = True
        ledVIDEOroiMask = np.arange(len(ledVIDEOroi)) < len(illumination)
        illuminationMask = np.arange(len(illumination)) < len(illumination)
        cc = crosscorr(1,illumination,ledVIDEOroi[ledVIDEOroiMask],20) # calculate cross-correlation between LED in video and LED from DAQ array
    else:
        illumniationLonger = False
        ledVIDEOroiMask = np.arange(len(ledVIDEOroi)) < len(ledVIDEOroi)
        illuminationMask = np.arange(len(illumination)) < len(ledVIDEOroi)
        cc = crosscorr(1, illumination[illuminationMask], ledVIDEOroi, 3)  # calculate cross-correlation between LED in video and LED from DAQ array
    peaks = find_peaks(cc[:,1],height=0)
    if len(peaks[0]) > 1:
        print('MULTIPLE peaks found in cross-correlogram between LED brigthness and DAQ array')
        pdb.set_trace()
    elif len(peaks[0]) == 0:
        print('NO peaks were found in cross-correlogram between LED brigthness and DAQ array')
        pdb.set_trace()
    else:
        pdb.set_trace()
        shift = cc[:,0][peaks[0][0]]
        shiftInt = int(shift)
        print('video trace has to be shifted by (float and int number) ', shift, shiftInt)
    #print(len(ledVIDEOroi),len(illumination))
    #pdb.set_trace()
    if verbose:
        if shiftInt >= 0:
            plt.plot(ledVIDEOroi[ledVIDEOroiMask][shiftInt:],'o-',ms=1,label='Video roi (shifted)')
        else:
            plt.plot(ledVIDEOroi[ledVIDEOroiMask][:shiftInt], 'o-', ms=1, label='Video roi (shifted)')
        plt.plot(illumination[illuminationMask],'o-',ms=1,label='from LED daq control')
        plt.legend()
        plt.show()

    frameIdx = np.arange(len(ledVIDEOroi))
    recordedFramesIdx = frameIdx[shiftInt:(len(illumination)+shiftInt)]
    #pdb.set_trace()
    return (startEndExpTime,startEndExpIdx,recordedFramesIdx)
    #####  end of current implementation ##############################################################################################
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
# convert ca traces in easily usable numpy array
#################################################################################
def getCaWheelPawInterpolatedDictsPerDay(nSess,allCorrDataPerSession,allStepData):
    showFig = False
    baselineTime = 5.
    # calcium traces ##############################################################
    trialStartUnixTimes = []

    fTraces = allCorrDataPerSession[nSess][3][0][0]
    timeStamps = allCorrDataPerSession[nSess][3][0][3] # the array containing the time-stamp array
    recordings = np.unique(timeStamps[:, 1]) # determine how many recordings where performed
    caTracesDict = {}
    for n in range(len(recordings)):
        mask = (timeStamps[:, 1] == recordings[n])
        triggerStart = timeStamps[:, 5][mask]
        trialStartUnixTimes.append(timeStamps[:, 3][mask][0])
        if n > 0:
            if oldTriggerStart > triggerStart[0]:
                print('problem in trial order')
                sys.exit(1)
        # for i in range(len(fTraces)):
        # triggerstart - time of the acq start trigger for the current acquisition
        # timeStamps[:, 4][mask] - time of the first pixel in the frame passed since acqModeEpoch
        caTracesTime = (timeStamps[:, 4][mask] - triggerStart) # triggerStart is negative
        #pdb.set_trace()
        caTracesFluo = fTraces[:, mask]
        # pdb.set_trace()
        # caTraces.append(np.column_stack((caTracesTime,caTracesFluo)))
        caTracesDict[n] = np.row_stack((caTracesTime, caTracesFluo))
        #print(np.shape(np.row_stack((caTracesTime, caTracesFluo))))
        oldTriggerStart = triggerStart[0]

    # wheel speed  ######################################################
    # also find calmest pre-motorization period
    minPreMotorMeanV = 1000.
    wheelTracks = allCorrDataPerSession[nSess][1]
    nRec = 0
    # print(len(wheelTracks))
    wheelSpeedDict = {}
    for n in range(len(wheelTracks)):
        wheelRecStartTime = wheelTracks[n][3]
        if (trialStartUnixTimes[nRec] - wheelRecStartTime) < 1.:
            # if not wheelTracks[n][4]:
            # recStartTime = wheelTracks[0][3]
            if nRec > 0:
                if oldRecStartTime > wheelRecStartTime:
                    print('problem in trial order')
                    sys.exit(1)
            wheelTime = wheelTracks[n][2]
            wheelSpeed = wheelTracks[n][1] # linear wheel speed in cm/s
            angleSpeed = wheelTracks[n][0]
            angleTime  = wheelTracks[n][5]

            wheelSpeedDict[nRec] = np.row_stack((wheelTime, wheelSpeed))
            #pdb.set_trace()
            preMMask = (wheelTime < baselineTime)
            preMmeanV = np.mean(np.abs(wheelSpeed[preMMask]))
            #print(nSess, nRec, preMmeanV)
            if preMmeanV < minPreMotorMeanV:
                slowestRec = nRec
                minPreMotorMeanV = np.copy(preMmeanV)

            nRec += 1
            oldRecStartTime = wheelRecStartTime

    # normalize ca-traces by baseline fluorescence : fluorescence during the first baselineTime seconds in the last active recording #############################################
    mask = (caTracesDict[slowestRec][0] < baselineTime)
    F0 = np.mean(caTracesDict[slowestRec][1:][:, mask], axis=1)
    #pdb.set_trace()
    for n in range(len(recordings)):
        normalizedCaTraces = (caTracesDict[n][1:] - F0[:, np.newaxis]) / F0[:, np.newaxis]
        caTracesDict[n][1:] = np.copy(normalizedCaTraces)

    # pdb.set_trace()
    # paw speed  ######################################################
    pawTracks = allCorrDataPerSession[nSess][2]
    nRec = 0
    pawTracksDict = {}
    pawID = []
    for n in range(len(pawTracks)):
        # if not wheelTracks[n][4]:
        pawRecStartTime = pawTracks[n][4]
        if (trialStartUnixTimes[nRec] - pawRecStartTime) < 1.:
            if nRec > 0:
                if oldRecStartTime > pawRecStartTime:
                    print('problem in trial order')
                    sys.exit(1)
            pawTracksDict[nRec] = {}
            for i in range(4):
                # pdb.set_trace()
                if nRec == 0:
                    pawID.append(pawTracks[n][2][i][0])
                # pawTracksDict[nFig][i]['pawID'] = pawTracks[n][2][i][0]
                pawSpeedTime = pawTracks[n][3][i][:,0] # times of cleared paw speed
                pawSpeed = pawTracks[n][3][i][:,1]    # 1 is combined speed in 2-d plane of the camera view from below,
                pawTracksDict[nRec][i] = np.row_stack((pawSpeedTime,pawSpeed))  # interp = interp1d(pawSpeedTime, pawSpeed)  # newPawSpeedAtCaTimes = interp(caTracesTime[nFig])  # pawTracksDict[i]['pawSpeed'].extend(newPawSpeedAtCaTimes)

            oldRecStartTime = pawRecStartTime
            nRec += 1
    # interpolation #############################################################################
    # interp = interp1d(wheelTime, wheelSpeed)
    # interpMask = (caTracesTime[nFig] >= wheelTime[0]) & (caTracesTime[nFig] <= wheelTime[-1])
    # newWheelSpeedAtCaTimes = interp(caTracesTime[nFig][interpMask])
    # wheelSpeedAll.extend(newWheelSpeedAtCaTimes)

    wheelSpeedDictInterp = wheelSpeedDict.copy()
    pawTracksDictInterp = pawTracksDict.copy()
    caTracesDictInterp = caTracesDict.copy()

    for nrec in range(len(caTracesDict)):
        # determine interpolation range
        startInterpTime = np.max((caTracesDict[nrec][0, 0], wheelSpeedDict[nrec][0, 0], pawTracksDict[nrec][0][0, 0], pawTracksDict[nrec][1][0, 0], pawTracksDict[nrec][2][0, 0], pawTracksDict[nrec][3][0, 0]))
        endInterpTime = np.min((caTracesDict[nrec][0, -1], wheelSpeedDict[nrec][0, -1], pawTracksDict[nrec][0][0, -1], pawTracksDict[nrec][1][0, -1], pawTracksDict[nrec][2][0, -1], pawTracksDict[nrec][3][0, -1]))

        interpMask = (caTracesDict[nrec][0] >= startInterpTime) & (caTracesDict[nrec][0] <= endInterpTime)

        # restrict ca-traces to interpolation range
        #pdb.set_trace()
        #matrix = np.copy(caTracesDict[nrec])
        caTracesDictInterp[nrec] = caTracesDict[nrec][:,interpMask]

        # interpolate wheel speed
        interpWheel = interp1d(wheelSpeedDict[nrec][0], wheelSpeedDict[nrec][1])#,kind='cubic')
        newWheelSpeedAtCaTimes = interpWheel(caTracesDict[nrec][0][interpMask])
        wheelSpeedDictInterp[nrec] = np.row_stack((caTracesDict[nrec][0][interpMask], newWheelSpeedAtCaTimes))

        # interpolate paw speed
        for i in range(4):
            interpPaw = interp1d(pawTracksDict[nrec][i][0], pawTracksDict[nrec][i][1])#,kind='cubic')
            newPawSpeedAtCaTimes = interpPaw(caTracesDict[nrec][0][interpMask])
            pawTracksDictInterp[nrec][i] = np.row_stack((caTracesDict[nrec][0][interpMask], newPawSpeedAtCaTimes))
            #pawAll[i].extend(newPawSpeedAtCaTimes)
        if showFig:
            cc = ['C0','C1']
            fig, ax = plt.subplots(figsize=(10, 6))
            for i in range(2):
                ax.plot(pawTracksDictInterp[nrec][i][0],pawTracksDictInterp[nrec][i][1]/np.max(pawTracksDictInterp[nrec][i][1]),c=cc[i])

                idxSwings = allStepData[nSess][4][nrec][3][i][1]
                # print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nSess][4][nrec][4][i][2]
                # pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                for k in range(len(idxSwings)):  # loop over all swings
                    startSwingTime = recTimes[idxSwings[k, 0]]
                    endSwingTime = recTimes[idxSwings[k, 1]]
                    ax.fill_between((startSwingTime,endSwingTime), 0, 1, color='0.6', alpha=0.5, transform=ax.get_xaxis_transform())
                    ax.fill_between((startSwingTime,endSwingTime), 0, 1, color='0.4', alpha=0.5, transform=ax.get_xaxis_transform())

            #ax.plot(caTracesDictInterp[nrec][0],caTracesDictInterp[nrec][1]/np.max(caTracesDictInterp[nrec][1]),c='black')
            plt.show()
            pdb.set_trace()

    return (wheelSpeedDictInterp,pawTracksDictInterp,caTracesDictInterp,wheelSpeedDict,pawTracksDict,caTracesDict)

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doRegressionAnalysis(mouse,allCorrDataPerSession,borders=None):
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    #SVR(kernel='rbf', C=1e3, gamma=0.1)
    #from sklearn.ensemble import RandomForestRegressor
    regressionN = 6
    recs = [0, 1, 2, 3, 4]
    Rvalues = []
    for nSess in range(len(allCorrDataPerSession)):
        print(allCorrDataPerSession[nSess][0],nSess)
        (wheelSpeedDict, pawTracksDict, caTracesDict,aa,bb,cc) = getCaWheelPawInterpolatedDictsPerDay(nSess, allCorrDataPerSession)
        if ((len(wheelSpeedDict) != 5) or (len(pawTracksDict) != 5) or (len(caTracesDict) != 5)):
            print('problem in number of recordings listed in dictionaries')
        # loop over 5 different regressions, each using a different combination of test and train samples
        RTempValues = []

        for reg in range(5): # loop over all recordings
            recsForTraining = recs.copy()
            recsForTraining.remove(reg)
            recsForTest = [reg]
            Rval1 = []
            for d in range(regressionN):  # loop over wheel speed, the four paw speeds and the combined speed
                #print('recording iteration %s, variable %s' %(reg,d))
                #pdb.set_trace()
                # concatenate data
                if borders is not None:
                    timeMaskTrain = (caTracesDict[recsForTraining[0]][0]>=borders[0])&(caTracesDict[recsForTraining[0]][0]<=borders[1])
                    timeMaskTest = (caTracesDict[recsForTest[0]][0] >= borders[0]) & (caTracesDict[recsForTest[0]][0] <= borders[1])
                else:
                    timeMaskTrain = (caTracesDict[recsForTraining[0]][0]>=0)&(caTracesDict[recsForTraining[0]][0]<=1000.)
                    timeMaskTest  = (caTracesDict[recsForTest[0]][0]>=0)&(caTracesDict[recsForTest[0]][0]<=1000.)
                X = np.copy(caTracesDict[recsForTraining[0]][1:][:,timeMaskTrain])
                Xtest = np.copy(caTracesDict[recsForTest[0]][1:][:,timeMaskTest])
                #pdb.set_trace()
                if d == 0:
                    Y = np.copy(wheelSpeedDict[recsForTraining[0]][1:][:,timeMaskTrain])
                    Ytest = np.copy(wheelSpeedDict[recsForTest[0]][1:][:,timeMaskTest])
                    YtestTime = np.copy(wheelSpeedDict[recsForTest[0]][0][timeMaskTest])
                elif (d>0) and (d<5):
                    pawId = d-1
                    Y = np.copy(pawTracksDict[recsForTraining[0]][pawId][1:][:,timeMaskTrain])
                    Ytest = np.copy(pawTracksDict[recsForTest[0]][pawId][1:][:,timeMaskTest])
                    YtestTime = np.copy(pawTracksDict[recsForTest[0]][pawId][0][timeMaskTest])
                elif d==5: # case where all four paw speeds are added together
                    pawSpeedTrain = []
                    pawSpeedTest = []
                    for i in range(4):
                        pawSpeedTrain.append(np.copy(pawTracksDict[recsForTraining[0]][i][1:][:, timeMaskTrain]))
                        pawSpeedTest.append(np.copy(pawTracksDict[recsForTest[0]][i][1:][:, timeMaskTest]))
                    Y = pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                    Ytest = pawSpeedTest[0] + pawSpeedTest[1] + pawSpeedTest[2] + pawSpeedTest[3]
                    #pdb.set_trace()
                for t in recsForTraining[1:]:
                    if borders is not None:
                        timeMaskTrain = (caTracesDict[t][0] >= borders[0]) & (caTracesDict[t][0] <= borders[1])
                    else:
                        timeMaskTrain = (caTracesDict[t][0] >= 0) & (caTracesDict[t][0] <= 1000.)
                    X = np.column_stack((X,caTracesDict[t][1:][:,timeMaskTrain]))
                    if d == 0:
                        Y = np.column_stack((Y,wheelSpeedDict[t][1:][:,timeMaskTrain]))
                    elif (d>0) and (d<5):
                        Y = np.column_stack((Y, pawTracksDict[t][pawId][1:][:,timeMaskTrain]))
                    elif d==5:
                        pawSpeedTrain = []
                        for i in range(4):
                            pawSpeedTrain.append(np.copy(pawTracksDict[t][i][1:][:, timeMaskTrain]))
                        speedTemp =  pawSpeedTrain[0] + pawSpeedTrain[1] + pawSpeedTrain[2] + pawSpeedTrain[3]
                        Y = np.column_stack((Y, speedTemp))
                Y = Y[0]
                X = np.transpose(X)
                Ytest = Ytest[0]
                Xtest = np.transpose(Xtest)
                # linear regression  ########################################
                linReg = LinearRegression()
                linReg.fit(X,Y)
                #svm_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
                #svm_rbf.fit(X,Y)
                YTrainPred = linReg.predict(X)
                YTestPred  = linReg.predict(Xtest)
                R2trainLR = linReg.score(X, Y)
                R2testLR = linReg.score(Xtest, Ytest) # 1. - np.sum((Ytest-YTestPred)**2)/np.sum((Ytest - np.mean(Ytest))**2)#linReg.score(Xtest, Ytest)
                #print(linReg.coef_)
                #print(linReg.intercept_)
                #yPred = linReg.predict(np.transpose(X))
                # random forest ##############################################
                #randForestReg = RandomForestRegressor(n_estimators=20)
                #randForestReg.fit(X, Y)
                #R2trainRF = randForestReg.score(X, Y)
                #R2testRF= randForestReg.score(Xtest, Ytest)
                #
                # print('R2 test :',R2testLR)
                # fig = plt.figure()
                # ax = fig.add_subplot(111)
                # ax.plot(YtestTime,Ytest,lw=2)
                # ax.plot(YtestTime,YTestPred,lw=2)
                # ax.spines['top'].set_visible(False)
                # ax.spines['right'].set_visible(False)
                # ax.spines['bottom'].set_position(('outward', 10))
                # ax.spines['left'].set_position(('outward', 10))
                # ax.yaxis.set_ticks_position('left')
                # ax.xaxis.set_ticks_position('bottom')
                # plt.show()

                Rval1.extend([R2trainLR,R2testLR])
            RTempValues.append(Rval1)
        #pdb.set_trace()
        Rs = np.zeros(regressionN*2)
        for reg in range(5):
            Rs += RTempValues[reg]
        Rs /=5.
        Rvalues.append(Rs)

    return Rvalues
#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateStepTriggeredCaTraces(mouse,allCorrDataPerSession,allStepData):
    matplotlib.use('TkAgg')
    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    timeAxis = np.linspace(-0.4,0.6,int((0.4+0.6)/0.02)+1)
    timeAxisRescaled = np.linspace(-1.,2.,int((1.+2.)/0.02)+1)
    preStanceMask = timeAxis<-0.1
    preStanceRescaledMask = timeAxisRescaled<-0.2
    K = len(timeAxis)
    KRescaled = len(timeAxisRescaled)
    caTraces = []
    for nDay in range(len(allCorrDataPerSession)):
        print(allCorrDataPerSession[nDay][0],allStepData[nDay][0],nDay)
        #print(allCorrDataPerSession[nDay][0],nDay)
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession,allStepData)
        #if len(allStepData[nDay-1][4])==6:
        #    print('more recordings :',len(allStepData[nDay][4]))
        #    addIdx = 1
        #else:
        #    addIdx = 0
        #pdb.set_trace()
        N = len(caTracesDict[0][1:])
        caSnippets = [[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)]] #np.zeros((4,N,K))
        caSnippetsRescaled = [[[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)],[[] for i in range(N)]]

        NRecs=len(allStepData[nDay][4])
        for nrec in range(NRecs): # loop over the five recordings of a day
            for i in range(4): # loop over the four paws
                #pdb.set_trace()
                idxSwings = allStepData[nDay][4][nrec][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay][4][nrec][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                for k in range(len(idxSwings)): # loop over all swings
                    startSwingTime = recTimes[idxSwings[k, 0]]
                    endSwingTime = recTimes[idxSwings[k, 1]]
                    if len(caTracesDict[nrec][1:])!=N: print('problem in number of ROIs')
                    for l in range(len(caTracesDict[nrec][1:])): # loop over all ROIs
                        interpCa = interp1d(caTracesDict[nrec][0]-startSwingTime, caTracesDict[nrec][l+1])#,kind='cubic')
                        interpCaRescaled = interp1d((caTracesDict[nrec][0]-startSwingTime)/(endSwingTime-startSwingTime), caTracesDict[nrec][l+1])#,kind='cubic')
                        ############
                        try:
                            newCaTraceAtSwing = interpCa(timeAxis)
                        except ValueError:
                            pass
                        else:
                            #caSnippets[i,l,:] += newCaTraceAtSwing
                            caSnippets[i][l].append(newCaTraceAtSwing)
                        ############
                        try:
                            newCaTraceAtSwingRescaled = interpCaRescaled(timeAxisRescaled)
                        except ValueError:
                            #print('error')
                            pass
                        else:
                            #caSnippets[i,l,:] += newCaTraceAtSwing
                            caSnippetsRescaled[i][l].append(newCaTraceAtSwingRescaled)
        #pdb.set_trace()
        caSnippetsArray = np.zeros((4, N, 2, K))
        caSnippetsRescaledArray = np.zeros((4, N, 2, KRescaled))
        for i in range(4): # loop over four paws
            for l in range(N): # loop over all ROIs
                caTempArray = np.asarray(caSnippets[i][l])
                #pdb.set_trace()
                caSnippetsZscores = (caTempArray - np.mean(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
                caTemp = np.mean(caSnippetsZscores,axis=0)
                caTempSTD = np.std(caSnippetsZscores,axis=0)
                caSnippetsArray[i,l,0,:] = caTemp
                caSnippetsArray[i,l,1,:] = caTempSTD
                #pdb.set_trace()
                caTempRescaledArray = np.asarray(caSnippetsRescaled[i][l])
                #pdb.set_trace()
                caSnippetsRescaledZscores = (caTempRescaledArray - np.mean(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis])# /np.std(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis]
                caTempRe = np.mean(caSnippetsRescaledZscores,axis=0)
                caTempReSTD = np.std(caSnippetsRescaledZscores,axis=0)
                caSnippetsRescaledArray[i,l,0,:] = caTempRe
                caSnippetsRescaledArray[i,l,1,:] = caTempReSTD
        caTraces.append([allCorrDataPerSession[nDay][0],allStepData[nDay][0],nDay,timeAxis,caSnippetsArray,timeAxisRescaled,caSnippetsRescaledArray])

    return caTraces

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateStepTriggeredCaTracesAllPaws(mouse,allCorrDataPerSession,allStepData):
    maxSeparation = 0.# min separation between swings in sec

    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    timeAxis = np.linspace(-0.4,0.6,(0.6+0.4)/0.02+1)
    timeAxisRescaled = np.linspace(-1.,2.,(2+1)/0.02+1)
    preStanceMask = timeAxis<-0.1
    preStanceRescaledMask = timeAxisRescaled<-0.2
    K = len(timeAxis)
    KRescaled = len(timeAxisRescaled)
    caTraces = []
    swingT = []
    for nDay in range(len(allCorrDataPerSession)):
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession)
        if len(allStepData[nDay-1][4])==6:
            print('more recordings :',len(allStepData[nDay-1][4]))
            addIdx = 1
        else:
            addIdx = 0
        #pdb.set_trace()
        N = len(caTracesDict[0][1:])
        print(allCorrDataPerSession[nDay][0], allStepData[nDay - 1][0], nDay, N)
        caSnippets = [[] for i in range(N)] #np.zeros((4,N,K))
        caSnippetsRescaled = [[] for i in range(N)]
        caSnippetsArray = np.zeros((N,2,K))
        caSnippetsRescaledArray = np.zeros((N, 2, KRescaled))
        swingSnippets = [[] for i in range(5)]
        for nrec in range(5): # loop over the five recordings of a day
            swingTimes = np.zeros(3)
            for i in range(4): # loop over the four paws and lump all swing times together
                idxSwings = allStepData[nDay-1][4][nrec+addIdx][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay-1][4][nrec+addIdx][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                startSwingTimes = recTimes[idxSwings[:,0]]
                endSwingTimes = recTimes[idxSwings[:,1]]
                swingTimes = np.vstack((swingTimes,np.column_stack((startSwingTimes, endSwingTimes,np.repeat(i,len(startSwingTimes))))))
            swingTimes = swingTimes[1:] # remove first element which was zeros only
            # sort swing times according to swing start
            swingTimes = swingTimes[swingTimes[:,0].argsort()]
            # remove swings with fall within the minimum separation betweeen swings
            diffSwings = np.diff(swingTimes[:,0]) # calculate inter-swing intervals
            swingTimesSparse = swingTimes[np.concatenate((diffSwings>maxSeparation,np.array([True])))] # only use swings which fall above separation time
            swingSnippets[nrec] = swingTimes
            for k in range(len(swingTimesSparse)): # loop over all swings
                startSwingTime = swingTimesSparse[k, 0]
                endSwingTime = swingTimesSparse[k, 1]

                if len(caTracesDict[nrec][1:])!=N: print('problem in number of ROIs')
                for l in range(len(caTracesDict[nrec][1:])): # loop over all ROIs
                    interpCa = interp1d(caTracesDict[nrec][0]-startSwingTime, caTracesDict[nrec][l+1])#,kind='cubic')
                    interpCaRescaled = interp1d((caTracesDict[nrec][0]-startSwingTime)/(endSwingTime-startSwingTime), caTracesDict[nrec][l+1])#,kind='cubic')
                    ############
                    try:
                        newCaTraceAtSwing = interpCa(timeAxis)
                    except ValueError:
                        pass
                    else:
                        #caSnippets[i,l,:] += newCaTraceAtSwing
                        caSnippets[l].append(newCaTraceAtSwing)
                    ############
                    try:
                        newCaTraceAtSwingRescaled = interpCaRescaled(timeAxisRescaled)
                    except ValueError:
                        #print('error')
                        pass
                    else:
                        #caSnippets[i,l,:] += newCaTraceAtSwing
                        caSnippetsRescaled[l].append(newCaTraceAtSwingRescaled)
        #pdb.set_trace()
        #for i in range(4):
        for l in range(N):
            caTempArray = np.asarray(caSnippets[l])
            caSnippetsZscores = (caTempArray - np.mean(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
            caTemp = np.mean(caSnippetsZscores,axis=0)
            caTempSTD = np.std(caSnippetsZscores,axis=0)
            caSnippetsArray[l,0,:] = caTemp
            caSnippetsArray[l,1,:] = caTempSTD
            #
            caTempRescaledArray = np.asarray(caSnippetsRescaled[l])
            #pdb.set_trace()
            caSnippetsRescaledZscores = (caTempRescaledArray - np.mean(caTempRescaledArray[:,preStanceRescaledMask],axis=1)[:,np.newaxis]) #/np.std(caTempArray[:,preStanceMask],axis=1)[:,np.newaxis]
            caTempRe = np.mean(caSnippetsRescaledZscores,axis=0)
            caTempReSTD = np.std(caSnippetsRescaledZscores,axis=0)
            caSnippetsRescaledArray[l,0,:] = caTempRe
            caSnippetsRescaledArray[l,1,:] = caTempReSTD
        swingT.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,swingSnippets])
        caTraces.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,caSnippetsArray,caSnippetsRescaledArray])

    return caTraces


#################################################################################
def calcualteAllDeltaTValues(t1,t2):
    l1 = len(t1)
    l2 = len(t2)

    fannedOut1 = np.tile(t1,(l2,1))
    fannedOut2 = np.tile(t2,(l1,1))
    transposedFannedOut2 = np.transpose(fannedOut2)
    #print l1, l2
    #print shape(fannedOut1), shape(transposedFannedOut2)
    #pdb.set_trace()
    differences = fannedOut1 - transposedFannedOut2 # subtract(fannedOut1,transposedFannedOut2)
    return differences.flatten()

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def generateInterstepTimeHistogram(mouse,allCorrDataPerSession,allStepData):
    # check for sanity
    if len(allCorrDataPerSession) != len(allStepData):
        print('both dictionaries are not of the same length')
        print('CaWheelPawDict:',len(allCorrDataPerSession),' StepStanceDict:,',len(allStepData))

    #timeAxis = np.linspace(-0.4,0.6,(0.6+0.4)/0.02+1)
    #timeAxisRescaled = np.linspace(-1.,2.,(2+1)/0.02+1)
    #preStanceMask = timeAxis<-0.1
    #preStanceRescaledMask = timeAxisRescaled<-0.2
    #K = len(timeAxis)
    #KRescaled = len(timeAxisRescaled)
    pawSwingTimes = []
    for nDay in range(1,len(allCorrDataPerSession)):
        print(allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay)
        (wheelSpeedDictInterP, pawTracksDictInterP, caTracesDictInterP,wheelSpeedDict,pawTracksDict,caTracesDict) = getCaWheelPawInterpolatedDictsPerDay(nDay, allCorrDataPerSession)
        if len(allStepData[nDay-1][4])==6:
            print('more recordings :',len(allStepData[nDay-1][4]))
            addIdx = 1
        else:
            addIdx = 0
        #pdb.set_trace()
        #N = len(caTracesDict[0][1:])
        #caSnippets = [[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)],[[] for i in range(N)]] #np.zeros((4,N,K))
        #caSnippetsRescaled = [[[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)], [[] for i in range(N)]]
        #caSnippetsArray = np.zeros((4,N,2,K))
        #caSnippetsRescaledArray = np.zeros((4, N, 2, KRescaled))
        allData = []
        for nrec in range(5): # loop over the five recordings of a day
            startSwingTimes = [[] for i in range(4)]
            endSwingTimes = [[] for i in range(4)]
            for i in range(4): # loop over the four paws
                idxSwings = allStepData[nDay-1][4][nrec+addIdx][3][i][1]
                #print('Wow nSess',nDay,allStepData[nDay-1][0])
                recTimes = allStepData[nDay-1][4][nrec+addIdx][4][i][2]
                #pdb.set_trace()
                idxSwings = np.asarray(idxSwings)
                startSwingT = recTimes[idxSwings[:,0]]
                endSwingT = recTimes[idxSwings[:,1]]
                startSwingTimes[i].append(startSwingT)
                endSwingTimes[i].append(endSwingT)
            allData.append([startSwingTimes,endSwingTimes])
        interPawSwingTimes = [[] for i in range(4)]
        interStepTimes = []
        stepLengths = [[] for i in range(4)]
        #pdb.set_trace()
        for nrec in range(5):
            for i in range(4):
                interPawSwingTimes[i].extend(calcualteAllDeltaTValues(allData[nrec][0][i][0],allData[nrec][0][i][0]))
                stepLengths[i].extend(allData[nrec][1][i][0]-allData[nrec][0][i][0])
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0],allData[nrec][0][1][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0], allData[nrec][0][2][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][0][0], allData[nrec][0][3][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][1][0], allData[nrec][0][2][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][1][0], allData[nrec][0][3][0]))
            interStepTimes.extend(calcualteAllDeltaTValues(allData[nrec][0][2][0], allData[nrec][0][3][0]))
        #pdb.set_trace()
        pawSwingTimes.append([allCorrDataPerSession[nDay][0],allStepData[nDay-1][0],nDay,interPawSwingTimes,stepLengths,interStepTimes])

    return pawSwingTimes
#################################################################################
# remove empty columns and row - from the image registration routine
#################################################################################
def removeEmptyColumnAndRows(img):
    # hmask = np.invert(np.sum(img, axis=0) == 0) # looking for zeros does not work as the boundary values are not zeros all the time
    # vmask = np.invert(np.sum(img, axis=1) == 0)
    htemp = (img == img[0,:]) # look instead for same values in row
    hmask = np.invert(htemp.all(axis=0))
    vtemp = (img == img[:,0])
    vmask = np.invert(vtemp.all(axis=1))
    idxH = np.arange(len(hmask))[np.hstack((False,np.diff(hmask)>0))]
    idxV = np.arange(len(vmask))[np.hstack((False, np.diff(vmask)>0))]
    #pdb.set_trace()
    if len(idxV) == 0:
        idxV = np.array([0,np.shape(img)[0]])
    if len(idxH) == 0:
        idxH = np.array([0,np.shape(img)[1]])

    croppedImg = img[idxV[0]:idxV[1],idxH[0]:idxH[1]]

    cutLengths = np.vstack((idxV,idxH))
    #pdb.set_trace()
    return cutLengths


#################################################################################
# remove empty columns and row - from the image registration routine
#################################################################################
def alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,refDate,otherDate,movementValues,figSave=False,figDir=''):
    matplotlib.use('TkAgg')
    column1 = np.maximum(cutLengthsA[:,0],cutLengthsB[:,0])
    column2 = np.minimum(cutLengthsA[:,1],cutLengthsB[:,1])
    cutLenghts = np.column_stack((column1,column2))

    imgA = imgA[cutLenghts[0,0]:cutLenghts[0,1],cutLenghts[1,0]:cutLenghts[1,1]]
    imgB = imgB[cutLenghts[0,0]:cutLenghts[0,1],cutLenghts[1,0]:cutLenghts[1,1]]
    # Find size of ref image
    sz = imgA.shape

    corr = signal.correlate(imgA - imgA.mean(), imgB - imgB.mean(), mode='same', method='fft')
    maxIdx = np.unravel_index(np.argmax(corr, axis=None), corr.shape)
    shifty = np.shape(imgA)[0]/2. - maxIdx[0]
    shiftx = np.shape(imgA)[1]/2. - maxIdx[1]
    print('max of cross-correlation : ', shiftx, shifty )
    #pdb.set_trace()
    # Define the motion model
    #warp_mode = cv2.MOTION_TRANSLATION  #cv2.MOTION_EUCLIDEAN # cv2.MOTION_TRANSLATION  # MOTION_EUCLIDEAN
    warp_mode = cv2.MOTION_AFFINE #EUCLIDEAN #HOMOGRAPHY
    warp_modes = [cv2.MOTION_AFFINE,cv2.MOTION_EUCLIDEAN,cv2.MOTION_TRANSLATION]

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 1000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    #try:
    imA_u8 = (((imgA-np.min(imgA))/(np.max(imgA)-np.min(imgA)))*255).astype(np.uint8) #cv2.cvtColor(imgA,cv2.COLOR_BGR2GRAY)
    imB_u8 = (((imgB-np.min(imgB))/(np.max(imgB)-np.min(imgB)))*255).astype(np.uint8)
    #pdb.set_trace()
    warpResults = []
    corrMax = []
    for w in range(len(warp_modes)):
        print('testing ', warp_modes[w])
        # warp_matrix1[0, 2] = aS.xOffset
        # warp_matrix1[1, 2] = aS.yOffset
        if (movementValues[0] != 0) and (movementValues[1] != 0):
            warp_matrix[0, 2] = movementValues[0]  # -20.
            warp_matrix[1, 2] = movementValues[1]  # -40.
        else:
            warp_matrix[0, 2] = shiftx  # -20.
            warp_matrix[1, 2] = shifty  # -40.
        #try:
        #    (cc, warp_matrixRet) = cv2.findTransformECC(imA_u8, imB_u8, warp_matrix, warp_modes[w], criteria, inputMask = None, gaussFiltSize=5)
        #except TypeError:
        try :
            (cc, warp_matrixRet) = cv2.findTransformECC(imA_u8, imB_u8, warp_matrix, warp_modes[w], criteria, inputMask = None)
        except:
            print('findTransformECC did not converge')
            cc = -1
            warp_matrixRet = warp_matrix
        #print(warp_matrixRet,warp_matrix)
        warpResults.append([w,warp_modes[w],np.copy(warp_matrixRet),np.copy(cc)])
        corrMax.append(cc)
        #if cc>0.8:
        #    break
    print(warpResults)
    corrMax = np.asarray(corrMax)
    maxCorr = np.argmax(corrMax)
    cc_max = warpResults[maxCorr][3]
    warp_matrix_max = warpResults[maxCorr][2]
    #except:
    #print('findTransformECC output : ',cc,warp_matrixRet)
    #print('find image transformation did not converge')
    #warp_matrixRet = np.copy(warp_matrix_max)
    #cc = None
    #else:
        #pass
    #(cc2, warp_matrix2Ret) = cv2.findTransformECC(imBD, im820, warp_matrix2, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        imgB_aligned = cv2.warpPerspective(imgB, warp_matrix_max, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        imgB_aligned = cv2.warpAffine(imgB, warp_matrix_max, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP);

    print('result of image alignment-> warp-matrix  and correlation coefficient : ', warp_matrix_max, cc_max)

    if figSave :
        ##################################################################
        # Show final results
        # figure #################################
        fig_width = 10  # width in inches
        fig_height = 10  # height in inches
        fig_size = [fig_width, fig_height]
        params = {'axes.labelsize': 11, 'axes.titlesize': 11, 'font.size': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'figure.figsize': fig_size, 'savefig.dpi': 600,
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
        gs = gridspec.GridSpec(2, 2  # ,
                               # width_ratios=[1.2,1]
                               # height_ratios=[1,1]
                               )

        # define vertical and horizontal spacing between panels
        gs.update(wspace=0.3, hspace=0.3)

        # possibly change outer margins of the figure
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.06)

        # sub-panel enumerations
        # plt.figtext(0.06, 0.92, 'A',clip_on=False,color='black', weight='bold',size=22)

        # first sub-plot #######################################################
        # gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0],hspace=0.2)
        # ax0 = plt.subplot(gssub[0])

        # fig = plt.figure(figsize=(10,10))

        #plt.figtext(0.1, 0.95, '%s ' % (aS.animalID), clip_on=False, color='black', size=14)

        ax0 = plt.subplot(gs[0])
        ax0.set_title('reference image %s' % refDate)
        ax0.imshow(imgA)

        ax0 = plt.subplot(gs[1])
        ax0.set_title('to-be-aligned image %s' % otherDate )
        ax0.imshow(imgB)

        ax0 = plt.subplot(gs[2])
        ax0.set_title('overlay of both images')
        overlayBefore = cv2.addWeighted(imgA/np.max(imgA), 1, imgB/np.max(imgB), 1, 0)
        ax0.imshow(overlayBefore)

        ax0 = plt.subplot(gs[3])
        ax0.set_title('overlay after alignement c = %s \nof BD-AD images' % np.round(cc_max,4), fontsize=10)
        overlayAfter = cv2.addWeighted(imgA/np.max(imgA), 1, imgB_aligned/np.max(imgB_aligned), 1, 0)
        ax0.imshow(overlayAfter)

        #plt.show()
        plt.savefig(figDir + 'ImageAlignment_%s-%s.pdf' % (refDate,otherDate))  # plt.savefig(figOutDir+'ImageAlignment_%s.png' % aS.animalID)  # plt.show()
        plt.close()

    return (warp_matrix_max,cc_max)

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def alignROIsCheckOverlap(statRef,opsRef,statAlign,opsAlign,warp_matrix,refDate,otherDate,figSave=False,figDir=''):
    ncellsRef= len(statRef)
    ncellsAlign = len(statAlign)

    imMaskRef   = np.zeros((opsRef['Ly'], opsRef['Lx']))
    imMaskAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

    intersectionROIs = []
    intersectionROIsA = []
    for n in range(0,ncellsRef):
        imMaskRef[:] = 0
        #if iscellBD[n][0]==1:
        #pdb.set_trace()
        ypixRef = statRef[n]['ypix']
        xpixRef = statRef[n]['xpix']
        imMaskRef[ypixRef,xpixRef] = 1
        for m in range(0,ncellsAlign):
            imMaskAlign[:] = 0
            #if iscellAD[m][0]==1:
            ypixAl = statAlign[m]['ypix']
            xpixAl = statAlign[m]['xpix']
            # perform homographic transform : rotation + translation
            #pdb.set_trace()
            points = np.column_stack((xpixAl,ypixAl))
            newPoints = np.copy(points)
            #pdb.set_trace()
            warp_matrix_inverse = np.copy(warp_matrix)
            cv2.invertAffineTransform(warp_matrix,warp_matrix_inverse)
            #newPoints = cv2.transform(points,warp_matrix_inverse)
            xpixAlPrime = np.rint(xpixAl*warp_matrix_inverse[0,0] + ypixAl*warp_matrix_inverse[0,1] + warp_matrix_inverse[0,2])
            ypixAlPrime = np.rint(xpixAl*warp_matrix_inverse[1,0] + ypixAl*warp_matrix_inverse[1,1] + warp_matrix_inverse[1,2]) # - np.rint(warp_matrix[1,2])
            xpixAlPrime = np.array(xpixAlPrime,dtype=int)
            ypixAlPrime = np.array(ypixAlPrime,dtype=int)
            #pdb.set_trace()
            # make sure pixels remain within
            xpixAlPrime2 = xpixAlPrime[(xpixAlPrime<opsAlign['Lx'])&(ypixAlPrime<opsAlign['Ly'])]
            ypixAlPrime2 = ypixAlPrime[(xpixAlPrime<opsAlign['Lx'])&(ypixAlPrime<opsAlign['Ly'])]
            imMaskAlign[ypixAlPrime2,xpixAlPrime2] = 1
            #imMaskAlign[xpixAlPrime2,ypixAlPrime2] = 1
            intersection = np.sum(np.logical_and(imMaskRef,imMaskAlign))
            eitherOr = np.sum(np.logical_or(imMaskRef,imMaskAlign))
            if intersection>0.2:
                #print(n,m,intersection,eitherOr,intersection/eitherOr)
                intersectionROIs.append([n,m,xpixRef,ypixRef,xpixAlPrime2,ypixAlPrime2,intersection,eitherOr,intersection/eitherOr])
                intersectionROIsA.append([n,m,intersection,eitherOr,intersection/eitherOr])
    # clean up intersection ROIs; each ROI should only overlap once
    def removeDoubleCellOccurrences(interROIs,column):
        uniquePerColumn = np.unique(interROIs[:,column],return_counts=True) # find unique occurrences
        multipleCells = uniquePerColumn[0][uniquePerColumn[1]>1]  # which cells occur more than once in the first column
        indiciesToRemove = []
        for i in multipleCells:
            indicies = np.argwhere(interROIs[:,column]==i)
            maxIdx = np.argmax(interROIs[indicies[:,0]][:,4])
            delIndicies = np.delete(indicies,maxIdx)
            indiciesToRemove.extend(delIndicies)
        return indiciesToRemove
    intersectionROIsA = np.asarray(intersectionROIsA)
    removeIdicies0 = removeDoubleCellOccurrences(intersectionROIsA,0)
    removeIdicies1 = removeDoubleCellOccurrences(intersectionROIsA,1)
    removeIndicies = np.asarray(removeIdicies0 + removeIdicies1)
    removeIndicies = np.unique(removeIndicies)
    cleanedIntersectionROIs = []
    for i in range(len(intersectionROIs)):
        if i not in removeIndicies:
            cleanedIntersectionROIs.append(intersectionROIs[i])
    #pdb.set_trace()
    intersectionROIsA = np.delete(intersectionROIsA,removeIndicies,axis=0)
    #pdb.set_trace()
    if figSave:
        imRef = opsRef['meanImg']
        imAlign = opsAlign['meanImg']
        ##################################################################
        # Show final results
        fig = plt.figure(figsize=(15, 15))  ########################

        plt.figtext(0.1, 0.95, '%s and %s' % (refDate,otherDate), clip_on=False, color='black', size=14)

        ax0 = fig.add_subplot(3, 2, 1)  #############################
        ax0.set_title('reference img')
        ax0.imshow(imRef)

        ax0 = fig.add_subplot(3, 2, 2)  #############################
        ax0.set_title('image to be aligned')
        ax0.imshow(imAlign)


        ax0 = fig.add_subplot(3, 2, 3)  #############################
        ax0.set_title('ROIs in reference image')
        imRef = np.zeros((opsRef['Ly'], opsRef['Lx']))
        imRefB = np.zeros((opsRef['Ly'], opsRef['Lx']))

        for n in range(0, ncellsRef):
            ypixR = statRef[n]['ypix']
            xpixR = statRef[n]['xpix']
            imRef[ypixR, xpixR] = n + 1
            imRefB[ypixR, xpixR] = 1

        ax0.imshow(imRef, cmap='gist_ncar')

        ax0 = fig.add_subplot(3, 2, 4)  #############################
        ax0.set_title('ROIs in aligned image')
        imAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))
        imAlignB = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

        for n in range(0, ncellsAlign):
            ypixA = statAlign[n]['ypix']
            xpixA = statAlign[n]['xpix']
            imAlign[ypixA, xpixA] = n + 1
            imAlignB[ypixA, xpixA] = 2

        ax0.imshow(imAlign, cmap='gist_ncar')


        ax0 = fig.add_subplot(3, 2, 5)  #############################
        ax0.set_title('overlapping ROIs Ref-Aligned')
        imRef = np.zeros((opsRef['Ly'], opsRef['Lx']))
        imAlign = np.zeros((opsAlign['Ly'], opsAlign['Lx']))

        for n in range(0, len(cleanedIntersectionROIs)):
            ypixR = cleanedIntersectionROIs[n][3]
            xpixR = cleanedIntersectionROIs[n][2]
            ypixA = cleanedIntersectionROIs[n][5]
            xpixA = cleanedIntersectionROIs[n][4]
            imRef[ypixR, xpixR] = 1
            imAlign[ypixA, xpixA] = 2

        overlayBothROIs1 = cv2.addWeighted(imRef, 1, imAlign, 1, 0)
        #overlayBothROIs1B = cv2.addWeighted(imRefB, 1, imAlignB, 1, 0)
        ax0.imshow(overlayBothROIs1)

        ax0 = fig.add_subplot(3, 2, 6)  #############################
        ax0.set_title('fraction of ROI overlap Ref-Aligned')
        interFractions1 = []
        for n in range(0, len(cleanedIntersectionROIs)):
            interFractions1.append(cleanedIntersectionROIs[n][8])

        ax0.hist(interFractions1, bins=15)

        plt.savefig(figDir + 'ROIalignment_%s-%s.pdf' % (refDate, otherDate))
        #plt.show()
        plt.close()

    return (cleanedIntersectionROIs,intersectionROIsA)
    #pickle.dump(intersectionROIs, open( dataOutDir + 'ROIintersections_%s.p' % aS.animalID, 'wb' ) )

#################################################################################
# find ROI recorded on ref day and on any other given day
#################################################################################
def findMatchingRois(mouse,allCorrDataPerSession,analysisLocation,refDate=0):
    # check for sanity
    nDays = len(allCorrDataPerSession)

    refDay = allCorrDataPerSession[refDate][0]
    print('fluo images will be aligned to recordings of :', refDay)
    refDayCaData = allCorrDataPerSession[refDate][3][0]
    refImg = refDayCaData[2]['meanImgE']
    refImgCutLengths = removeEmptyColumnAndRows(refImg)
    opsRef = refDayCaData[2]
    statRef = refDayCaData[4]

    # create list of recoridng day indicies
    recDaysList = [i for i in range(nDays)]
    movementValuesPreset = np.zeros((len(recDaysList), 2))
    # movementValuesPreset[0] = np.array([-20,-47])
    movementValuesPreset[1] = np.array([143, 153])
    # movementValuesPreset[3] = np.array([-1,15])

    # remove day used for referencing
    recDaysList.remove(refDate)
    if os.path.exists(analysisLocation+'/alignmentData.p'):
        allDataRead = pickle.load(open(analysisLocation+'/alignmentData.p'))
    else:
        allDataRead = None
    allData = []
    for nDay in recDaysList:
        print(allCorrDataPerSession[nDay][0],nDay)
        #imgE = allCorrDataPerSession[nDay][3][0][2]['meanImgE']
        img = allCorrDataPerSession[nDay][3][0][2]['meanImgE']
        cutLengths = removeEmptyColumnAndRows(img)
        if allDataRead is not None:
            warp_matrix = allDataRead[nDay][3]
        else:
            (warp_matrix,cc) = alignTwoImages(refImg,refImgCutLengths,img,cutLengths,allCorrDataPerSession[refDate][0],allCorrDataPerSession[nDay][0],movementValuesPreset[nDay],figShow=True,)
        opsAlign  = allCorrDataPerSession[nDay][3][0][2]
        statAlign = allCorrDataPerSession[nDay][3][0][4]
        (cleanedIntersectionROIs,intersectionROIsA) = alignROIsCheckOverlap(statRef,opsRef,statAlign,opsAlign,warp_matrix,allCorrDataPerSession[refDate][0],allCorrDataPerSession[nDay][0],showFig=True)
        print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statRef), len(statAlign), len(cleanedIntersectionROIs))
        allData.append([allCorrDataPerSession[nDay][0],nDay,cutLengths,warp_matrix,cc,cleanedIntersectionROIs,intersectionROIsA])

    intersectingCellsInRefRecording = np.arange(len(statRef))
    for nDay in recDaysList:
        intersectingCellsInRefRecording = np.intersect1d(intersectingCellsInRefRecording,allData[nDay][5][:,0])
        print(nDay,allCorrDataPerSession[nDay][0],intersectingCellsInRefRecording)

    pdb.set_trace()

    return 0


#################################################################################
# correlates mean fluo images recorded all possible recording day combinations
#################################################################################
def findOverlayMatchingRoisAllDayCombinations(mouse, allCorrDataPerSession, analysisLocation, expDate, figLocation, allDataRead=None,saveFigure=True):
    nDays = len(allCorrDataPerSession)
    movementValuesPreset = np.zeros((nDays*nDays, 2))
    allCorrData = []
    #corrMatrix = np.zeros((nDays,nDays))
    nPair = 0
    for nDayA in range(nDays - 1):
        for nDayB in range(nDayA+1,nDays):
            print(nDayA,nDayB, allCorrDataPerSession[nDayA][0], allCorrDataPerSession[nDayB][0])

            imgA = allCorrDataPerSession[nDayA][3][0][2]['meanImg']
            cutLengthsA = removeEmptyColumnAndRows(imgA)
            opsA = allCorrDataPerSession[nDayA][3][0][2]
            statA = allCorrDataPerSession[nDayA][3][0][4]

            imgB = allCorrDataPerSession[nDayB][3][0][2]['meanImg']
            cutLengthsB = removeEmptyColumnAndRows(imgB)
            opsB = allCorrDataPerSession[nDayB][3][0][2]
            statB = allCorrDataPerSession[nDayB][3][0][4]

            if (allDataRead is not None) and (allDataRead[nPair][0]==allCorrDataPerSession[nDayA][0]) and (allDataRead[nPair][1]==allCorrDataPerSession[nDayB][0]):
                print('warp_matrix for current pair of recordings exists and will be used')
                warp_matrix = allDataRead[nPair][6]
                cc = allDataRead[nPair][7]
            else:
                (warp_matrix,cc) = alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],movementValuesPreset[nPair],figSave=saveFigure,figDir=figLocation)
            #corrMatrix[nDayA,nDayB] = cc
            (cleanedIntersectionROIs, intersectionROIsA) = alignROIsCheckOverlap(statA, opsA, statB, opsB, warp_matrix, allCorrDataPerSession[nDayA][0], allCorrDataPerSession[nDayB][0],figSave=saveFigure, figDir=figLocation)
            print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statA), len(statB), len(cleanedIntersectionROIs))
            allCorrData.append([allCorrDataPerSession[nDayA][0], allCorrDataPerSession[nDayB][0], nDayA, nDayB, cutLengthsA, cutLengthsB, warp_matrix, cc,cleanedIntersectionROIs, intersectionROIsA,statA,statB])
            nPair+=1
    return allCorrData

#################################################################################
# find ROIs recorded across successive recording days
#################################################################################
def findMatchingRoisSuccessivDays(mouse,allCorrDataPerSession,analysisLocation,expDate,figLocation, allDataRead=None):
    # check for sanity
    nDays = len(allCorrDataPerSession)

    # create list of recoridng day indicies
    #recDaysList = [i for i in range(nDays)]
    movementValuesPreset = np.zeros((nDays, 2))
    # movementValuesPreset[0] = np.array([-20,-47])
    # movementValuesPreset[1] = np.array([143, 153])
    # movementValuesPreset[3] = np.array([-1,15])

    allDataStore = []
    for nPair in range(nDays-1):
        nDayA = nPair
        nDayB = nPair + 1
        print(nPair, allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0])

        imgA = allCorrDataPerSession[nDayA][3][0][2]['meanImg']
        cutLengthsA = removeEmptyColumnAndRows(imgA)
        opsA = allCorrDataPerSession[nDayA][3][0][2]
        statA = allCorrDataPerSession[nDayA][3][0][4]

        imgB = allCorrDataPerSession[nDayB][3][0][2]['meanImg']
        cutLengthsB = removeEmptyColumnAndRows(imgB)
        opsB = allCorrDataPerSession[nDayB][3][0][2]
        statB = allCorrDataPerSession[nDayB][3][0][4]

        if (allDataRead is not None) and (allDataRead[nPair][0]==allCorrDataPerSession[nDayA][0]) and (allDataRead[nPair][1]==allCorrDataPerSession[nDayB][0]):
            print('warp_matrix for current pair of recordings exists and will be used')
            warp_matrix = allDataRead[nPair][6]
            cc = allDataRead[nPair][7]
        else:
            (warp_matrix,cc) = alignTwoImages(imgA,cutLengthsA,imgB,cutLengthsB,allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],movementValuesPreset[nPair],figShow=True,figDir=figLocation)

        (cleanedIntersectionROIs,intersectionROIsA) = alignROIsCheckOverlap(statA,opsA,statB,opsB,warp_matrix,allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],showFig=True,figDir=figLocation)
        print('Number of ROIs in Ref and aligned images, intersection ROIs :', len(statA), len(statB), len(cleanedIntersectionROIs))
        allDataStore.append([allCorrDataPerSession[nDayA][0],allCorrDataPerSession[nDayB][0],nDayA,nDayB,cutLengthsA,cutLengthsB,warp_matrix,cc,cleanedIntersectionROIs,intersectionROIsA])

    #intersectingCellsInRefRecording = np.arange(len(statA))
    #for nDay in recDaysList:
    #    intersectingCellsInRefRecording = np.intersect1d(intersectingCellsInRefRecording,allData[nDay][5][:,0])
    #    print(nDay,allCorrDataPerSession[nDay][0],intersectingCellsInRefRecording)
    #pdb.set_trace()
    return allDataStore

#################################################################################
# check which ROIs were recoreded across recordings days
#################################################################################
def roisRecordedAllDays(allData):
    def getMatchingPairs(aD):
        ROIpairs = []
        for i in range(len(aD)):
            ROIpairs.append(aD[i][:2])
        ROIpairs = np.asarray(ROIpairs)
        return ROIpairs

    #pdb.set_trace()
    Npairs = len(allData)
    dat  = []
    matchingRoisBefore = getMatchingPairs(allData[0][8])
    for n in range(Npairs-1):
        print(allData[n][0],allData[n][1],allData[n+1][0],allData[n+1][1],allData[n][7],allData[n+1][7])
        data = allData[n]
        matchingRoisAfter = getMatchingPairs(allData[n+1][8])
        idxRemaining = np.intersect1d(matchingRoisBefore[:,1],matchingRoisAfter[:,0])

        idxRemainingBefore = [key for key,val in enumerate(matchingRoisBefore[:,1]) if val in idxRemaining]
        idxRemainingAfter = [key for key,val in enumerate(matchingRoisAfter[:,0]) if val in idxRemaining]
        BeforeAlsoAfter = matchingRoisBefore[idxRemainingBefore]
        AfterAlsoBefore = matchingRoisAfter[idxRemainingAfter]

        dat.append([n,matchingRoisBefore,matchingRoisAfter,idxRemaining,BeforeAlsoAfter,AfterAlsoBefore])
        print('ROIS remaining before and after : ', len(idxRemaining))
        matchingRoisBefore = np.copy(AfterAlsoBefore)
        #pdb.set_trace()

    pdb.set_trace()

#################################################################################
# calculate correlations between ca-imaging, wheel speed and paw speed
#################################################################################
def doCorrelationAnalysis(mouse,allCorrDataPerSession):
    #
    xPixToUm = 0.79
    yPixToUm = 0.8
    #itertools.combinations(arr,2)
    sessionCorrelations = []
    varExplained = []

    for nSess in range(len(allCorrDataPerSession)):
        # calculate correcorrelation between individual calcium traces
        fTraces = allCorrDataPerSession[nSess][3][0][0]
        stat = allCorrDataPerSession[nSess][3][0][4]
        #pdb.set_trace()
        #frameNumbers = [0] + frameNumbers
        combis = list(itertools.combinations(np.arange(len(fTraces)), 2))
        ppCaTraces = []

        for i in range(len(combis)):
            #pdb.set_trace()
            xy0 = stat[combis[i][0]]['med']
            xy1 = stat[combis[i][1]]['med']

            euclDist = np.sqrt(((xy0[1]-xy1[1])*xPixToUm)**2 + ((xy0[0]-xy1[0])*yPixToUm)**2)
            xyDist = ([(xy1[1]-xy0[1])*xPixToUm,(xy0[0]-xy1[0])*yPixToUm])
            corrTemp = scipy.stats.pearsonr(fTraces[combis[i][0]],fTraces[combis[i][1]])
            ppCaTraces.append([i,combis[i][0],combis[i][1],corrTemp[0],corrTemp[1],euclDist,xyDist])

        # allCoord = []
        # for j in range(len(stat)):
        #     allCoord.append(stat[j]['med'])
        # aa = np.asarray(allCoord)
        # plt.scatter(aa[:,1],512-aa[:,0])
        # pdb.set_trace()
        ppCaTraces = np.asarray(ppCaTraces)
        # code taken from figure generation
        #for nSess in range(len(allCorrDataPerSession)):
        trialStartUnixTimes = []



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

        ###################################################################
        # correlation btw. PCA components and wheel, paw speeds
        #pdb.set_trace()
        pcaComponents = 4
        print('doing PCA ...')
        X = np.transpose(caAll)
        pca = PCA(n_components=pcaComponents)
        pca.fit(X)
        X_pca = pca.transform(X)
        #print(pca.components_)
        pcaCorrs = []
        varExplained.append(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_)
        for i in range(pcaComponents):
            corrWheelTemp = scipy.stats.pearsonr(X_pca[:,i], wheelAll)
            corrPaw0Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[0])
            corrPaw1Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[1])
            corrPaw2Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[2])
            corrPaw3Temp = scipy.stats.pearsonr(X_pca[:,i], pawAll[3])
            pcaCorrs.append([i,corrWheelTemp[0],corrWheelTemp[1],corrPaw0Temp[0],corrPaw0Temp[1],corrPaw1Temp[0],corrPaw1Temp[1],corrPaw2Temp[0],corrPaw2Temp[1],corrPaw3Temp[0],corrPaw3Temp[1]])


        ###################################################################
        sessionCorrelations.append([nSess,ppCaTraces,corrWheel,corrPaws,pcaCorrs])

    return (sessionCorrelations,varExplained)

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

    speedDiffThresh = 10  # cm/s Speed threshold, determine with variance
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
        newX  = (pawTracks[5][i][:,1][maskAngle])*0.025 + (newWheelAngleAtPawTimes*80./360.) #- (pawTracks[5][i][:,1][maskAngle][0])*0.025
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
            #pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameters])
            # (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5))
            rungLocs = rungMotion[3][n][3]
            xPaw = pawTracks[6][0] + pawTracks[0][n,(i*3+1)]
            yPaw = pawTracks[6][2] + pawTracks[0][n,(i*3+2)]
            distances = calculateDistanceBtwLineAndPoint(rungLocs[:,0],rungLocs[:,1],rungLocs[:,2],rungLocs[:,3],xPaw,yPaw)
            sortedArguments  = np.argsort(np.abs(distances))
            #closestRungIdx = np.argmin(np.abs(distances))
            closestRungNumber = rungMotion[3][n][2][sortedArguments[0]]
            closestDist = distances[sortedArguments[0]]
            secondClosestRungNumber = rungMotion[3][n][2][sortedArguments[1]]
            secondClosestDist = distances[sortedArguments[1]]
            rungInd.append([n,closestDist,sortedArguments[0],closestRungNumber,secondClosestDist,sortedArguments[1],secondClosestRungNumber,xPaw,yPaw])
            #pdb.set_trace()
        rungInd = np.asarray(rungInd)
        pawRungDistances.append([i,rungInd])

    ##############################################################################################################
    # determine regions during which the speed is different for more than xLength values #########################
    #stanceDistances = [[10, 40],[10,40],[-4,40],[-4,40]]
    stanceDistances = [[-4, 20], [-4, 20], [-20, 20], [-10, 20]]
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
                #pdb.set_trace()
                indecisiveStep = False
                closeIndicies = None
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
            fig = plt.figure(figsize=(27,7))
            ax = fig.add_subplot(1,3,1)
            ax.axvline(x=stanceDistances[i][0],color='0.6')
            ax.axvline(x=stanceDistances[i][1],color='0.6')
            ax.hist(pawRungDistances[i][1][:, 1],bins=100)
            plt.xlabel('paw rung distance (cm/s)')
            plt.ylabel('occurrence')
            #plt.show()

            ax1 = fig.add_subplot(1,3,2)
            ax2 = fig.add_subplot(1,3,3)
            ax1.fill_between(forFit[i][2],stanceDistances[i][0],stanceDistances[i][1],color='0.8')
            ax1.plot(forFit[i][2], forFit[i][0])
            ax1.plot(forFit[i][2], forFit[i][1] * p1)

            ax2.fill_between(forFit[i][2], stanceDistances[i][0], stanceDistances[i][1], color='0.8')
            ax2.plot(forFit[i][2], forFit[i][0])
            ax2.plot(forFit[i][2], forFit[i][1] * p1)
            # ax.plot(forFit[0][5][:,0],forFit[0][5][:,1])#+forFit[0][5][:,1][0])
            # ax.plot(forFit[1][5][:,0],forFit[1][5][:,1])#+forFit[1][5][:,1][0])
            # ax.plot(forFit[2][5][:,0],forFit[2][5][:,1])#+forFit[2][5][:,1][0])
            # ax.plot(forFit[3][5][:,0],forFit[3][5][:,1])#+forFit[3][5][:,1][0])
            uniqueRungIdx = np.unique(np.concatenate((pawRungDistances[i][1][:,3],pawRungDistances[i][1][:,6])))
            #c0 = np.append(pawRungDistances[i][1][:,3][1:],0)
            #c1 = np.append(pawRungDistances[i][1][:,6][1:],0)
            # for j in uniqueRungIdx:
            #     rMask1 = pawRungDistances[i][1][:,3] == j
            #     rMask2 = pawRungDistances[i][1][:,6] == j
            #     #ax.plot(pawRungDistances[i][1][:,0][rMask1],pawRungDistances[i][1][:,1][rMask1],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
            #     #ax.plot(pawRungDistances[i][1][:,0][rMask2],pawRungDistances[i][1][:,4][rMask2],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
            #     ax.plot(forFit[i][2][rMask1],pawRungDistances[i][1][:,1][rMask1],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
            #     ax.plot(forFit[i][2][rMask2],pawRungDistances[i][1][:,4][rMask2],'.',color=plt.cm.prism(j/np.max(uniqueRungIdx)))
            for n in range(len(cleanedSwingIndicies)):
                startI = int(cleanedSwingIndicies[n][0])
                endI   = int(cleanedSwingIndicies[n][1]) + 1
                ax1.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                ax2.fill_between(forFit[i][2][range(startI,endI)], 0, 1, color='0.5', alpha=0.5, transform=ax.get_xaxis_transform())
                #print(n,startI,endI,endI-startI,len(cleanedSwingIndicies))
                if stepCharacter[n][3]:
                    ax1.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='C2')
                    ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1, c='C2')
                else:
                    ax1.plot(forFit[i][2][range(startI,endI)],forFit[i][1][startI:endI] * p1,c='C2')
                    ax2.plot(forFit[i][2][range(startI, endI)], forFit[i][1][startI:endI] * p1, c='C2')
                #plt.fill_between()
                #plt.plot(range(startI,endI),forFit[i][1][startI:endI] * p1,c='C2')
                ax1.set_xlim(15,20)
                ax1.set_xlabel('time (s)')
                ax1.set_ylabel('speed (cm)')

            #plt.xlim(4610, 4720)
            #plt.savefig('')
            plt.show()


        swingPhases.append([i,cleanedSwingIndicies,stanceRungIdentity,stepCharacter])
    return (swingPhases,forFit)

