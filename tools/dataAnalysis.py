import time
import numpy as np
import sys
import scipy

def getSpeed(angles,times,circumsphere):
    angleJumps = angles[np.concatenate((([False]),np.diff(angles)!=0.))]
    timePoints = times[np.concatenate((([False]),np.diff(angles)!=0.))]
    angularSpeed = np.diff(angleJumps[::2])/np.diff(timePoints[::2])
    linearSpeed = angularSpeed*circumsphere/360.
    speedTimes = timePoints[::2][1:]
    return (angularSpeed,linearSpeed,speedTimes)


def crosscorr(deltat, y0, y1, correlationRange=1.5, fast=False):
    """
            home-written routine to calcualte cross-correlation between two contiuous traces
            new version from February 9th, 2011
    """

    if len(y0) != len(y1):
        print 'Data to be correlated has different dimensions!'
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
                print 'Problem!'
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
        print 'problem in length of spikeStart and spikeEnd'
        sys.exit(1)
    spikeT = []
    for i in range(len(spikeStart)):
        if (spikeEnd[i] - spikeStart[i]) > 10:  # ignore if difference between end and start is smaller than 15 points
            nMin = spikeStart[i] #np.argmin(eDataHP[spikeStart[i]:spikeEnd[i]]) + spikeStart[i]
            spikeT.append(nMin)
    # detectionTreshold = tresh
    spikeTimes = ephysTimes[np.array(spikeT,dtype=int)]
    return spikeTimes