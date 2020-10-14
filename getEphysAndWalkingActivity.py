import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import numpy as np
import pdb
import tools.createVisualizations as createVisualizations
import scipy


mouse = '180201_f48'
expDate = '180314'

wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse,expDate)
(foldersRecordings,dataFolders) = eSD.getRecordingsList(mouse) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)
# loop over all recording folders
#pdb.set_trace()
for f in range(len(foldersRecordings)):
    print f, foldersRecordings[f][1]
    for r in range(len(foldersRecordings[f][1])):
        (existenceRot, fileHandleRot) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r], 'RotaryEncoder')
        (existenceEphys, fileHandleEphys) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r], 'AxoPatch200_2')
        #print existenceRot, existenceEphys
        #tracks = []
        if existenceRot and existenceEphys:
            (angles, aTimes,timeStamp,monitor) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1][r],'RotaryEncoder',fileHandleRot)
            (angularSpeed, linearSpeed, sTimes)  = dataAnalysis.getSpeed(angles,aTimes,wheelCircumsphere)
            #
            (current, ephysTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1][r], 'AxoPatch200_2', fileHandleEphys)
            #eSD.saveWalkingActivity(angularSpeed, linearSpeed, sTimes,timeStamp,monitor, [dataFolder,rec,'walking_activity'])  # save motion corrected image stack
            dt = np.mean(ephysTimes[1:]-ephysTimes[:-1])
            rate = 1./dt
            print 'rate', rate
            highpassfreq = 150.
            currentHP = dataAnalysis.butter_highpass(current,highpassfreq,rate,order=4)
            spikeT = dataAnalysis.detectSpikeTimes(2.E-11,currentHP,ephysTimes,positive=True)

            binWidth = 1.E-3 # in sec
            spikecountwindow = 0.05
            tbins =  np.linspace(0.,len(ephysTimes)*dt,int(len(ephysTimes)*dt/binWidth)+1)
            nspikecountwindow = spikecountwindow/binWidth
            np.save('currentHP.npy',np.column_stack((ephysTimes,currentHP)))
            np.save('tempScriptOutput/spikeTimes.npy', spikeT)

            binnedspikes, _ = np.histogram(spikeT, tbins)
            spikesconv = scipy.ndimage.filters.gaussian_filter1d(np.array(binnedspikes, float), sigma=nspikecountwindow)
            # convert the convolved spike trains to units of spikes/sec
            spikesconv *= 1. / binWidth

            cV.generateWalkEphysFigure(foldersRecordings[f][0], foldersRecordings[f][1][r], currentHP, ephysTimes , angularSpeed, linearSpeed, sTimes, timeStamp, monitor, spikesconv, binnedspikes, binWidth)  # plot fluorescent traces of rois

            pdb.set_trace()

del eSD



