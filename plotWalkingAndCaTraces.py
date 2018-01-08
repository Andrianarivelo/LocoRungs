import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

import numpy as np

mouse = '170927_m68'
expDate = '171115'
wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

tracks = []
for rec in recordings:
    (WalkExistence, fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'RotaryEncoder')
    (CaExistence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'Imaging')
    if CaExistence and WalkExistence:
        # get walking activity
        (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor) = eSD.getWalkingActivity([dataFolder,rec,'walking_activity'])
        tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,rec])
        # get ca-activity
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'Imaging',fileHandle)  # read raw data from experiment
        (imStack,motionCoordinates,tifFile) = eSD.getMotioncorrectedStack(dataFolder,rec,'moco') # read motion corrected image stack and displacement data
        eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
        (img,rois,rawSignals) = eSD.extractRoiSignals(dataFolder,rec,tifFile) # determine/read rois, and get traces

        cV.generateWalkCaImage(dataFolder, rec, np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo,motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor)  # plot fluorescent traces of rois
        cV.generateWalkCaCorrelationsImage(dataFolder, rec, np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo,motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor)  # plot fluorescent traces of rois
        #break


del eSD

#cV.generateWalkingFigure(mouse,dataFolder,tracks)