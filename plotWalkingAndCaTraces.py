import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

import numpy as np

mouse = '171218_f8'
expDate = '180123'

wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse,expDate)
(foldersRecordings,dataFolders)  = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)


for f in range(len(foldersRecordings)):
    tracks = []
    for r in range(20,21):#,len(foldersRecordings[f][1])):
        (WalkExistence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r],'RotaryEncoder')
        (CaExistence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r],'Imaging')
        if CaExistence and WalkExistence:
            # get walking activity
            (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][1][r],'walking_activity'])
            tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,foldersRecordings[f][1][r]])
            # get ca-activity
            (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1][r],'Imaging',fileHandle)  # read raw data from experiment
            (imStack,motionCoordinates,tifFile) = eSD.getMotioncorrectedStack(foldersRecordings[f][0],foldersRecordings[f][1][r],'moco') # read motion corrected image stack and displacement data
            eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
            (img,rois,rawSignals) = eSD.extractRoiSignals(foldersRecordings[f][0],foldersRecordings[f][1][r],tifFile) # determine/read rois, and get traces

            cV.generateWalkCaImage(foldersRecordings[f][0],foldersRecordings[f][1][r], np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo,motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor)  # plot fluorescent traces of rois
            cV.generateWalkCaCorrelationsImage(foldersRecordings[f][0],foldersRecordings[f][1][r], np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo,motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor)  # plot fluorescent traces of rois
            cV.generateWalkCaSpectralAnalysis(foldersRecordings[f][0],foldersRecordings[f][1][r], np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo,motionCoordinates,angluarSpeed,linearSpeed,sTimes,timeStamp,monitor)  # plot fluorescent traces of rois
            #break


del eSD

#cV.generateWalkingFigure(mouse,dataFolder,tracks)