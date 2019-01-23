import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

import numpy as np
import pdb


mouse = '180602_m78'
expDate = '180724'

eSD      = extractSaveData.extractSaveData(mouse,expDate)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate)

#dA      = dataAnalysis.dataAnalysis()
#print eSD.figureLocation
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r], 'Imaging')
        if existence:
            (frames, fTimes, imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r], 'Imaging', fileHandle)  # read raw data from experiment
            (imStack, motionCoordinates, tifFile) = eSD.getMotioncorrectedStack(foldersRecordings[f][0],foldersRecordings[f][2][r], 'moco')  # read motion corrected image stack and displacement data
            eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
            (img, rois, rawSignals) = eSD.extractRoiSignals(foldersRecordings[f][0],foldersRecordings[f][2][r], tifFile)
            cV.generateROIImage(foldersRecordings[f][0],foldersRecordings[f][2][r], np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo, motionCoordinates)

        #for rec in recordings[16:]:
        #(existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'Imaging')  # check if specific data was recorded

        #(frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'Imaging',fileHandle)  # read raw data from experiment
        #(imStack,motionCoordinates,tifFile) = eSD.getMotioncorrectedStack(dataFolder,rec,'moco') # read motion corrected image stack and displacement data
        #eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
        #(img,rois,rawSignals) = eSD.extractRoiSignals(dataFolder,rec,tifFile) # determine/read rois, and get traces

        #cV.generateROIImage(dataFolder,rec,np.average(imStack,axis=0),fTimes,rois,rawSignals,imageMetaInfo,motionCoordinates) # plot fluorescent traces of rois
        #break

del eSD, cV
