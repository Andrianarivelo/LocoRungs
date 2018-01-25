import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

import numpy as np
import pdb


mouse = '171218_f8'
expDate = '180123'

eSD      = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate)

#dA      = dataAnalysis.dataAnalysis()
#print eSD.figureLocation
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

for rec in recordings[20:]:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'Imaging')  # check if specific data was recorded
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'Imaging',fileHandle)  # read raw data from experiment
        (imStack,motionCoordinates,tifFile) = eSD.getMotioncorrectedStack(dataFolder,rec,'moco') # read motion corrected image stack and displacement data
        eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
        (img,rois,rawSignals) = eSD.extractRoiSignals(dataFolder,rec,tifFile) # determine/read rois, and get traces

        cV.generateROIImage(dataFolder,rec,np.average(imStack,axis=0),fTimes,rois,rawSignals,imageMetaInfo,motionCoordinates) # plot fluorescent traces of rois
        #break

del eSD, cV
