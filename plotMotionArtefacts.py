import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import numpy as np
import pdb

#mouse = '170927_m68'
#expDate = '171115'
mouse = '171126_m90'
expDate = '180118'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

#dA      = dataAnalysis.dataAnalysis()
motion = []
for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraPixelfly')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'CameraPixelfly',fileHandle,readRawData=False)
        (imStack, motionCoordinates, tifFile) = eSD.getMotioncorrectedStack(dataFolder, rec,'moco')  # read motion corrected image stack and displacement data
        motion.append([rec,np.average(imStack, axis=0),motionCoordinates,imageMetaInfo,fTimes])

cV.generateMotionArtefactImage(dataFolder, motion)
