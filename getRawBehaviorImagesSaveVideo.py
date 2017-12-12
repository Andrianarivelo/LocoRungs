import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

mouse = '170927_m68'
expDate = '171115'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

#dA      = dataAnalysis.dataAnalysis()
for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraGigEBehavior')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'CameraGigEBehavior',fileHandle)
        eSD.saveBehaviorVideo(frames,fTimes,imageMetaInfo,'raw_behavior')
        #eSD.saveTif(frames, mouse, dataFolder, rec) # tif file for possible image registration in ImageJ