import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

#mouse = '170927_m68'
#expDate = '171115'
#mouse = '171126_m90'
#expDate = '180118'
mouse = '171218_f8'
expDate = '180123'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraGigEBehavior')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'CameraGigEBehavior',fileHandle)
        eSD.saveBehaviorVideo(mouse,dataFolder,rec,frames,fTimes,imageMetaInfo)