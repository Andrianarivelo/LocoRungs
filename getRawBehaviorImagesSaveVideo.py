import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

#mouse = '170927_m68'
#expDate = '171115'
#mouse = '171126_m90'
#expDate = '180118'
mouse = '180107_m27'
expDate = '180215'

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolders) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

pdb.set_trace()
for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraGigEBehavior')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'CameraGigEBehavior',fileHandle)
        eSD.saveBehaviorVideo(mouse,dataFolder,rec,frames,fTimes,imageMetaInfo)