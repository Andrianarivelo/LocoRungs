import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

#mouse = '170927_m68'
#expDate = '171115'
#mouse = '171126_m90'
#expDate = '180118'
mouse = '180112_m33'
expDate = '180309'

eSD         = extractSaveData.extractSaveData(mouse,expDate)
(foldersRecordings,dataFolders) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

# loop over all recording folders
pdb.set_trace()
for f in range(len(foldersRecordings)):
    for r in range(4,len(foldersRecordings[f][1])):
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1][r],'CameraGigEBehavior')
        if existence:
            (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1][r],'CameraGigEBehavior',fileHandle)
            eSD.saveBehaviorVideo(mouse,foldersRecordings[f][0],foldersRecordings[f][1][r],frames,fTimes,imageMetaInfo)
            pdb.set_trace()