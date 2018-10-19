import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

mouse = '180602_m78'

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse) # get recordings for specific mouse and date

#pdb.set_trace()

#dA      = dataAnalysis.dataAnalysis()
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'Imaging')
        if existence:
            (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'Imaging',fileHandle)
            eSD.saveImageStack(frames,fTimes,imageMetaInfo,[foldersRecordings[f][0],foldersRecordings[f][2][r],'raw_imaging_data'])
            eSD.saveTif(frames[:,:,:,0], mouse,foldersRecordings[f][0],foldersRecordings[f][2][r]) # tif file for possible image registration in ImageJ
        #break
    #break
