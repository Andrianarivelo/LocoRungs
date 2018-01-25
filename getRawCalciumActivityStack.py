import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

mouse = '171218_f8'
expDate = '180123'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

#dA      = dataAnalysis.dataAnalysis()
for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'Imaging')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'Imaging',fileHandle)
        eSD.saveImageStack(frames,fTimes,imageMetaInfo,'raw_data')
        eSD.saveTif(frames[:,:,:,0], mouse, dataFolder, rec) # tif file for possible image registration in ImageJ

