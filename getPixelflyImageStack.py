import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

#mouse = '170927_m68'
#expDate = '171115'
mouse = '171126_m90'
expDate = '180118'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

#dA      = dataAnalysis.dataAnalysis()
for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraPixelfly')
    if existence:
        (frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'CameraPixelfly',fileHandle)
        eSD.saveImageStack(frames,fTimes,imageMetaInfo,'raw_pixelfly_data')
        eSD.saveTif(frames, mouse, dataFolder, rec) # tif file for possible image registration in ImageJ

