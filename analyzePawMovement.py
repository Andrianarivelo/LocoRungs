import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb

mouse = '170927_m68'
expDate = '171115'
#mouse = '171126_m90'
#expDate = '180118'
#mouse = '171218_f8'
#expDate = '180123'

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)

for rec in recordings:
    (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'CameraGigEBehavior')
    if existence:
        cv2Tools.trackPawsAndRungs(mouse,dataFolder,rec)
