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

eSD         = extractSaveData.extractSaveData(mouse,expDate)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)

for f in range(len(recordings)) :
    for r in recordings[f][1]:
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(recordings[f][0],r,'CameraGigEBehavior')
        if existence:
            cv2Tools.trackPawsAndRungs(mouse,recordings[f][0],r)
