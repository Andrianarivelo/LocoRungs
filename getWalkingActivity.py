import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis

mouse = '171218_f8'
expDate = '180123'

wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

#dA      = dataAnalysis.dataAnalysis()

tracks = []
for rec in recordings:
    (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'RotaryEncoder')
    if existence:
        (angles, aTimes,timeStamp,monitor) = eSD.readRawData(rec,'RotaryEncoder',fileHandle)
        (angularSpeed, linearSpeed, sTimes)  = dataAnalysis.getSpeed(angles,aTimes,wheelCircumsphere)
        eSD.saveWalkingActivity(angularSpeed, linearSpeed, sTimes,timeStamp,monitor, [dataFolder,rec,'walking_activity'])  # save motion corrected image stack

del eSD



