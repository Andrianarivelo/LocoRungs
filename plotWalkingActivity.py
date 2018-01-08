import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

mouse = '170927_m68'
expDate = '171115'
wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse)
(recordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

tracks = []
for rec in recordings:
    (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'RotaryEncoder')
    if existence:  #angularSpeed,linearSpeed,wTimes,startTime,monitor
        (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor) = eSD.getWalkingActivity([dataFolder,rec,'walking_activity'])
        tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,rec])
del eSD

cV.generateWalkingFigure(mouse,dataFolder,tracks)
