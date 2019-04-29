import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations

mouse = '190409_st1'
expDate = '190409'
wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)


for f in range(len(foldersRecordings)):
    tracks = []
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'RotaryEncoder')
        if existence:  #angularSpeed,linearSpeed,wTimes,startTime,monitor
            (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
            tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,foldersRecordings[f][2][r]])

    cV.generateWalkingFigure(mouse,foldersRecordings[f][0],tracks)
    del tracks

del eSD