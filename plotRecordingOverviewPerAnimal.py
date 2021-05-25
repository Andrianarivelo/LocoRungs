from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recs", help="specify index of the specify recording on that day", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import pdb, pickle, os

###########################################

#mouseD = '190101_f15' # id of the mouse to analyze
mouseD = '210120_m85'
expDateD = 'all910'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordingsD ='all910'     # 'all or 'some'

readDataAgain = True
wheelCircumsphere = 80.65 # in cm

###########################################

if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    try:
        expDate = expDateD
    except :
        expDate = 'all'
else:
    expDate = args.date

if args.recs == None:
    try:
        recordings = recordingsD
    except :
        recordings = 'all'
else:
    recordings = args.recs


eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)


if os.path.isfile(eSD.analysisLocation + '/allDataPerSession.p') and not readDataAgain:
    pass
    allDataPerSession = pickle.load( open( eSD.analysisLocation + '/allDataPerSession.p', 'rb' ) )
else:
    allDataPerSession = []
    for f in range(len(foldersRecordings)):
        tracks = []
        frames = []
        caImaging = []
        for r in range(len(foldersRecordings[f][2])): # loop over all trials
            # check for rotary encoder
            (rotaryExistence, rotFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
            if rotaryExistence:  #angularSpeed,linearSpeed,wTimes,startTime,monitor
                (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes,foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]])
            # check for video recording during trial
            (camExistence, camFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
            if camExistence:
                (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, startTime) = eSD.readBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo'])
                firstLastRecordedFrame = eSD.getBehaviorVideoFrames([foldersRecordings[f][0], foldersRecordings[f][2][r],'behaviorVideoFirstLastFrames'])
                #pdb.set_trace()
                frames.append([firstLastRecordedFrame, startEndExposureTime, startTime])
        # check for ca-imaging data during entire session
        (caImgExistence, tiffList,recLocation) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][0], 'SICaImaging')
        if caImgExistence:
            (nframes,meanImg,meanImgE,scanZoomFactor,timeStamps) =  eSD.getAnalyzedCaImagingData(eSD.analysisLocation+foldersRecordings[f][0],tiffList)
            caImaging.append([nframes,meanImg,meanImgE,scanZoomFactor,timeStamps])
        # combine all recordings from a session
        if (not tracks) and (not frames) and (not caImaging):
            pass
        else:
            allDataPerSession.append([foldersRecordings[f][0],tracks,frames,caImaging])

    pickle.dump( allDataPerSession, open( eSD.analysisLocation + '/allDataPerSession.p', 'wb' ) ) #eSD.analysisLocation,

# generate overview figure for animal
cV.generateOverviewFigure(mouse,allDataPerSession,wheelCircumsphere)





