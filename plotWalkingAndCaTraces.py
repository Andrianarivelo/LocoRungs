from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import pdb, pickle, os
import tools.parameters as pas

###########################################

mouseD = '190101_f15' # id of the mouse to analyze
#mouseD = '190108_m24'
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'

readDataAgain = False

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


eSD = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

if os.path.isfile(eSD.analysisLocation + '/allDataPerSession.p') and not readDataAgain:
    allCorrDataPerSession = pickle.load( open( eSD.analysisLocation + '/allCorrDataPerSession.p', 'rb' ) )
else:
    allCorrDataPerSession = []
    for f in range(len(foldersRecordings)):
        tracks = []
        pawTracks = []
        caImagingRois = []
        for r in range(len(foldersRecordings[f][2])): # loop over all trials
            # check for rotary encoder
            (rotaryExistence, rotFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
            if rotaryExistence:  #angularSpeed,linearSpeed,wTimes,startTime,monitor
                (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                tracks.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes,foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]])
            # check for video recording during trial
            (camExistence, camFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
            if camExistence :
                (rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r])
                #pdb.set_trace()
                pawTracks.append([rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime])
        # check for ca-imaging data during entire session
        (caImgExistence, tiffList) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][0], 'SICaImaging')
        if caImgExistence: # (Fluo,nRois,ops,frameNumbers)
            (Fluo,nRois,ops,timeStamps) =  eSD.getCaImagingRoiData(eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p/',tiffList)
            caImagingRois.append([Fluo,nRois,ops,timeStamps])
        # combine all recordings from a session
        if (not tracks) and (not pawTracks) and (not caImagingRois):
            pass
        else:
            allCorrDataPerSession.append([foldersRecordings[f][0],tracks,pawTracks,caImagingRois])

    pickle.dump(allCorrDataPerSession, open(eSD.analysisLocation + '/allCorrDataPerSession.p', 'wb'))  # eSD.analysisLocation,

#pdb.set_trace()
# generate overview figure for animal
correlationData = dataAnalysis.doCorrelationAnalysis(mouse,allCorrDataPerSession)
#pdb.set_trace()
cV.generateCaWheelPawImage(mouse,allCorrDataPerSession)
cV.generateCorrelationPlotsCaWheelPaw(mouse,correlationData,allCorrDataPerSession)

