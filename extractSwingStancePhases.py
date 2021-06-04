from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recs", help="specify index of the specify recording on that day", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import pickle
import os
import pdb
import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy.stats as stats


mouseD = '210122_f84' # id of the mouse to analyze
expDateD = 'all910'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordingsD='all910'     # 'all or 'some'
DLCinstance = 'DLC_resnet_50_2021-Apr_PawExtraction_210122_f84Apr23shuffle2_200000'

readDataAgain = True

# in case mouse, and date were specified as input arguments
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
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings)  # get recordings for specific mouse and date
cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

#########################################################
# Get paws coordinates
if os.path.isfile(eSD.analysisLocation + '/allSingStanceDataPerSession.p') and not readDataAgain:
    recordingsM = pickle.load( open( eSD.analysisLocation + '/allSingStanceDataPerSession.p', 'rb' ) )
else:
    recordingsM = []
    for f in range(len(foldersRecordings)):
        # loop over all recordings in that folder
        tracks = []
        pawTracks = []
        rungMotion = []
        swingPhases = []
        for r in range(len(foldersRecordings[f][2])):
            # read rotary encoder data for wheel speed
            (rotaryExistence, rotFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
            if rotaryExistence:
                (angluarSpeed, linearSpeed, sTimes, timeStamp, monitor, angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
                tracks.append([angluarSpeed, linearSpeed, sTimes, timeStamp, monitor, angleTimes, foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r]])
            # read paw data
            (camExistence, camFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
            if camExistence:
                (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime,rawPawSpeed,pawPos,croppingParameters) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], DLCinstance)
                pawTracks.append([rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, pawPos, croppingParameters])
                rungPositions = eSD.getRungMotionData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
                rungMotion.append([mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],rungPositions])

            if rotaryExistence and camExistence:
                (swingP,forFit) = dataAnalysis.findStancePhases(tracks[-1],pawTracks[-1],rungMotion[-1],showFigFit=False,showFigPaw=False)
                #pdb.set_trace()
                print(len(swingP[0][1]),len(swingP[1][1]),len(swingP[2][1]),len(swingP[3][1]))
                swingPhases.append([mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],swingP,forFit])

        recordingsM.append([foldersRecordings[f][0],tracks,pawTracks,rungMotion,swingPhases])
        #pdb.set_trace()
    pickle.dump(recordingsM, open(eSD.analysisLocation + '/allSingStanceDataPerSession.p', 'wb'))  # eSD.analysisLocation,

cV.createSwingStanceFigure(recordingsM)
#cV.createSwingTraceFigure(recordingsM,linear=False)
#cV.createSwingTraceFigure(recordingsM,linear=True)
cV.createSwingSpeedProfileFigure(recordingsM,linear=False)
cV.createRungCrossingFigure(recordingsM)