from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
#import tools.openCVImageProcessingTools as openCVImageProcessingTools

import pdb

mouseD = '190101_f15'
expDateD = 'all' # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='all' # 'all or 'some'

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


eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV       = createVisualizations.createVisualizations(eSD.figureLocation,mouse)
# = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)) :
    # loop over all recordings in that folder
    for r in range(len(foldersRecordings[f][2])): # for r in recordings[f][1]:
        (existenceFrames,FramesFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '/' +foldersRecordings[f][2][r][-3:],'CameraGigEBehavior')
        (existencePawPos,PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '-' +foldersRecordings[f][2][r][-3:])
        if existenceFrames and existencePawPos:
            (pawPositions,pawMetaData) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'pawTraces',PawFileHandle)
            pawTrackingOutliers = dataAnalysis.detectPawTrackingOutlies(pawPositions,pawMetaData)
            cV.createPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],pawTrackingOutliers)
            (firstLastFrames, expStartTime, expEndTime,startTime) = eSD.readBehaviorVideoData([foldersRecordings[f][0],foldersRecordings[f][2][r],'behavior_video'])
            eSD.savePawTrackingData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],pawPositions,pawTrackingOutliers,pawMetaData,expStartTime, expEndTime,startTime,generateVideo=False)
        #pdb.set_trace()

