from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify day of recording", required=False)
tools.argparser.add_argument("-r","--recs", help="specify index of the specify recording on that day", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pickle
import os
import pdb

mouseD = '210214_m15'
expDateD = 'all910' # specific date e.g. '180214', 'some' for manual selection, 'all' for all, 'all910' for all recordings at 910 nm
recordingsD='all910' # 'all or 'some' or 'all910', or index of the recoding - e.g. 0,1 - when running analysis for a specific day
DLCinstance = 'DLC_resnet_50_2021Jun_PawExtraction_m15Jun16shuffle3_200000'

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

#recordings = [int(i) for i in recordings.split(',')]

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV       = createVisualizations.createVisualizations(eSD.figureLocation,mouse)
# = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
# loop over all folders, mostly days but sometimes there were two recording sessions per day

if expDateD == 'all910' or expDateD == 'all820':
    pickleFileName = eSD.analysisLocation + '/allOutlierFramesPerSession_%s_%s.p' % (DLCinstance, expDateD)
else:
    pickleFileName = eSD.analysisLocation + '/allOutlierFramesPerSession_%s.p' % (DLCinstance)


#pdb.set_trace()
if os.path.isfile(pickleFileName) and not readDataAgain:
    outlierData = pickle.load( open(pickleFileName, 'rb') )
else:
    outlierData = []
    for f in range(len(foldersRecordings)) :
        # loop over all recordings in that folder
        for r in range(len(foldersRecordings[f][2])): # for r in recordings[f][1]:
            (existenceFrames,FramesFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r],'CameraGigEBehavior')
            (existencePawPos,PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], DLCinstance)
            if existenceFrames and existencePawPos:
                (pawPositions,pawMetaData) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'pawTraces',PawFileHandle)
                #pdb.set_trace()
                pawTrackingOutliers = dataAnalysis.detectPawTrackingOutlies(pawPositions,pawMetaData)
                #DLCinstance = pawMetaData['data']['DLC-model-config file']['snapshot_prefix']
                outlierData.append([foldersRecordings[f][0],foldersRecordings[f][2][r],pawTrackingOutliers,DLCinstance])
                cV.createPawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],pawTrackingOutliers)
                (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo) = eSD.readBehaviorVideoTimeData([foldersRecordings[f][0],foldersRecordings[f][2][r],'behaviorVideo'])  #[foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo']
                eSD.savePawTrackingData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],DLCinstance,pawPositions,pawTrackingOutliers,pawMetaData,startEndExposureTime,imageMetaInfo,generateVideo=False)

    pickle.dump(outlierData, open(pickleFileName, 'wb'))
cV.createOutlierStatFigure(foldersRecordings,outlierData,DLCinstance,expDateD)
