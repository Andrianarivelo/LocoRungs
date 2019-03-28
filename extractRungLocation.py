from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb

mouseD = '190101_f15'
expDateD = 'some' # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some' # 'all or 'some'

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

cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)) :
    # loop over all recordings in that folder
    for r in range(len(foldersRecordings[f][2])): # for r in recordings[f][1]:
        #print foldersRecordings[f][2][r]
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        if existence:
            cv2Tools.trackRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
            #cv2Tools.trackPawsAndRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
        pdb.set_trace()