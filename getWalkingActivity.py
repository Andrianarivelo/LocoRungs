from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.parameters as par
import pdb

mouseD = '190610_f1' # id of the mouse to analyze
expDateD = '190815'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='all'     # 'all or 'some'


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

#tracks = []
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
        if existence:
            (angles, aTimes,timeStamp,monitor) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder',fileHandle)
            (angularSpeed, linearSpeed, sTimes)  = dataAnalysis.getSpeed(angles,aTimes,par.wheelCircumsphere)
            #pdb.set_trace()
            eSD.saveWalkingActivity(angularSpeed, linearSpeed, sTimes, angles, aTimes, timeStamp,monitor, [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])  # save motion corrected image stack

del eSD
