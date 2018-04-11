from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb

mouseD = '180107_m27'
expDateD = '180214'
#mouse = '180112_m33'
#expDate = '180306'

if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    expDate = expDateD
else:
    expDate = args.date

wheelCircumsphere = 79.796 # in cm

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

tracks = []
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'RotaryEncoder')
        if existence:
            (angles, aTimes,timeStamp,monitor) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'RotaryEncoder',fileHandle)
            (angularSpeed, linearSpeed, sTimes)  = dataAnalysis.getSpeed(angles,aTimes,wheelCircumsphere)
            #pdb.set_trace()
            eSD.saveWalkingActivity(angularSpeed, linearSpeed, sTimes,timeStamp,monitor, [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])  # save motion corrected image stack

del eSD
