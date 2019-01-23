from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import tools.createVisualizations as createVisualizations

mouseD = '190122_t1'
expDateD = '190122'
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


eSD         = extractSaveData.extractSaveData(mouse,expDate)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

genericData = []
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'PreAmpInput')
        if existence:
            (values, vTimes) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'PreAmpInput',fileHandle)
            genericData.append([foldersRecordings[f][0],foldersRecordings[f][2][r],values, vTimes])

#pdb.set_trace()
cV.generateHistogram(mouse,expDate,foldersRecordings[f][0],genericData)
del eSD
