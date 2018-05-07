from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.pawClassifier as pawClassifier
import pdb
import numpy as np
import pickle

mouseD = '180107_m27'
expDateD = '180214'
#mouse = '171126_m90'
#expDate = '180118'
#mouse = '171218_f8'
#expDate = '180123'

# in case mouse, and date were specified as input arguments
if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    expDate = expDateD
else:
    expDate = args.date


eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

PClassifier = pawClassifier.pawClassifier(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)

thresholdPaws = np.zeros((6007,3))
thresholdPaws[:,0][:] = 0.3
thresholdPaws[:,1][:] = 0
thresholdPaws[:,2][:] = 1

roiInformation = pickle.load(open(eSD.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/Masks.f'))

#pdb.set_trace()

for f in range(len(foldersRecordings)) :
    for r in range(8,len(foldersRecordings[f][2])): # for r in recordings[f][1]:
        print foldersRecordings[f][2][r]
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'CameraGigEBehavior')
        print existence
        if existence:
            masks = roiInformation[mouse][foldersRecordings[f][0]][foldersRecordings[f][2][r]]
            #pdb.set_trace()
            PClassifier.extratContourInformation(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],thresholdPaws,masks['WheelMask'])
        pdb.set_trace()