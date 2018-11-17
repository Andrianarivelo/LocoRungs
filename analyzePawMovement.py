from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import tools.createVisualizations as createVisualizations

import pdb
import sys

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

cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

for f in range(len(foldersRecordings)) :
    for r in range(8,len(foldersRecordings[f][2])): # for r in recordings[f][1]:
        print foldersRecordings[f][2][r]
        (GigExistence,GigFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (RotExistence,RotFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'RotaryEncoder')
        #print existence
        if GigExistence and RotExistence:
            (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'CameraGigEBehavior',GigFileHandle,readRawData=False)
            (angularSpeed, linearSpeed, sTimes, timeStamp, monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
            (frontpawPos,hindpawPos,rungs) = eSD.getPawRungPickleData(foldersRecordings[f][0],foldersRecordings[f][2][r])
            #(fp[1:],hp[1:],rungs,center_2b, R_2b,rungsNumbered)

            (fp, hp, rungs,centerR,Radius,rungsNumbered,fpLinear,hpLinear,frontpawRungDist,hindpawRungDist,startStopFPStep,startStopHPStep) = cv2Tools.analyzePawsAndRungs(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],frontpawPos,hindpawPos,rungs,fTimes,angularSpeed,linearSpeed,sTimes,angleTimes)
            #pdb.set_trace()
            cV.generatePawMovementFigure(foldersRecordings[f][0],foldersRecordings[f][2][r],fp,hp,rungs,fTimes,centerR,Radius,rungsNumbered,fpLinear,hpLinear,linearSpeed,sTimes,frontpawRungDist,hindpawRungDist,startStopFPStep,startStopHPStep)
        pdb.set_trace()