from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import sys


mouseD = '201017_m99' #'200801_m58' # id of the mouse to analyze
#mouseD = '190108_m24'
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'

# comment

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


#print mouse, expDate
#sys.exit(0) #pdb.set_trace()
eSD         = extractSaveData.extractSaveData(mouse)  # find data folder of specific mouse, create data folder, and hdf5 handle
#pdb.set_trace()
#(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
openCVtools  = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
#print(len(foldersRecordings))

#pdb.set_trace()
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range(1,len(foldersRecordings[f][2])):
        #pdb.set_trace()
        (existenceFrames,fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (existenceFTimes,fileHandleFTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes')
        (existenceLEDControl, fileHandleLED) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput')
        (currentCoodinatesExist,SavedLEDcoordinates) = eSD.checkForLEDPositionCoordinates(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r)
        (erroneousFramesExist,idxToExclude) = eSD.checkForErroneousFramesIdx(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r)
        # if camera was recorded
        if existenceFrames:
            #print('exists',foldersRecordings[f][0],foldersRecordings[f][2][r])
            (frames,softFrameTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior',fileHandleFrames)
            (ledCoordinates,ledTraces) = openCVtools.findLEDNumberArea(frames,coordinates=SavedLEDcoordinates,currentCoordExist=currentCoodinatesExist,determineAgain=True,verbose=False)
            #pdb.set_trace()
            eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinVideo'],ledCoordinates)
        if existenceFTimes:
            (exposureDAQArray,exposureDAQArrayTimes) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes',fileHandleFTimes)
        if existenceLEDControl:
            (ledDAQControlArray, ledDAQControlArrayTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput', fileHandleLED)
        # save data
        if not erroneousFramesExist:
            idxToExclude = dataAnalysis.determineErroneousFrames(frames)
            eSD.saveErroneousFramesIdx([foldersRecordings[f][0], foldersRecordings[f][2][r], 'erroneousFrames'],idxToExclude)
        #pdb.set_trace()
        if existenceFrames and existenceFTimes and existenceLEDControl:
            (idxVideo,idxTimePoints,startEndExposureTime,startEndExposurepIdx,rightShift) = dataAnalysis.determineFrameTimesBasedOnLED([ledTraces,ledCoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],verbose=True)
            #framesDuringRecording = frames[recordedFramesIdx]
            eSD.saveBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video'], idxVideo,idxTimePoints,startEndExposureTime,startEndExposurepIdx,rightShift, imageMetaInfo)
            #eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)