from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import numpy as np
import sys

mouseD = '210214_m15'
#mouseD = '201017_m99' #'200801_m58' # id of the mouse to analyze
#mouseD = '190108_m24'
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'

# further analaysis parameter
assumePerfectRecording = True  # that means a recording without any flash-back - or double frames
startRecording = None # each session/day per animal is composed of 5 recordings, this index allows chose with which recording to start, default is 0
endRecording = None # in case only specify recording will be analyzed, otherwise set to None
DetermineAgainLEDcoordinates = True # whether or not to determine LED coordinates even though they exist already for current or previous recording
DetermineAgainErronousFrames = False # whether or not to determine errnonous frames even though the are already exist for current recording
recordingWithTail = True

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
    for r in range((0 if startRecording is None else startRecording),(len(foldersRecordings[f][2]) if endRecording is None else endRecording)):
        #pdb.set_trace()
        (existenceFrames,fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (existenceFTimes,fileHandleFTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes')
        (existenceLEDControl, fileHandleLED) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput')
        (currentCoodinatesExist,SavedLEDcoordinates) = eSD.checkForLEDPositionCoordinates(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r)
        (erroneousFramesExist,idxToExclude,canBeUsed) = eSD.checkForErroneousFramesIdx(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r,determineAgain=DetermineAgainErronousFrames)
        # if camera was recorded
        if existenceFrames:
            #print('exists',foldersRecordings[f][0],foldersRecordings[f][2][r])
            (frames,softFrameTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior',fileHandleFrames)
        # determine the
        if (SavedLEDcoordinates is None) or DetermineAgainLEDcoordinates :
            LEDcoordinates = openCVtools.findLEDNumberArea(frames,coordinates=SavedLEDcoordinates,currentCoordExist=currentCoodinatesExist,determineAgain=True,verbose=False)
            eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinVideo'],LEDcoordinates)
        else:
            LEDcoordinates = SavedLEDcoordinates
            eSD.saveLEDPositionCoordinates([foldersRecordings[f][0], foldersRecordings[f][2][r], 'LEDinVideo'], LEDcoordinates)
        LEDtraces = openCVtools.extractLEDtraces(frames,LEDcoordinates,verbose=False)
        #pdb.set_trace()
        if existenceFTimes:
            (exposureDAQArray,exposureDAQArrayTimes) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes',fileHandleFTimes)
        if existenceLEDControl:
            (ledDAQControlArray, ledDAQControlArrayTimes) = eSD.readRawData(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput', fileHandleLED)
        # save data
        #idxToExclude = np.array([], dtype=np.int64)
        if assumePerfectRecording:
            idxToExclude = np.array([], dtype=np.int64)
            canBeUsed = True
        else:
            if (not erroneousFramesExist) and canBeUsed:
                (idxToExclude,canBeUsed) = dataAnalysis.determineErroneousFrames(frames)
                eSD.saveErroneousFramesIdx([foldersRecordings[f][0], foldersRecordings[f][2][r], 'erroneousFrames'],idxToExclude,canBeUsed=canBeUsed)
        #pdb.set_trace()
        if existenceFrames and existenceFTimes and existenceLEDControl and canBeUsed:
            #(idxIllumFinal, frameTimes, frameStartStopIdx, videoIdx, frameSummary)
            (idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary) = dataAnalysis.determineFrameTimesBasedOnLED([LEDtraces,LEDcoordinates,frames,softFrameTimes,imageMetaInfo,idxToExclude],[exposureDAQArray,exposureDAQArrayTimes],[ledDAQControlArray, ledDAQControlArrayTimes],eSD.recordingMachine,verbose=True,tail=recordingWithTail)
            #framesDuringRecording = frames[recordedFramesIdx]
            eSD.saveBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo'],idxTimePoints,startEndExposureTime,startEndExposurepIdx,videoIdx,frameSummary, imageMetaInfo)
            #eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)