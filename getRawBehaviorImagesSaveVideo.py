from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import sys


mouseD = '201211_m45' # id of the mouse to analyze
#mouseD = '190108_m24'
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='all'     # 'all or 'some'


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
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

#print(len(foldersRecordings))

#pdb.set_trace()
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range(len(foldersRecordings[f][2])):
        #pdb.set_trace()
        (existenceFrames,fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (existenceFTimes,fileHandleFTimes) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes')
        (existenceLEDControl, fileHandleLED) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'PreAmpInput')
        # if camera was recorded
        if existenceFrames:
            #print('exists',foldersRecordings[f][0],foldersRecordings[f][2][r])
            (frames,softFrameTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior',fileHandleFrames)
        # use frame drop/miss and timing information to save video
        if existenceFrames and existenceFTimes and existenceLEDControl:
            (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfoCopy ) =  eSD.readBehaviorVideoTimeData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behaviorVideo'])
            eSD.saveBehaviorVideo(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], frames, idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo)