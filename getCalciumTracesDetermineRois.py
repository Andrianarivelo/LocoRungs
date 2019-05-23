from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.caImagingSuite2p as caImaging
import pdb
import sys

mouseD = '190101_f15' # id of the mouse to analyze
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'


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
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

caI      = caImaging.caImagingSuite2p(eSD.analysisLocation,eSD.figureLocation,eSD.f)

#pdb.set_trace()
# loop over all folders, mostly days but sometimes there were two recording sessions per day
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    #for r in range(len(foldersRecordings[f][2])):
    (existence, tiffList) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][0], 'SICaImaging')
    # if camera was recorded
    if existence:
        #pdb.set_trace()
        caI.setSuite2pParameters(eSD.dataBase2+foldersRecordings[f][0]+'/',eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p/',tiffList)
        #pdb.set_trace()
        caI.runSuite2pPipeline()
        #caI.generateSummaryFigures()#eSD.saveCaImagingData()





# for f in range(len(foldersRecordings)):
#     for r in range(len(foldersRecordings[f][2])):
#         (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r], 'Imaging')
#         if existence:
#             (frames, fTimes, imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r], 'Imaging', fileHandle)  # read raw data from experiment
#             (imStack, motionCoordinates, tifFile) = eSD.getMotioncorrectedStack(foldersRecordings[f][0],foldersRecordings[f][2][r], 'moco')  # read motion corrected image stack and displacement data
#             eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
#             (img, rois, rawSignals) = eSD.extractRoiSignals(foldersRecordings[f][0],foldersRecordings[f][2][r], tifFile)
#             cV.generateROIImage(foldersRecordings[f][0],foldersRecordings[f][2][r], np.average(imStack, axis=0), fTimes, rois, rawSignals, imageMetaInfo, motionCoordinates)
#
#         #for rec in recordings[16:]:
#         #(existence,fileHandle) = eSD.checkIfDeviceWasRecorded(rec,'Imaging')  # check if specific data was recorded
#
#         #(frames,fTimes,imageMetaInfo) = eSD.readRawData(rec,'Imaging',fileHandle)  # read raw data from experiment
#         #(imStack,motionCoordinates,tifFile) = eSD.getMotioncorrectedStack(dataFolder,rec,'moco') # read motion corrected image stack and displacement data
#         #eSD.saveImageStack(imStack, fTimes, imageMetaInfo, 'motion_corrected',motionCorrection=motionCoordinates) # save motion corrected image stack
#         #(img,rois,rawSignals) = eSD.extractRoiSignals(dataFolder,rec,tifFile) # determine/read rois, and get traces
#
#         #cV.generateROIImage(dataFolder,rec,np.average(imStack,axis=0),fTimes,rois,rawSignals,imageMetaInfo,motionCoordinates) # plot fluorescent traces of rois
#         #break
#
# del eSD, cV
