from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
#import tools.createVisualizations as createVisualizations

import pdb
import sys

mouseD = '180602_m78'
expDATE = '180724'

normalizationRec = ['180724','2pScanning_calibration_larger_002']

# in case mouse, and date were specified as input arguments
if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    expDate = expDATE
else:
    expDate = args.date


eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

#cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)
#cV      = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

# read normalization mask first
for f in range(len(foldersRecordings)) :
    if foldersRecordings[f][1]==normalizationRec[0]:
        for r in range(len(foldersRecordings[f][2])):
            if foldersRecordings[f][2][r] == normalizationRec[1]:
                (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][2][r], 'Imaging')
                if existence:
                    (normFrame,_,normImageMetaInfo) = eSD.readImageStack([foldersRecordings[f][0],foldersRecordings[f][2][r],'raw_imaging_data'])

#pdb.set_trace()

for f in range(len(foldersRecordings)):
    for r in [14]: #range(len(foldersRecordings[f][2])):
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'Imaging')
        if existence:
            print r, foldersRecordings[f][2][r]
            (frames,fTimes,imageMetaInfo) = eSD.readImageStack([foldersRecordings[f][0],foldersRecordings[f][2][r],'raw_imaging_data'])
            normFrames =  dataAnalysis.applyImageNormalizationMask(frames,imageMetaInfo,normFrame,normImageMetaInfo,mouse,foldersRecordings[f][0],foldersRecordings[f][2][r])
            eSD.saveTif(normFrames[:,:,:,0], mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],norm='GaussSmoothed') # tif file for possible image registration in ImageJ
