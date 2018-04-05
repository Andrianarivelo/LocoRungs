from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
import pdb
import sys

#mouse = '170927_m68'
#expDate = '171115'
#mouse = '171126_m90'
#expDate = '180118'
mouseD = '180112_m33'
expDateD = '180309'

# in case mouse, and date were specified as input arguments
if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    expDate = expDateD
else:
    expDate = args.date


#print mouse, expDate
sys.exit(0) #pdb.set_trace()
eSD         = extractSaveData.extractSaveData(mouse)  # find data folder of specific mouse, create data folder, and hdf5 handle
(foldersRecordings,dataFolders) = eSD.getRecordingsList(mouse,expDate) # get recordings for specific mouse and date

# loop over all recording folders

for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'CameraGigEBehavior')
        if existence:
            #print 'exists'
            (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'CameraGigEBehavior',fileHandle)
            eSD.saveBehaviorVideo(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],frames,fTimes,imageMetaInfo)
            #pdb.set_trace()