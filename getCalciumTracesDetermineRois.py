from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.caImagingSuite2p as caImaging
import os
import pdb
import sys

mouseD = '201220_m59' # id of the mouse to analyze
expDateD = 'all910'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='all910'     # 'all or 'some'

onAllData = False

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
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

caI      = caImaging.caImagingSuite2p(eSD.analysisLocation,eSD.figureLocation,eSD.f)

# loop over all folders, mostly days but sometimes there were two recording sessions per day
dataDirs = []
allTiffs = []
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    #for r in range(len(foldersRecordings[f][2])):
    (existence, tiffList,recLocation) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][0], 'SICaImaging')
    #pdb.set_trace()
    #tiffList = tiffList[-7:-1]
    # if camera was recorded
    if existence:
        #pdb.set_trace()
        specificTiffLists = caI.decideWhichTiffFilesToUse(recLocation,tiffList)
        print(len(specificTiffLists),specificTiffLists)
        #pdb.set_trace()
        dataDirs.append(eSD.dataBase+foldersRecordings[f][0]+'/')
        allTiffs.extend(tiffList)
        if not onAllData:
            for i in range(len(specificTiffLists)): # loop over the the lists of tiff files to analyze
                print('analysis on :',specificTiffLists[i])
                #inputList = []
                #inputList.append(recLocation)
                #inputList.append(eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[i][1])
                #saveLocation = eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[i][1]
                #inputList.extend(specificTiffLists[i][0])
                #pdb.set_trace()
                #os.system(eSD.suite2pPath+' tools/runSuite2p.py %s %s ' % (recLocation,saveLocation) + ' '.join(specificTiffLists[i][0]) )
                caI.runSuite2pPipeline(eSD.suite2pPath,recLocation,eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[i][1],specificTiffLists[i][0])
                #
                #caI.runSuite2pPipeline()
                #
                eSD.extractAndSaveCaTimeStamps(recLocation,eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[i][1],specificTiffLists[i][0])
                #
                caI.generateOverviewFigure(eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[i][1],specificTiffLists[i][0],mouseD,foldersRecordings[f][0])
                #pdb.set_trace()

# pdb.set_trace()
#if onAllData:
#    caI.setSuite2pParameters(dataDirs,eSD.analysisLocation+'suite2p/')
#    caI.runSuite2pPipeline()



