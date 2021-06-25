from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-r","--recs", help="specify index of the specify recording on that day", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import tools.caImagingSuite2p as caImaging
import pdb, pickle, os
import tools.parameters as pas

###########################################

mouseD = '210214_m15' # id of the mouse to analyze
expDateD = 'all910'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordingsD='all910'     # 'all or 'some'
DLCinstance = 'DLC_resnet_50_2021Jun_PawExtraction_m15Jun16shuffle3_200000'

readDataAgain = False

###########################################

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

if args.recs == None:
    try:
        recordings = recordingsD
    except :
        recordings = 'all'
else:
    recordings = args.recs


eSD = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

###################################
# pickle file names
#if expDateD == 'all910' or expDateD == 'all820':
pickleSwingStanceFileName = eSD.analysisLocation + '/allSingStanceDataPerSession_%s.p' % expDate
pickleAnalysisFileName = eSD.analysisLocation + '/allCorrDataPerSession_%s.p' % expDate

if os.path.isfile(pickleSwingStanceFileName):
    recordingsM = pickle.load(open( pickleSwingStanceFileName, 'rb' ) )

if os.path.isfile(pickleAnalysisFileName) and not readDataAgain:
    allCorrDataPerSession = pickle.load( open( pickleAnalysisFileName, 'rb' ) )
else:
    allCorrDataPerSession = []
    caI = caImaging.caImagingSuite2p(eSD.analysisLocation, eSD.figureLocation, eSD.f)
    for f in range(len(foldersRecordings)): # loop over all days
        wheel = []
        paws = []
        caimg = []
        for r in range(len(foldersRecordings[f][2])): # loop over all recordings
            #print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
            # check for rotary encoder
            (rotaryExistence, rotFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'RotaryEncoder')
            if rotaryExistence:  #angularSpeed,linearSpeed,wTimes,startTime,monitor
                (angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity'])
                wheel.append([angluarSpeed,linearSpeed,sTimes,timeStamp,monitor,angleTimes,foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r]])
            # check for video recording during trial
            (camExistence, camFileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'CameraGigEBehavior')
            if camExistence :
                (rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime,rawPawSpeed,cPawPos,croppingParameters) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r], DLCinstance)
                #pdb.set_trace()
                paws.append([rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime])
        # check for ca-imaging data during entire session
        (caImgExistence, tiffList, recLocation) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][0], 'SICaImaging')
        if caImgExistence: # (Fluo,nRois,ops,frameNumbers)
            specificTiffLists = caI.decideWhichTiffFilesToUse(recLocation, tiffList, eSD.expDict[foldersRecordings[f][1]], recordings)
            #pdb.set_trace()
            (Fluo,nRois,ops,timeStamps,stat) =  eSD.getCaImagingRoiData(eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p_%s/' % specificTiffLists[0][2],tiffList)
            caimg.append([Fluo,nRois,ops,timeStamps,stat])
        # combine all recordings from a session only if all three data-sets were recorded
        if (not wheel) and (not paws) and (not caimg):
            pass
        else:
            allCorrDataPerSession.append([foldersRecordings[f][0],wheel,paws,caimg])
    pickle.dump(allCorrDataPerSession, open(pickleAnalysisFileName, 'wb'))  # eSD.analysisLocation,

#######################################################
# remove second and third recording since rotary encoder was not working for them
for n in range(len(allCorrDataPerSession)):
    print(allCorrDataPerSession[n][0])

if mouse == '210122_f84':
    allCorrDataPerSession.pop(1)
    allCorrDataPerSession.pop(1)
    recordingsM.pop(1)
    recordingsM.pop(1)
    #del allCorrDataPerSession[0:4]
print()
for n in range(len(allCorrDataPerSession)):
    print(allCorrDataPerSession[n][0])

pdb.set_trace()
#######################################################
# check which ROIs have been recorded across days

#allCorrDataPerSessionOrdered = dataAnalysis.findMatchingRoisSuccessivDays(mouse,allCorrDataPerSession,eSD.analysisLocation,expDate,eSD.figureLocation,allDataRead=allCorrDataPerSessionOrderedRead)
(allCorrDataPerSessionOrdered,corrMatrix) = dataAnalysis.findMeanImageOverlay(mouse,allCorrDataPerSession,eSD.analysisLocation,expDate,eSD.figureLocation)
pdb.set_trace()
allCorrDataPerSessionOrdered = dataAnalysis.findMatchingRoisSuccessivDays(mouse,allCorrDataPerSession,eSD.analysisLocation,expDate,eSD.figureLocation)
# pickle.dump(allCorrDataPerSessionOrdered, open(eSD.analysisLocation+'/imageAlignmentData_%s.p' % expDate, 'wb'))
# allCorrDataPerSessionOrdered = pickle.load(open(eSD.analysisLocation + '/imageAlignmentData_%s.p' % expDate, 'rb'))
# dataAnalysis.roisRecordedAllDays(allCorrDataPerSessionOrdered)
# pdb.set_trace()

#######################################################
# print('Do regression analysis between calcium and paw speed ...')
# Rvalues = dataAnalysis.doRegressionAnalysis(mouse,allCorrDataPerSession)
# cV.generateR2ValueFigure(mouse,Rvalues,'R2-Values')
# print('done')


#borders = [10,25] # borders ?
#Rvalues2 = dataAnalysis.doRegressionAnalysis(mouse,allCorrDataPerSession,borders=borders)
#cV.generateR2ValueFigure(mouse,Rvalues2,'R2-Values-LocomotionPeriod_%s-%s_only' % (borders[0],borders[1]))

# pdb.set_trace()

#######################################################
# step triggered averages where steps from all paws are used

# caTriggeredAveragesAllPaws = dataAnalysis.generateStepTriggeredCaTracesAllPaws(mouse,allCorrDataPerSession,recordingsM)
# pickle.dump(caTriggeredAveragesAllPaws, open(eSD.analysisLocation + '/caSwingPhaseTriggeredAveragesAllPaws.p', 'wb'))
# caTriggeredAveragesAllPaws = pickle.load(open(eSD.analysisLocation + '/caSwingPhaseTriggeredAveragesAllPaws.p', 'rb'))
# maxMin = cV.generateSwingTriggeredCaTracesFigureAllPaws(caTriggeredAveragesAllPaws,rescal=False)
# pickle.dump(maxMin, open(eSD.analysisLocation + '/caSwingPhaseMaximumMinimum.p', 'wb'))
# maxMin = pickle.load(open(eSD.analysisLocation + '/caSwingPhaseMaximumMinimum.p', 'rb'))
#
# cV.directionOfChangeSpatialOrganization(maxMin,allCorrDataPerSession)
# pdb.set_trace()
#cV.generateSwingTriggeredCaTracesFigureAllPaws(caTriggeredAveragesAllPaws,rescal=True)

#######################################################
print( 'Do step triggered, paw-specific averages of the calcium signal ...')
caTriggeredAverages = dataAnalysis.generateStepTriggeredCaTraces(mouse,allCorrDataPerSession,recordingsM)
pickle.dump(caTriggeredAverages, open(eSD.analysisLocation + '/caSwingPhaseTriggeredAverages_%s.p' % expDate, 'wb'))  # eSD.analysisLocation,
caTriggeredAverages = pickle.load(open(eSD.analysisLocation + '/caSwingPhaseTriggeredAverages_%s.p' % expDate, 'rb'))
cV.generateSwingTriggeredCaTracesFigure(caTriggeredAverages,expDate,rescal=False)
cV.generateSwingTriggeredCaTracesFigure(caTriggeredAverages,expDate,rescal=True)
cV.generateSwingTriggeredCa3DProfilesFigure(caTriggeredAverages,expDate,rescal=False)
print('done')

pdb.set_trace()

#######################################################
pawSwingTimes = dataAnalysis.generateInterstepTimeHistogram(mouse,allCorrDataPerSession,recordingsM)
cV.generateSwingTimesHistograms(pawSwingTimes)


pdb.set_trace()
(correlationData,varExplained) = dataAnalysis.doCorrelationAnalysis(mouse,allCorrDataPerSession)
# cV.generateCaWheelPawImage(mouse,allCorrDataPerSession)
cV.generateCorrelationPlotCaTraces(mouse,correlationData,allCorrDataPerSession)
#cV.generatePCACorrelationPlot(mouse,correlationData,allCorrDataPerSession,varExplained)
cV.generateCorrelationPlotsCaWheelPaw(mouse,correlationData,allCorrDataPerSession)

