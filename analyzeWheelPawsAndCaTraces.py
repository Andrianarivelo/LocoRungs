from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.createVisualizations as createVisualizations
import pdb, pickle, os
import tools.parameters as pas

###########################################

mouseD = '190101_f15' # id of the mouse to analyze
#mouseD = '190108_m24'
expDateD = 'all'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='all'     # 'all or 'some'

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


eSD = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

cV = createVisualizations.createVisualizations(eSD.figureLocation,mouse)

if os.path.isfile(eSD.analysisLocation + '/allSingStanceDataPerSession.p'):
    recordingsM = pickle.load( open( eSD.analysisLocation + '/allSingStanceDataPerSession.p', 'rb' ) )

if os.path.isfile(eSD.analysisLocation + '/allDataPerSession.p') and not readDataAgain:
    allCorrDataPerSession = pickle.load( open( eSD.analysisLocation + '/allCorrDataPerSession.p', 'rb' ) )
else:
    allCorrDataPerSession = []
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
                (rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime,rawPawSpeed,cPawPos) = eSD.readPawTrackingData(foldersRecordings[f][0], foldersRecordings[f][2][r])
                #pdb.set_trace()
                paws.append([rawPawPositionsFromDLC,pawTrackingOutliers,jointNamesFramesInfo,pawSpeed,recStartTime])
        # check for ca-imaging data during entire session
        (caImgExistence, tiffList) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][0], 'SICaImaging')
        if caImgExistence: # (Fluo,nRois,ops,frameNumbers)
            (Fluo,nRois,ops,timeStamps,stat) =  eSD.getCaImagingRoiData(eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p/',tiffList)
            caimg.append([Fluo,nRois,ops,timeStamps,stat])
        # combine all recordings from a session only if all three data-sets were recorded
        if (not wheel) and (not paws) and (not caimg):
            pass
        else:
            allCorrDataPerSession.append([foldersRecordings[f][0],wheel,paws,caimg])
    pickle.dump(allCorrDataPerSession, open(eSD.analysisLocation + '/allCorrDataPerSession.p', 'wb'))  # eSD.analysisLocation,


#allCorrDataPerSessionOrdered = dataAnalysis.findMatchingRois(mouse,allCorrDataPerSession,eSD.analysisLocation,refDate=3)

#pdb.set_trace()
# generate overview figure for animal
#Rvalues = dataAnalysis.doRegressionAnalysis(mouse,allCorrDataPerSession)
#cV.generateR2ValueFigure(mouse,Rvalues,'R2-Values')

#borders = [10,25]
#Rvalues2 = dataAnalysis.doRegressionAnalysis(mouse,allCorrDataPerSession,borders=borders)
#cV.generateR2ValueFigure(mouse,Rvalues2,'R2-Values-LocomotionPeriod_%s-%s_only' % (borders[0],borders[1]))
#pdb.set_trace()

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
#caTriggeredAverages = dataAnalysis.generateStepTriggeredCaTraces(mouse,allCorrDataPerSession,recordingsM)
#pickle.dump(caTriggeredAverages, open(eSD.analysisLocation + '/caSwingPhaseTriggeredAverages.p', 'wb'))  # eSD.analysisLocation,
caTriggeredAverages = pickle.load(open(eSD.analysisLocation + '/caSwingPhaseTriggeredAverages.p', 'rb'))
cV.generateSwingTriggeredCaTracesFigure(caTriggeredAverages,rescal=False)
cV.generateSwingTriggeredCaTracesFigure(caTriggeredAverages,rescal=True)
#pickle.dump(caTriggeredAverages, open('caSwingPhaseTriggeredAverages.p', 'wb'))

pdb.set_trace()

pawSwingTimes = dataAnalysis.generateInterstepTimeHistogram(mouse,allCorrDataPerSession,recordingsM)
cV.generateSwingTimesHistograms(pawSwingTimes)


pdb.set_trace()
(correlationData,varExplained) = dataAnalysis.doCorrelationAnalysis(mouse,allCorrDataPerSession)
# cV.generateCaWheelPawImage(mouse,allCorrDataPerSession)
cV.generateCorrelationPlotCaTraces(mouse,correlationData,allCorrDataPerSession)
#cV.generatePCACorrelationPlot(mouse,correlationData,allCorrDataPerSession,varExplained)
cV.generateCorrelationPlotsCaWheelPaw(mouse,correlationData,allCorrDataPerSession)

