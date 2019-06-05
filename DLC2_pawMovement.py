from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import matplotlib.pyplot as plt
import numpy as np
import h5py

mouseD = '190101_f15' # id of the mouse to analyze
expDateD = 'all'     # specific date e.g. '180214', 'some' for manual selection or 'all'
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

eSD         = extractSaveData.extractSaveData(mouse)
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings)  # get recordings for specific mouse and date

#########################################################
# Get paws coordinates
recordingsM = []
mouse_OF = [] # OF = Outliers Frames
total_of = 0
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    recordingsD = []
    day_OF = []
    tmp = foldersRecordings[f][2][-5:]
    tmp.insert(0, foldersRecordings[f][2][0])
    foldersRecordings[f][2] = tmp
    for r in range(len(foldersRecordings[f][2])):
        (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '-' +foldersRecordings[f][2][r][-3:])
        if existencePawPos:
            (pawPositions,pawMetaData) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r][:-4] + '-' + foldersRecordings[f][2][r][-3:],'pawTraces',PawFileHandle)
            pawTrackingOutliers = dataAnalysis.detectPawTrackingOutlies(pawPositions,pawMetaData,showFig=False)
            total_of = total_of + (pawTrackingOutliers[0][1]-pawTrackingOutliers[0][2]) + (pawTrackingOutliers[1][1]-pawTrackingOutliers[1][2]) + (pawTrackingOutliers[2][1]-pawTrackingOutliers[2][2]) + (pawTrackingOutliers[3][1]-pawTrackingOutliers[3][2])
            (FR_th, FL_th, HL_th, HR_th) = pawPositions[pawTrackingOutliers[0][3],:][:, [1, 2, 0]], pawPositions[pawTrackingOutliers[1][3],:][:, [4, 5, 0]], pawPositions[pawTrackingOutliers[2][3],:][:, [7, 8, 0]], pawPositions[pawTrackingOutliers[3][3],:][:, [10, 11, 0]] # Remove outlier data
            day_OF.append(pawTrackingOutliers)
            recordingsD.append([FR_th, FL_th, HL_th, HR_th])
    mouse_OF.append(day_OF)
    recordingsM.append(recordingsD)
del recordingsD

#########################################################
# Get session speed from rotary encoder and video exposure data
mouse_tracks = []
for f in range(len(foldersRecordings)):
    day_tracks = []
    for r in range(1, len(foldersRecordings[f][2])):
        (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '/' + foldersRecordings[f][2][r][-3:], 'RotaryEncoder')
        if existence:
            (angularSpeed,linearSpeed,wTimes,startTime,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
            (startExposure, endExposure) = eSD.getBehaviorVideoData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video'])
            day_tracks.append([-linearSpeed, wTimes, startExposure, endExposure, angularSpeed])
    mouse_tracks.append(day_tracks)

#########################################################
# Get day's overall activity

mouse_activity = []
for f in range(len(foldersRecordings)):
    (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1],
                                                           foldersRecordings[f][2][0], 'RotaryEncoder')
    if existence:
        (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes) = eSD.getWalkingActivity(
            [foldersRecordings[f][0], foldersRecordings[f][2][0], 'walking_activity'])
        mouse_activity.append([linearSpeed, wTimes])
#########################################################
# Absolute speed over time

mouse_speed, mouse_time = dataAnalysis.getPawSpeed(recordingsM, mouse_tracks)

##########################################################
# Average stride and step length

#########################################################
# Stance phases detection based on wheel and paw speed difference

speedDiffThresh = 16  # Speed threshold, determine with variance
thStance = 10
thSwing = 0
trailingStart = 3
trailingEnd = 3
mouse_swing = []
mouse_speedDiff = []
for day in range(len(recordingsM)):
    day_swing = []
    day_speedDiff = []
    for sess in range(len(recordingsM[day])):
        paw_swing = []
        paw_speedDiff = []
        for p in range(len(recordingsM[day][sess])):
            speedDiff = np.squeeze(np.diff([np.interp(mouse_time[day][sess][p], mouse_tracks[day][sess][1], mouse_tracks[day][sess][0]), mouse_speed[day][sess][p]*5], axis=0))
            swingIndices, swingPhases = dataAnalysis.findStancePhases(speedDiff, speedDiffThresh, thStance, thSwing, trailingStart, trailingEnd)
            paw_swing.append((swingIndices, swingPhases))
            paw_speedDiff.append(speedDiff)
        day_swing.append(paw_swing)
        day_speedDiff.append(paw_speedDiff)
    mouse_swing.append(day_swing)
    mouse_speedDiff.append(day_speedDiff)

day = 0
sess = 0
plt.figure()
plt.suptitle('Mouse:' + mouse + '; Day:' + foldersRecordings[day][0] + '; Session:' + foldersRecordings[day][2][sess+1])
plt.subplot(2,1,1)
plt.title('Stance and step phases detection during a recording session')
plt.plot([0,30], [speedDiffThresh,speedDiffThresh], [0,30], [-speedDiffThresh,-speedDiffThresh], linestyle='--', c='0.5')
# plt.text(0, 20, 'Threshold=%s' % th,fontsize=6 )
# plt.text(0, -20, 'Threshold=%s' % -th,fontsize=6 )
FR_paw, = plt.plot(mouse_time[day][sess][0][trailingStart:-(trailingEnd+1)], mouse_speedDiff[day][sess][0][trailingStart:-(trailingEnd+1)] , c='b', label='Front right paw')
plt.plot(mouse_time[day][sess][0], mouse_swing[day][sess][0][1], c='slateblue')
# FL_paw, = plt.plot(mouse_time[day][sess][1][trailingStart:-(trailingEnd+1)], mouse_speedDiff[day][sess][1][trailingStart:-(trailingEnd+1)] , c='orange', label='Front left paw')
# plt.plot(mouse_time[day][sess][1], mouse_swing[day][sess][1][1], c='moccasin')

plt.xlabel('Time during recording session(s)')
plt.ylabel('X speed difference between a paw and the wheel (a.u.)')
plt.legend(handles=[FR_paw]) #, FL_paw])

plt.subplot(2,1,2)
plt.plot([0,30], [speedDiffThresh,speedDiffThresh], [0,30], [-speedDiffThresh,-speedDiffThresh], linestyle='--', c='0.5')
# plt.text(0, 20, 'Threshold=%s' % th,fontsize=6 )
# plt.text(0, -20, 'Threshold=%s' % -th,fontsize=6 )
HR_paw, = plt.plot(mouse_time[day][sess][3][trailingStart:-(trailingEnd+1)], mouse_speedDiff[day][sess][3][trailingStart:-(trailingEnd+1)] , c='b', label='Hind right paw')
plt.plot(mouse_time[day][sess][3], mouse_swing[day][sess][3][1], c='slateblue')
# HL_paw, = plt.plot(mouse_time[day][sess][2][trailingStart:-(trailingEnd+1)], mouse_speedDiff[day][sess][2][trailingStart:-(trailingEnd+1)] , c='orange', label='Hind left paw')
# plt.plot(mouse_time[day][sess][2], mouse_swing[day][sess][2][1], c='moccasin')
plt.ylabel('X speed difference between a paw and the wheel (a.u.)')
plt.legend(handles=[HR_paw]) #, HL_paw])
plt.show()
#########################################################
# Plots for multiple sessions



# TTL pulse at 5s (index 1000), wheel accelerate at 7s (index 1400), max speed at 10.8 (index 2160) during 12s, wheel decelerate at 22.8 (index 4560), wheel disconnect at 26.6 (index 5320)
# Speed for 9 cm.s^-1 is 20 on arduino2P, arduinoSetup1, 16 on arduinoSetup2