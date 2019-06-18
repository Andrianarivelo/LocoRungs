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
import scipy.stats as stats


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
for f in range(1, len(foldersRecordings)):
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
for f in range(1, len(foldersRecordings)):
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
for f in range(1, len(foldersRecordings)):
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
# Stance and swing phases detection based on wheel and paw speed difference

speedDiffThresh = 16  # Speed threshold, determine with variance
thStance = 10
thSwing = 2
trailingStart = 3
trailingEnd = 3
mouse_swing = []
mouse_stance = []
mouse_speedDiff = []
bounds = [1400, 5320]  # or bounds = np.arange(0, len(speedDiff))
for day in range(len(recordingsM)):
    day_swing = []
    day_stance = []
    day_speedDiff = []
    for sess in range(len(recordingsM[day])):
        paw_swing = []
        paw_stance = []
        paw_speedDiff = []
        for p in range(len(recordingsM[day][sess])):
            speedDiff = np.squeeze(np.diff([np.interp(mouse_time[day][sess][p], mouse_tracks[day][sess][1], mouse_tracks[day][sess][0]), mouse_speed[day][sess][p]*5], axis=0))
            swingIndices, swingPhases, stanceIndices, stancePhases = dataAnalysis.findStancePhases(speedDiff, speedDiffThresh, thStance, thSwing, trailingStart, trailingEnd, bounds)
            paw_swing.append((swingIndices, swingPhases))
            paw_stance.append((stanceIndices, stancePhases))
            paw_speedDiff.append(speedDiff)
        day_swing.append(paw_swing)
        day_stance.append(paw_stance)
        day_speedDiff.append(paw_speedDiff)
    mouse_swing.append(day_swing)
    mouse_stance.append(day_stance)
    mouse_speedDiff.append(day_speedDiff)



day = 11
sess = 4
p = 0
# plt.figure(figsize=[6.4, 4.8])
# plt.suptitle('Mouse:' + mouse + '; Day:' + foldersRecordings[day+1][0] + '; Session:' + foldersRecordings[day+1][2][sess+1])
# plt.title('Front right paw and wheel ')
# plt.plot(mouse_time[day][sess][p], mouse_speed[day][sess][p]*5)
# plt.plot(mouse_tracks[day][sess][1], mouse_tracks[day][sess][0])
# plt.xlabel('Time during recording session(s)')
# plt.ylabel('Speed of paw and wheel (a.u)')
# plt.legend(labels=['Front right paw speed', 'Linear speed of the wheel'])

# plt.figure(figsize=[6.4, 4.8])
# plt.title('Stance and step phases detection during a recording session')
# plt.plot([0,30], [speedDiffThresh,speedDiffThresh], linestyle='--', c='0.3')
# plt.plot([0,30], [-speedDiffThresh,-speedDiffThresh], linestyle='--', c='0.7')
# # plt.text(0, 20, 'Threshold=%s' % th,fontsize=6 )
# # plt.text(0, -20, 'Threshold=%s' % -th,fontsize=6 )
# FR_paw, = plt.plot(mouse_time[day][sess][0][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)], mouse_speedDiff[day][sess][0][bounds[0]:bounds[1]][trailingStart:-(trailingEnd+1)] , c='b', label='Front right paw')
# plt.plot(mouse_time[day][sess][0][bounds[0]:bounds[1]], mouse_swing[day][sess][0][1][bounds[0]:bounds[1]], c='slateblue')
# plt.xlabel('Time during recording session(s)')
# plt.ylabel('X speed difference between a paw and the wheel (a.u.)')
# plt.legend(labels=['Positive threshold', 'Negative threshold', 'Stance phases', 'Swing phases'])



# dataAnalysis.plotSpeedDiff(mouse, foldersRecordings[day][0], foldersRecordings[day][2][sess+1], mouse_time[day][sess], mouse_speedDiff[day][sess], mouse_swing[day][sess],trailingStart, trailingEnd, speedDiffThresh, bounds, saveFig=False, showFig=True)


# Histogram of speed profile per day per paw
param = dataAnalysis.plotHist(mouse_speedDiff, bounds, showHist=False)


plt.figure()
param_ttest = np.asarray([])
for p in range(param.shape[1]):
    if p==0:
        color='#1f77b4'
    if p==1:
        color='#ff7f0e'
    if p == 2:
        color = '#2ca02c'
    if p == 3:
        color = '#d62728'
    for v in range(param.shape[2]):
        plt.subplot(1, 4, v+1)
        plt.plot(np.arange(param.shape[0]), param[:, p, v], c=color, marker='o')
    param_ttest = np.append(param_ttest, stats.ttest_rel(param[:6, p, :], param[6:, p, :]))

swing_number = np.empty((len(mouse_swing), len(mouse_swing[0]), len(mouse_swing[0][0])))
for day in range(len(mouse_swing)):
    for sess in range(len(mouse_swing[day])):
        swing_number[day, sess, 0] = len(mouse_swing[day][sess][0][0])
        swing_number[day, sess, 1] = len(mouse_swing[day][sess][1][0])
        swing_number[day, sess, 2] = len(mouse_swing[day][sess][2][0])
        swing_number[day, sess, 3] = len(mouse_swing[day][sess][3][0])

stance_number = np.empty((len(mouse_stance), len(mouse_stance[0]), len(mouse_stance[0][0])))
for day in range(len(mouse_stance)):
    for sess in range(len(mouse_stance[day])):
        stance_number[day, sess, 0] = len(mouse_stance[day][sess][0][0])
        stance_number[day, sess, 1] = len(mouse_stance[day][sess][1][0])
        stance_number[day, sess, 2] = len(mouse_stance[day][sess][2][0])
        stance_number[day, sess, 3] = len(mouse_stance[day][sess][3][0])

plt.figure()
swing_ttest = np.asarray([])
stance_ttest = np.asarray([])
for p in range(swing_number.shape[2]):
    if p==0:
        color='#1f77b4'
    if p==1:
        color='#ff7f0e'
    if p == 2:
        color = '#2ca02c'
    if p == 3:
        color = '#d62728'

    plt.subplot(1, 2, 1)
    plt.plot(range(12), np.mean(swing_number[:, :, p], axis=1), c=color, marker='o')
    plt.ylim(25, 75)
    plt.subplot(1, 2, 2)
    plt.plot(range(12), np.mean(stance_number[:, :, p], axis=1), c=color, marker='o')
    plt.ylim(25, 75)
    swing_ttest = np.append(swing_ttest, stats.ttest_rel(np.mean(swing_number[:6, :, p],axis=1), np.mean(swing_number[6:, :, p], axis=1)))
    stance_ttest = np.append(stance_ttest, stats.ttest_rel(np.mean(stance_number[:6, :, p],axis=1), np.mean(stance_number[6:, :, p], axis=1)))

# plt.figure()
# for day in range(swing_number.shape[0]):
#     plt.subplot(2, 4, 1)
#     plt.scatter([day] * 5, swing_number[day, :, 0], c='0.7')
#
#     plt.scatter([day], np.mean(swing_number[day, :, 0]), c='0')
#     plt.ylim(25, 75)
#     plt.subplot(2, 4, 3)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, swing_number[day, :, 1], c='0.7')
#     plt.scatter([day], np.mean(swing_number[day, :, 1]), c='0')
#     plt.subplot(2, 4, 5)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, swing_number[day, :, 2], c='0.7')
#     plt.scatter([day], np.mean(swing_number[day, :, 2]), c='0')
#     plt.subplot(2, 4, 7)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, swing_number[day, :, 3], c='0.7')
#     plt.scatter([day], np.mean(swing_number[day, :, 3]), c='0')
#
#     plt.subplot(2, 4, 2)
#     plt.ylim(25, 75)
#     plt.scatter([day]*5, stance_number[day,:, 0], c='0.7')
#     plt.scatter([day], np.mean(stance_number[day, :, 0]), c='0')
#     plt.subplot(2, 4, 4)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, stance_number[day, :, 1], c='0.7')
#     plt.scatter([day], np.mean(stance_number[day, :, 1]), c='0')
#     plt.subplot(2, 4, 6)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, stance_number[day, :, 2], c='0.7')
#     plt.scatter([day], np.mean(stance_number[day, :, 2]), c='0')
#     plt.subplot(2, 4, 8)
#     plt.ylim(25, 75)
#     plt.scatter([day] * 5, stance_number[day, :, 3], c='0.7')
#     plt.scatter([day], np.mean(stance_number[day, :, 3]), c='0')





# TTL pulse at 5s (index 1000), wheel accelerate at 7s (index 1400), max speed at 10.8 (index 2160) during 12s, wheel decelerate at 22.8 (index 4560), wheel disconnect at 26.6 (index 5320)
# Speed for 9 cm.s^-1 is 20 on arduino2P, arduinoSetup1, 16 on arduinoSetup2