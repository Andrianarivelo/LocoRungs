from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
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

recordingsM = []
mouse_OF = [] # OF = Outliers Frames
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    recordingsD = []
    day_OF = []
    for r in range(len(foldersRecordings[f][2])):
        (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '-' +foldersRecordings[f][2][r][-3:])
        if existencePawPos:
            (pawPositions,pawMetaData) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r][:-4] + '-' + foldersRecordings[f][2][r][-3:],'pawTraces',PawFileHandle)
            pawTrackingOutliers = dataAnalysis.detectPawTrackingOutlies(pawPositions,pawMetaData,showFig=False)
            (FR_th, FL_th, HL_th, HR_th) = pawPositions[pawTrackingOutliers[0][3],:][:, [1, 2, 0]], pawPositions[pawTrackingOutliers[1][3],:][:, [4, 5, 0]], pawPositions[pawTrackingOutliers[2][3],:][:, [7, 8, 0]], pawPositions[pawTrackingOutliers[3][3],:][:, [10, 11, 0]] # Remove outlier data
            # (FR_th, FL_th, HL_th, HR_th) = FR_th[(FR_th[:, 2] > 2160) & (FR_th[:, 2] < 4560)], FL_th[(FL_th[:, 2] > 2160) & (FL_th[:, 2] < 4560)], HL_th[(HL_th[:, 2] > 2160) & (HL_th[:, 2] < 4560)], HR_th[(HR_th[:, 2] > 2160) & (HR_th[:, 2] < 4560)]# Keep data at max wheel speed
            day_OF.append(pawTrackingOutliers)
            recordingsD.append([FR_th, FL_th, HL_th, HR_th])
    mouse_OF.append(day_OF)
    recordingsM.append(recordingsD)
# del FR_th, FL_th, HL_th, HR_th, recordingsD

#########################################################

# Absolute speed over time

mouse_speed = []
for d in range(len(recordingsM)):
    day_speed = []
    for s in range(len(recordingsM[d])):
        sess_speed = []
        for p in range(len(recordingsM[d][s])):
            paw_speed = np.sqrt((np.diff(recordingsM[d][s][p][:, 0])) ** 2 + (np.diff(recordingsM[d][s][p][:, 1])) ** 2) / np.diff(recordingsM[d][s][p][:, 2])
            sess_speed.append(paw_speed)
        day_speed.append(sess_speed)
    mouse_speed.append(day_speed)
del day_speed, sess_speed, paw_speed

# Speed for 9 cm.s^-1 is 20 on arduino2P, arduinoSetup1, 16 on arduinoSetup2

##########################################################

# Average stride length (intralimb)

dist, wth = 50, 15  # Good parameters for forepaws
mouse_stride = []
mouse_peaks = []
mouse_peaks_inv = []
for d in range(len(recordingsM)):
    day_stride = []
    day_peaks = []
    day_peaks_inv = []
    for s in range(len(recordingsM[d])):
        sess_stride = []
        sess_peaks = []
        sess_peaks_inv = []
        for p in range(len(recordingsM[d][s])):
            paw_peaks, _ = find_peaks(recordingsM[d][s][p][:, 0], distance=dist, width=wth)
            paw_peaks_inv, _ = find_peaks(1/recordingsM[d][s][p][:, 0], distance=dist, width=wth)
            sess_peaks.append(paw_peaks)
            sess_peaks_inv.append(paw_peaks_inv)
            sess_stride.append(np.mean(recordingsM[d][s][p][:, 0][paw_peaks]) - np.mean(recordingsM[d][s][p][:, 0][paw_peaks_inv]))
        day_stride.append(sess_stride)
        day_peaks.append(sess_peaks)
        day_peaks_inv.append(sess_peaks_inv)
    mouse_stride.append(day_stride)
    mouse_peaks.append(day_peaks)
    mouse_peaks_inv.append(day_peaks_inv)
    # frame index have to be determined for peaks
#########################################################

# Average step length (interlimb)

# bounds have to be determined

try :
    mouse_step = []
    for d in range(len(recordingsM)):
        day_step = []
        for s in range(len(recordingsM[d])):
            day_step.append([np.mean(recordingsM[d][s][0][:, 0][mouse_peaks[d][s][0]])-np.mean(recordingsM[d][s][1][:, 0][mouse_peaks[d][s][0]]), np.mean(recordingsM[d][s][1][:, 0][mouse_peaks[d][s][1]])-np.mean(recordingsM[d][s][0][:, 0][mouse_peaks[d][s][1]]), np.mean(recordingsM[d][s][2][:, 0][mouse_peaks[d][s][2]])-np.mean(recordingsM[d][s][3][:, 0][mouse_peaks[d][s][2]]), np.mean(recordingsM[d][s][3][:, 0][mouse_peaks[d][s][3]])-np.mean(recordingsM[d][s][2][:, 0][mouse_peaks[d][s][3]])])
except IndexError:
    print(IndexError)

# step_diff = np.diff([FR[:, 0], FL[:, 0]], axis=0), np.diff([HR[:, 0], HL[:, 0]], axis=0)

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
            day_tracks.append([linearSpeed, wTimes, startExposure, endExposure])
    mouse_tracks.append(day_tracks)

#########################################################

# Get day's overall activity
mouse_activity = []
for f in range(len(foldersRecordings)):
    (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][0], 'RotaryEncoder')
    if existence:
        (angularSpeed,linearSpeed,wTimes,startTime,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][0], 'walking_activity'])
        mouse_activity.append([linearSpeed, wTimes])

#########################################################

# Plots for a single session
# plt.plot(mouse_tracks[0][0][1],mouse_tracks[0][0][0]*10, mouse_tracks[0][0][2][mouse_OF[0][0][0][3]], recordingsM[0][0][0][:, 0])
# row = 3
# col = 4
# plt.figure()
# plt.subplot(row, col, 1)
# plt.plot(speed[0])
# plt.title("FR")
#
# plt.subplot(row, col, 2)
# plt.plot(speed[1])
# plt.title("FL")
#
# plt.subplot(row, col, 3)
# plt.plot(speed[2])
# plt.title("HL")
#
# plt.subplot(row, col, 4)
# plt.plot(speed[3])
# plt.title("HR")
#
# for i in range(4):
#     plt.subplot(row, col, i+5)
#     plt.plot(paws_th[i][:, 0])
#     plt.plot(peaks[i], paws_th[i][:, 0][peaks[i]], 'x')
#     plt.plot(peaks_inv[i], paws_th[i][:, 0][peaks_inv[i]], 'o')
#
# plt.subplot(row, 2, 5)
# plt.plot(step_diff[0][0, :])
#
# plt.subplot(row, 2, 6)
# plt.plot(step_diff[1][0, :])

#########################################################

# Plots for multiple sessions

# TTL pulse at 5s (index 1000), wheel accelerate at 7s (index 1400), max speed at 10.8 (index 2160) during 12s, wheel decelerate at 22.8 (index 4560), wheel disconnect at 26.6 (index 5320)
for i in range(len(mouse_stride)):
    for j in range(len(mouse_stride[i])):
        plt.scatter(i/(1+j), mouse_stride[i][j][0])