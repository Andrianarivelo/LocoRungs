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
total_of = 0
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    recordingsD = []
    day_OF = []
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
# Absolute speed over time

mouse_speed = dataAnalysis.getPawSpeed(recordingsM)

##########################################################
# Average stride length (intralimb)

mouse_stride, mouse_peaks, mouse_peaks_inv = dataAnalysis.getStride(recordingsM)
    # frame index have to be determined for peaks
#########################################################
# Average step length (interlimb)

mouse_step = dataAnalysis.getStep(recordingsM, mouse_peaks)

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
row = 3
col = 4
plt.figure()
plt.subplot(row, col, 1)
plt.plot(mouse_speed[0][0][0])
plt.title("FR")

plt.subplot(row, col, 2)
plt.plot(mouse_speed[0][0][1])
plt.title("FL")

plt.subplot(row, col, 3)
plt.plot(mouse_speed[0][0][2])
plt.title("HL")

plt.subplot(row, col, 4)
plt.plot(mouse_speed[0][0][3])
plt.title("HR")

for i in range(len(recordingsM[0][0])):
    plt.subplot(row, col, i+5)
    plt.plot(recordingsM[0][0][i][:, 0])
    plt.plot(mouse_peaks[0][0][i], recordingsM[0][0][i][:, 0][mouse_peaks[0][0][i]], 'x')
    plt.plot(mouse_peaks_inv[0][0][i], recordingsM[0][0][i][:, 0][mouse_peaks_inv[0][0][i]], 'o')

plt.subplot(row, 1, 3)
for i in range(len(recordingsM[0][0])):
    plt.plot(mouse_tracks[0][i][2][mouse_OF[0][0][i][3]], recordingsM[0][0][i][:, 0])
plt.plot(mouse_tracks[0][0][1], mouse_tracks[0][0][0])

#########################################################

# Plots for multiple sessions


# TTL pulse at 5s (index 1000), wheel accelerate at 7s (index 1400), max speed at 10.8 (index 2160) during 12s, wheel decelerate at 22.8 (index 4560), wheel disconnect at 26.6 (index 5320)
# Speed for 9 cm.s^-1 is 20 on arduino2P, arduinoSetup1, 16 on arduinoSetup2