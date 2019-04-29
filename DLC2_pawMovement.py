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
for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    recordingsD = []
    for r in range(len(foldersRecordings[f][2])):
        (existencePawPos, PawFileHandle) = eSD.checkIfPawPositionWasExtracted(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '-' +foldersRecordings[f][2][r][-3:])
        if existencePawPos:
            (pawPositions,pawMetaData) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r][:-4] + '-' + foldersRecordings[f][2][r][-3:],'pawTraces',PawFileHandle)
            pawTrackingOutliers = dataAnalysis.detectPawTrackingOutlies(pawPositions,pawMetaData,showFig=False)
            paws_data = np.column_stack((pd.read_hdf(PawFileHandle).values[:, [0, 1, 3, 4, 6, 7, 9, 10]], np.arange(pd.read_hdf(PawFileHandle).values.shape[0])))
            (FR_th, FL_th, HL_th, HR_th) = paws_data[pawTrackingOutliers[0][3],:][:, [0, 1, 8]], paws_data[pawTrackingOutliers[1][3],:][:, [2, 3, 8]], paws_data[pawTrackingOutliers[2][3],:][:, [4, 5, 8]], paws_data[pawTrackingOutliers[3][3],:][:, [6, 7, 8]]
            recordingsD.append([FR_th, FL_th, HL_th, HR_th])
    recordingsM.append(recordingsD)
del paws_data, FR_th, FL_th, HL_th, HR_th, recordingsD

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
mouse_peaks = []
peaks_inv = []
stride_means = []
for d in range(len(recordingsM)):
    day_peaks = []
    for s in range(len(recordingsM[d])):
        sess_peaks = []
        sess_stride = []
        for p in range(len(recordingsM[d][s])):
            paw_peaks, _ = find_peaks(recordingsM[d][s][p][:, 0], distance=dist, width=wth)
            paw_peaks_inv, _ = find_peaks(1/recordingsM[d][s][p][:, 0], distance=dist, width=wth)
            sess_peaks.append(np.sort(np.concatenate((paw_peaks, paw_peaks_inv))))
            sess_stride.append(np.mean(recordingsM[d][s][p][:, 0][peaks[p]]) - np.mean(recordingsM[d][s][p][:, 0][peaks_inv[p]]))

# #########################################################
#
# # Average step length (interlimb)
#
# step_means = np.mean(paws_th[0][:, 0][peaks[0]])-np.mean(paws_th[1][:, 0][peaks[0]]), np.mean(paws_th[1][:, 0][peaks[1]])-np.mean(paws_th[0][:, 0][peaks[1]]), np.mean(paws_th[2][:, 0][peaks[2]])-np.mean(paws_th[3][:, 0][peaks[2]]), np.mean(paws_th[3][:, 0][peaks[3]])-np.mean(paws_th[2][:, 0][peaks[3]])
#
# step_diff = np.diff([FR[:, 0], FL[:, 0]], axis=0), np.diff([HR[:, 0], HL[:, 0]], axis=0)
#
# #########################################################
#
# # Get session speed from rotary encoder and video exposure data
# tracks = []
# for f in range(len(foldersRecordings)):
#     subtracks = []
#     for r in range(1, len(foldersRecordings[f][2])):
#         (existence, fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r][:-4] + '/' + foldersRecordings[f][2][r][-3:], 'RotaryEncoder')
#         if existence:
#             (angularSpeed,linearSpeed,wTimes,startTime,monitor,angleTimes) = eSD.getWalkingActivity([foldersRecordings[f][0], foldersRecordings[f][2][r], 'walking_activity'])
#             (startExposure, endExposure) = eSD.getBehaviorVideoData([foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video'])
#             subtracks.append([foldersRecordings[f][2][r], linearSpeed, wTimes, startExposure, endExposure])
#     tracks.append(subtracks)
#
#########################################################

# Get day's overall activity


# # #########################################################
# #
# # # Plots for a single session
# # # row = 3
# # # col = 4
# # # plt.figure()
# # # plt.subplot(row, col, 1)
# # # plt.plot(speed[0])
# # # plt.title("FR")
# # #
# # # plt.subplot(row, col, 2)
# # # plt.plot(speed[1])
# # # plt.title("FL")
# # #
# # # plt.subplot(row, col, 3)
# # # plt.plot(speed[2])
# # # plt.title("HL")
# # #
# # # plt.subplot(row, col, 4)
# # # plt.plot(speed[3])
# # # plt.title("HR")
# # #
# # # for i in range(4):
# # #     plt.subplot(row, col, i+5)
# # #     plt.plot(paws_th[i][:, 0])
# # #     plt.plot(peaks[i], paws_th[i][:, 0][peaks[i]], 'x')
# # #     plt.plot(peaks_inv[i], paws_th[i][:, 0][peaks_inv[i]], 'o')
# # #
# # # plt.subplot(row, 2, 5)
# # # plt.plot(step_diff[0][0, :])
# # #
# # # plt.subplot(row, 2, 6)
# # # plt.plot(step_diff[1][0, :])
# #
# # #########################################################
# #
# # # Plots for multiple sessions
