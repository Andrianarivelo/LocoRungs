from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sh import mount
import numpy as np
from scipy.signal import find_peaks

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

tracking_paths = []

for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range(len(foldersRecordings[f][2])):
        # loop over all ScanImage sessions
        recordingD = []
        for s in range(len(foldersRecordings[f][2][r][1])):
            (existenceDLC2, DLC2_file) = eSD.getDLC2TrackingFiles(mouse, foldersRecordings[f][0], foldersRecordings[f][2][r][0], foldersRecordings[f][2][r][1][s])
            if existenceDLC2:
                recordingD.append(DLC2_file)
        if existenceDLC2:
            tracking_paths.append(recordingD)


paws_data = np.column_stack((pd.read_hdf(tracking_paths[0][0]).values[:, [0, 1, 3, 4, 6, 7, 9, 10]], np.arange(
    pd.read_hdf(tracking_paths[0][0]).values.shape[0])))  # import HDF5 file into numpy array
FR, FL, HL, HR = paws_data[:, [0, 1, 8]], paws_data[:, [2, 3, 8]], paws_data[:, [4, 5, 8]], paws_data[:, [6, 7, 8]]

# Removing outliers frames depending speed displacement

FR_th, FL_th, HL_th, HR_th = FR, FL, HL, HR
bool_dist = False
i = 1
while np.any(bool_dist == False):  # loop until no value above threshold
    dist = []
    print('%snd loop' % i)
    i = i+1
    for paw in FR_th, FL_th, HL_th, HR_th:
        tmp_dist = np.sqrt((np.diff(paw[:, 0])) ** 2 + (np.diff(paw[:, 1])) ** 2) / np.diff(paw[:, 2])
        dist.append(tmp_dist)

    th = 40  # Displacement threshold
    bool_dist = []
    for i in range(len(dist)):
        tmp_bool_dist = dist[i] < th  # boolean array
        tmp_bool_dist = np.insert(tmp_bool_dist, 0, True)  # add a row to harmonize shapes
        bool_dist.append(tmp_bool_dist)

    FR_th = (FR_th.T*bool_dist[0]).T
    FR_th = FR_th[~np.any(FR_th[:, [0, 1]] == 0, axis=1)]

    FL_th = (FL_th.T*bool_dist[1]).T
    FL_th = FL_th[~np.any(FL_th[:, [0, 1]] == 0, axis=1)]

    HL_th = (HL_th.T*bool_dist[2]).T
    HL_th = HL_th[~np.any(HL_th[:, [0, 1]] == 0, axis=1)]

    HR_th = (HR_th.T*bool_dist[3]).T
    HR_th = HR_th[~np.any(HR_th[:, [0, 1]] == 0, axis=1)]

    paws_th = [FR_th, FL_th, HL_th, HR_th]


# Absolute speed over time

speed = []
for paw in FR_th, FL_th, HL_th, HR_th:
    tmp_speed = []
    for i in range(paw.shape[0]-1):
        tmp_speed.append((((paw[i+1, 0]-paw[i, 0])**2+(paw[i+1, 1]-paw[i, 1])**2)**0.5)/(paw[i+1, 2]-paw[i, 2]))
    speed.append(tmp_speed)

# Speed for arduino on 20 is 9 cm.s^-1

# Average stride length (intralimb)

dist, wth = 50, 15  # Good parameters for forepaws
peaks = []
peaks_inv = []
stride_means = []
for i in range(4):
    peaks_tmp, _ = find_peaks(paws_th[i][:, 0], distance=dist, width=wth)
    peaks.append(peaks_tmp)
    peaks_inv_tmp, _ = find_peaks(1/paws_th[i][:, 0], distance=dist, width=wth)
    peaks_inv.append(peaks_inv_tmp)
    stride_means.append(np.mean(paws_th[i][:, 0][peaks[i]]) - np.mean(paws_th[i][:, 0][peaks_inv[i]]))


# Average step length (interlimb)

step_means = np.mean(paws_th[0][:, 0][peaks[0]])-np.mean(paws_th[1][:, 0][peaks[0]]), np.mean(paws_th[1][:, 0][peaks[1]])-np.mean(paws_th[0][:, 0][peaks[1]]), np.mean(paws_th[2][:, 0][peaks[2]])-np.mean(paws_th[3][:, 0][peaks[2]]), np.mean(paws_th[3][:, 0][peaks[3]])-np.mean(paws_th[2][:, 0][peaks[3]])

step_diff = np.diff([FR[:, 0], FL[:, 0]], axis=0), np.diff([HR[:, 0], HL[:, 0]], axis=0)


# Plotting
## Plots for a single session
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

## Plots for multiple sessions
