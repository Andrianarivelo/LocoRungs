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
expDateD = 'some'     # specific date e.g. '180214', 'some' for manual selection or 'all'
recordings='some'     # 'all or 'some'


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
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date

for f in range(len(foldersRecordings)):
    # loop over all recordings in that folder
    for r in range(len(foldersRecordings[f][2])):
        (existenceFrames,fileHandleFrames) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'CameraGigEBehavior')
        (existenceTracking, DLC2_Data) = eSD.checkIfDLC2TrackingDone(mouse, expDate, recordings)
        if existenceFrames and existenceTracking:
            paws_data = np.column_stack((pd.read_hdf(DLC2_Data).values[:, [0, 1, 3, 4, 6, 7, 9, 10]], np.arange(
                pd.read_hdf(DLC2_Data).values.shape[0])))  # import HDF5 file into numpy array
            FR, FL, HR, HL = paws_data[:, [0, 1]], paws_data[:, [2, 3]], paws_data[:, [4, 5]], paws_data[:, [6, 7]]

            # Removing outliers frames along x euclidean distance threshold

            dist_x = []
            for paw in FR, FL, HR, HL:
                tmp_dist_x = []
                for i in range(paw.shape[0]-1):
                    tmp_dist_x.append(((paw[i+1, 0]-paw[i, 0])**2+1)**0.5)
                dist_x.append(tmp_dist_x)
            dist_x = np.asarray(dist_x).transpose()


            th = 14
            bool_x = dist_x < th  # boolean array
            bool_x = np.insert(bool_x, 0, [True, True, True, True], axis=0)  # add a row to harmonize shapes
            bounds = range(0, 6003)  # Need to be better defined
            FR_th = np.asarray([FR[bounds, 0]*bool_x[bounds, 0], FR[bounds, 1], paws_data[bounds, 8]]).transpose()
            FR_th = FR_th[~np.any(FR_th[:, [0, 1]] == 0, axis=1)]

            FL_th = np.asarray([FL[bounds, 0]*bool_x[bounds, 1], FL[bounds, 1], paws_data[bounds, 8]]).transpose()
            FL_th = FL_th[~np.any(FL_th[:, [0, 1]] == 0, axis=1)]

            HR_th = np.asarray([HR[bounds, 0]*bool_x[bounds, 2], HR[bounds, 1], paws_data[bounds, 8]]).transpose()
            HR_th = HR_th[~np.any(HR_th[:, [0, 1]] == 0, axis=1)]

            HL_th = np.asarray([HL[bounds, 0]*bool_x[bounds, 3], HL[bounds, 1], paws_data[bounds, 8]]).transpose()
            HL_th = HL_th[~np.any(HL_th[:, [0, 1]] == 0, axis=1)]

            paws_th = [FR_th, FL_th, HR_th, HL_th]


            # Absolute speed over time

            speed = []
            for paw in FR_th, FL_th, HR_th, HL_th:
                tmp_speed = []
                for i in range(paw.shape[0]-1):
                    tmp_speed.append((((paw[i+1, 0]-paw[i, 0])**2+(paw[i+1, 1]-paw[i, 1])**2)**0.5)/(paw[i+1, 2]-paw[i, 2]))
                speed.append(tmp_speed)

            # Speed for arduino on 25 is 11.16 cm.s^-1

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
            row = 3
            col = 4
            plt.figure()
            plt.subplot(row, col, 1)
            plt.plot(speed[0])
            plt.title("FR")

            plt.subplot(row, col, 2)
            plt.plot(speed[1])
            plt.title("FL")

            plt.subplot(row, col, 3)
            plt.plot(speed[2])
            plt.title("HR")

            plt.subplot(row, col, 4)
            plt.plot(speed[3])
            plt.title("HL")

            for i in range(4):
                plt.subplot(row, col, i+5)
                plt.plot(paws_th[i][:, 0])
                plt.plot(peaks[i], paws_th[i][:, 0][peaks[i]], 'x')
                plt.plot(peaks_inv[i], paws_th[i][:, 0][peaks_inv[i]], 'o')

            plt.subplot(row, 2, 5)
            plt.plot(step_diff[0][0, :])

            plt.subplot(row, 2, 6)
            plt.plot(step_diff[1][0, :])