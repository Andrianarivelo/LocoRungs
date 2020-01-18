import pdb
import sys
import os
import numpy as np
import h5py
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

################################################################################
def layoutOfPanel(ax,xLabel=None,yLabel=None,Leg=None,xyInvisible=[False,False]):


    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #
    if xyInvisible[0]:
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
    else:
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
    #
    if xyInvisible[1]:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_visible(False)
    else:
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')


    if xLabel != None :
        ax.set_xlabel(xLabel)

    if yLabel != None :
        ax.set_ylabel(yLabel)

    if Leg != None :
        ax.legend(loc=Leg[0], frameon=False)
        if len(Leg)>1 :
            legend = ax.get_legend()  # plt.gca().get_legend()
            ltext = legend.get_texts()
            plt.setp(ltext, fontsize=Leg[1])
################################################################


recFolder = '2019.12.09_000'
recordingN = 4

# compare one calibration with another one
compare = False
recFolderCompare = '2019.12.09_000'
recordingNCompare = 1

##################################################################
baseDir = '/media/invivodata2/altair_data/dataMichael/'
dataDirectory = baseDir + recFolder+ '/LaserCalibration_%03d/' % recordingN
compareDataDirectory = baseDir + recFolderCompare+ '/LaserCalibration_%03d/' % recordingNCompare

scalingFactor = 5 # V/W
startMean = 1.2 # in s, starting time to average
endMean = 1.9 # in s, ending time to average
baseline = 0.01 # in s, time for baseline estimation (PC pulse occurs at 0.01 s)

#pdb.set_trace()
# read data from ACQ4 files ####################################
dirs = os.listdir(dataDirectory)
dirs.remove('.index')
recs = len(dirs)
dirs.sort()

DATA = []
for i in range(recs):
    print(dirs[i])
    fNameDaq = dataDirectory+dirs[i] + '/DaqDevice.ma'
    fNamePC = dataDirectory+dirs[i] + '/PockelsCell.ma'
    #pdb.set_trace()

    fDaq = h5py.File(fNameDaq,'r')
    valuesDaq = fDaq['data'][()]
    valueTimesDaq = fDaq['info/1/values'][()]
    fPC = h5py.File(fNamePC,'r')
    valuesPC = fPC['data'][()]
    valueTimesPC = fPC['info/1/values'][()]
    #pdb.set_trace()

    bigArray = np.column_stack((valueTimesDaq,valuesPC[0],valuesDaq[1],valuesDaq[2]))
    DATA.append([dirs[i],bigArray])

if compare :
    dirsC = os.listdir(compareDataDirectory)
    dirsC.remove('.index')
    recsC = len(dirsC)
    dirsC.sort()

    DATAC = []
    for i in range(recsC):
        print(dirsC[i])
        fNameDaq = compareDataDirectory + dirsC[i] + '/DaqDevice.ma'
        fNamePC = compareDataDirectory + dirsC[i] + '/PockelsCell.ma'
        # pdb.set_trace()

        fDaq = h5py.File(fNameDaq, 'r')
        valuesDaq = fDaq['data'][()]
        valueTimesDaq = fDaq['info/1/values'][()]
        fPC = h5py.File(fNamePC, 'r')
        valuesPC = fPC['data'][()]
        valueTimesPC = fPC['info/1/values'][()]
        # pdb.set_trace()

        bigArray = np.column_stack((valueTimesDaq, valuesPC[0], valuesDaq[1], valuesDaq[2]))
        DATAC.append([dirs[i], bigArray])

# extract mean values from plot ####################################

baselineMask = (DATA[0][1][:,0]<baseline)
mask = (DATA[0][1][:,0]>startMean)&(DATA[0][1][:,0]<endMean)
#pockelCellAmps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
laserPower = np.array([0,2.3,9.0,20.6,37,58.6,82,110,141,173,208])
meanMeasurements = np.zeros((recs,4))

for i in range(recs):
    meanPockelCell = np.mean(DATA[i][1][:,1][mask])
    #pdb.set_trace()
    meanPowerMeter = np.mean(DATA[i][1][:,3][mask])
    powerMeterBaseline = np.mean(DATA[i][1][:,3][baselineMask])
    meanPhotoDiode = np.mean(DATA[i][1][:,2][mask])
    meanMeasurements[i] = [meanPockelCell,meanPhotoDiode,meanPowerMeter,powerMeterBaseline]

if compare :
    baselineMask = (DATAC[0][1][:, 0] < baseline)
    mask = (DATAC[0][1][:, 0] > startMean) & (DATAC[0][1][:, 0] < endMean)
    # pockelCellAmps = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    meanMeasurementsCompare = np.zeros((recs, 4))

    for i in range(recs):
        meanPockelCell = np.mean(DATAC[i][1][:, 1][mask])
        # pdb.set_trace()
        meanPowerMeter = np.mean(DATAC[i][1][:, 3][mask])
        powerMeterBaseline = np.mean(DATAC[i][1][:, 3][baselineMask])
        meanPhotoDiode = np.mean(DATAC[i][1][:, 2][mask])
        meanMeasurementsCompare[i] = [meanPockelCell, meanPhotoDiode, meanPowerMeter, powerMeterBaseline]

# plot data ########################################################

fig_width = 12  # width in inches
fig_height = 12  # height in inches
fig_size = [fig_width, fig_height]
params = {'axes.labelsize': 12,
          'axes.titlesize': 13,
          'font.size': 11,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'figure.figsize': fig_size,
          'savefig.dpi': 600,
          'axes.linewidth': 1.3,
          'ytick.major.size': 4,  # major tick size in points
          'xtick.major.size': 4  # major tick size in points
          # 'edgecolor' : None
          # 'xtick.major.size' : 2,
          # 'ytick.major.size' : 2,
          }
rcParams.update(params)

# set sans-serif font to Arial
rcParams['font.sans-serif'] = 'Arial'

# create figure instance
fig = plt.figure()

# define sub-panel grid and possibly width and height ratios
gs = gridspec.GridSpec(3, 3  # ,
                       # width_ratios=[1.2,1]
                       # height_ratios=[1,1]
                       )

# define vertical and horizontal spacing between panels
gs.update(wspace=0.3, hspace=0.3)

# possibly change outer margins of the figure
plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.07)

# sub-panel enumerations
if compare:
    plt.figtext(0.06, 0.96, 'LaserCalibration %s rec%03d (compared with %s rec%03d)' % (recFolder,recordingN,recFolderCompare,recordingNCompare),clip_on=False,color='black', weight='bold',size=16)
else:
    plt.figtext(0.06, 0.96, 'LaserCalibration %s rec%03d' % (recFolder,recordingN),clip_on=False,color='black', weight='bold',size=16)

# third sub-plot #######################################################
# gssub1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], hspace=0.2)
# sub-panel 1 #############################################
#gssub = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], hspace=0.2)
ax0 = plt.subplot(gs[0])
ax0.set_title('Pockel cell command')
for i in range(recs):
    ax0.plot(DATA[i][1][:,0],DATA[i][1][:,1])

layoutOfPanel(ax0,xLabel='time (s)',yLabel='Command Voltage (V)')

ax1 = plt.subplot(gs[1])
ax1.set_title('PhotoDiode recording')
for i in range(recs):
    ax1.plot(DATA[i][1][:,0],DATA[i][1][:,2])
layoutOfPanel(ax1,xLabel='time (s)',yLabel='Voltage (V)')

ax2 = plt.subplot(gs[2])
ax2.set_title('Optical Power meter recording')
for i in range(recs):
    ax2.plot(DATA[i][1][:,0],DATA[i][1][:,3])
layoutOfPanel(ax2,xLabel='time (s)',yLabel='Voltage (V)')

#ax3 = plt.subplot(gs[3])
#ax3.plot(pockelCellAmps,meanMeasurements[:,0],'o-')
#ax3.grid(True)
#layoutOfPanel(ax3,xLabel='set pockel voltage (V)',yLabel='recorded pockel voltage (V)')

ax4 = plt.subplot(gs[3])
ax4.set_title(r'Pockel cell $\rightarrow$ Photodiode')
ax4.plot(meanMeasurements[:,0],meanMeasurements[:,1],'o-',label='%s rec%03d' % (recFolder,recordingN))
if compare:
    ax4.plot(meanMeasurementsCompare[:, 0], meanMeasurementsCompare[:, 1], 'o-',label='%s rec%03d' %(recFolderCompare,recordingNCompare))
ax4.grid(True)
if compare:
    layoutOfPanel(ax4,xLabel='pockel voltage (V)',yLabel='recorded photodiode voltage (V)',Leg=[2,9])
else:
    layoutOfPanel(ax4,xLabel='pockel voltage (V)',yLabel='recorded photodiode voltage (V)')

ax5 = plt.subplot(gs[4])
ax5.set_title(r'Photodiode $\rightarrow$ Power Meter')
ax5.plot(meanMeasurements[:,1],meanMeasurements[:,2],'o-')
if compare:
    ax5.plot(meanMeasurementsCompare[:, 1], meanMeasurementsCompare[:, 2], 'o-')
ax5.grid(True)
layoutOfPanel(ax5,xLabel='recorded photodiode voltage (V)',yLabel='recorded power meter voltage (V)')

ax6 = plt.subplot(gs[5])
ax6.set_title(r'Pockel cell $\rightarrow$ Power Meter')
ax6.plot(meanMeasurements[:,0],meanMeasurements[:,2],'o-')
if compare:
    ax6.plot(meanMeasurementsCompare[:, 0], meanMeasurementsCompare[:, 2], 'o-')
ax6.grid(True)
layoutOfPanel(ax6,xLabel='pockel voltage (V)',yLabel='recorded power meter voltage (V)')


ax6 = plt.subplot(gs[7])
ax6.plot(meanMeasurements[:,1],(meanMeasurements[:,2]-meanMeasurements[:,3])/scalingFactor,'o-')
if compare:
    ax6.plot(meanMeasurementsCompare[:, 1], (meanMeasurementsCompare[:, 2] - meanMeasurementsCompare[:, 3]) / scalingFactor, 'o-')
#ax6.plot(meanMeasurements[:,0],laserPower/1000.,'o')
ax6.grid(True)
layoutOfPanel(ax6,xLabel='recorded photodiode voltage (V)',yLabel='laser power under objective (W)')

# ax7 = plt.subplot(gs[7])
# ax7.plot(meanMeasurements[:,0],meanMeasurements[:,1],'o-')
# ax7.grid(True)
# layoutOfPanel(ax7,xLabel='recorded pockel voltage (V)',yLabel='recorded PhotoDiode voltage (V)')

ax8 = plt.subplot(gs[8])
ax8.plot(meanMeasurements[:,0],(meanMeasurements[:,2]-meanMeasurements[:,3])/scalingFactor,'o-')
if compare:
    ax8.plot(meanMeasurementsCompare[:, 0], (meanMeasurementsCompare[:, 2] - meanMeasurementsCompare[:, 3]) / scalingFactor, 'o-')
#ax8.plot(meanMeasurements[:,0],laserPower/1000.,'o',label='read from power meter software')
#ax8.plot(pockelCellAmps,meanMeasurements[:,0],'o-')
ax8.grid(True)
layoutOfPanel(ax8,xLabel='pockel cell voltage (V)',yLabel='laser power under objective (W)')

plt.savefig('LaserCalibration_%s_rec%03d.pdf' % (recFolder,recordingN))
#pdb.set_trace()

