import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import pdb
from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

mouseD = '201008_t00' # id of the mouse to analyze
#mouseD = '190108_m24'
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


#print mouse, expDate
#sys.exit(0) #pdb.set_trace()
eSD         = extractSaveData.extractSaveData(mouse)  # find data folder of specific mouse, create data folder, and hdf5 handle
#pdb.set_trace()
(foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse,expDate=expDate,recordings=recordings) # get recordings for specific mouse and date
dataDirs = []
allTiffs = []
allRawRecs = []
imagesRead = False
#dA      = dataAnalysis.dataAnalysis()
for f in range(len(foldersRecordings)):
    for r in range(len(foldersRecordings[f][2])):
        (existenceImaging, tiffList) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2][r], 'SICaImaging')
        (existenceGeneric, fileHandleGeneric) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'PreAmpInput')
        # pdb.set_trace()
        # tiffList = tiffList[-7:-1]
        # if camera was recorded
        if existenceImaging and not imagesRead:
            # pdb.set_trace()
            dataDirs.append(eSD.dataBase2 + foldersRecordings[f][0] + '/')
            allTiffs.extend(tiffList)

            print('analysis on :', tiffList[2:], len(tiffList[2:]))
            imgData = eSD.getRawCalciumImagingData(foldersRecordings[f][0], tiffList[2:])
            imagesRead = True
        if existenceGeneric:
            #print(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r])
            # (foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r],'frameTimes',fileHandleFTimes)
            (values, vTimes) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][1],foldersRecordings[f][2][r], 'PreAmpInput', fileHandleGeneric)
            allRawRecs.append([vTimes,values])
        #(existence,fileHandle) = eSD.checkIfDeviceWasRecorded(foldersRecordings[f][0],foldersRecordings[f][2][r],'Imaging')
        #if existence:
        #    (frames,fTimes,imageMetaInfo) = eSD.readRawData(foldersRecordings[f][0],foldersRecordings[f][2][r],'Imaging',fileHandle)
        #    eSD.saveImageStack(frames,fTimes,imageMetaInfo,[foldersRecordings[f][0],foldersRecordings[f][2][r],'raw_imaging_data'])
        #    eSD.saveTif(frames[:,:,:,0], mouse,foldersRecordings[f][0],foldersRecordings[f][2][r]) # tif file for possible image registration in ImageJ
        #break
    #break

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

f = plt.figure()

gs = gridspec.GridSpec(4, 3,
                       #width_ratios=[1,2],
                       #height_ratios=[4,1]
                       )

#plt.imshow(np.average(imgData[0][1][2::],axis=0))
ax10 = plt.subplot(gs[0])
ax10.set_title('Jorge\'s power-supply \n average image')
ax20 = plt.subplot(gs[1])
ax20.set_title('Hamamatsu\'s power-supply \n average image')
ax30 = plt.subplot(gs[2])
ax30.set_title('Arduino\'s power-supply \n average image')
ax11 = plt.subplot(gs[3])
ax11.set_title('single image')
ax21 = plt.subplot(gs[4])
ax21.set_title('single image')
ax31 = plt.subplot(gs[5])
ax31.set_title('single image')

#ax22 = plt.subplot(gs[7])
#ax32 = plt.subplot(gs[8])

jorgePS = 218
hamaPS = 231
arduPS = 234

widthh = 60
#pdb.set_trace()

ax10.imshow(np.average(imgData[0][1][::2],axis=0))
ax10.axhline(y=jorgePS)
ax10.axhline(y=jorgePS+widthh)
ax20.imshow(np.average(imgData[1][1][::2],axis=0))
ax20.axhline(y=hamaPS)
ax20.axhline(y=hamaPS+widthh)
ax30.imshow(np.average(imgData[3][1][::2],axis=0))
ax30.axhline(y=arduPS)
ax30.axhline(y=arduPS+widthh)

#pdb.set_trace()
ax11.imshow(imgData[0][1][::2][-1])
ax11.axhline(y=jorgePS)
ax11.axhline(y=jorgePS+widthh)
ax21.imshow(imgData[1][1][::2][-1])
ax21.axhline(y=hamaPS)
ax21.axhline(y=hamaPS+widthh)
ax31.imshow(imgData[3][1][::2][-1])
ax31.axhline(y=arduPS)
ax31.axhline(y=arduPS+widthh)

ax12 = plt.subplot(gs[6])
ax12.set_title('cuts through avg. image')
ax12.plot(np.average(np.average(imgData[0][1][::2],axis=0)[:,jorgePS:(jorgePS+widthh)],axis=1),c='C0',label='jorge')
#ax12.plot(np.average(np.average(imgData[1][1][::2],axis=0)[:,jorgePS:(jorgePS+30)],axis=1),c='C0')
ax12.plot(np.average(np.average(imgData[1][1][::2],axis=0)[:,hamaPS:(hamaPS+widthh)],axis=1),c='C1',label='hamamatsu')
#ax12.plot(np.average(np.average(imgData[3][1][::2],axis=0)[:,jorgePS:(jorgePS+30)],axis=1),c='C0')
ax12.plot(np.average(np.average(imgData[3][1][::2],axis=0)[:,arduPS:(arduPS+widthh)],axis=1),c='C2',label='arduino')

ax12.legend()

###########################################
ax22 = plt.subplot(gs[7])
#ax22.plot(np.average(imgData[0][1][::2][-1][:,jorgePS:(jorgePS+30)],axis=1),c='C0',label='jorge')
#ax22.plot(np.average(imgData[2][1][::2][-1][:,hamaPS:(hamaPS+30)],axis=1),c='C1',label='hamamatsu')
#ax22.plot(np.average(imgData[4][1][::2][-1][:,arduPS:(arduPS+30)],axis=1),c='C2',label='arduino')
ax12.set_title('normalized cuts through single image')
avgJ = np.average(imgData[0][1][::2][-1][:,jorgePS:(jorgePS+widthh)],axis=1)
avgH = np.average(imgData[1][1][::2][-1][:,hamaPS:(hamaPS+widthh)],axis=1)
avgA = np.average(imgData[3][1][::2][-1][:,arduPS:(arduPS+widthh)],axis=1)

sigJ = (avgJ - np.mean(avgJ[:200]))/np.std(avgJ[:200])
sigH = (avgH - np.mean(avgH[:200]))/np.std(avgH[:200])
sigA = (avgA - np.mean(avgA[:200]))/np.std(avgA[:200])

#pdb.set_trace()

ax22.plot(sigJ,c='C0',label='jorge')
ax22.plot(sigH,c='C1',label='hamamatsu')
ax22.plot(sigA,c='C2',label='arduino')

ax22.legend()

###########################################
ax13 = plt.subplot(gs[9])
ax13.set_title('raw data (out of pre-AMP) trace')
recIdx = 3

ax13.plot(allRawRecs[0][0],allRawRecs[0][1][recIdx])

ax23 = plt.subplot(gs[10])
ax23.set_title('raw data (out of pre-AMP) trace')

ax23.plot(allRawRecs[1][0],allRawRecs[1][1][recIdx])

ax33 = plt.subplot(gs[11])

ax33.plot(allRawRecs[2][0],allRawRecs[2][1][recIdx])
ax33.set_title('raw data (out of pre-AMP) trace')

##################################################
ax13 = plt.subplot(gs[8])
ax13.set_title('histogram of raw data')

jorgeData = np.copy(allRawRecs[0][1][recIdx])
hamaData = np.copy(allRawRecs[1][1][recIdx])
arduiData = np.copy(allRawRecs[2][1][recIdx])

#pdb.set_trace()
binss = np.linspace(-0.062,0.004,200)
ax13.hist(jorgeData,bins=binss,histtype='step',label='jorge')
ax13.hist(hamaData,bins=binss,histtype='step',label='hamamatsu')
ax13.hist(arduiData,bins=binss,histtype='step',label='arduino')

ax13.set_yscale('log')
plt.legend(loc=2)

#ax11.plot(np.average(np.average(imgData[0][1][2::],axis=0)[:,215:235],axis=1),c='C0')
#ax11.plot(np.average(np.average(imgData[1][1][2::],axis=0)[:,215:235],axis=1),c='C0')
#plt.plot(np.average(np.average(imgData[2][1][2::],axis=0)[:,215:235],axis=1),c='C1')
#plt.plot(np.average(np.average(imgData[3][1][2::],axis=0)[:,215:235],axis=1),c='C1')
#plt.plot(np.average(np.average(imgData[4][1][2::],axis=0)[:,215:235],axis=1),c='C2')
#plt.plot(np.average(np.average(imgData[5][1][2::],axis=0)[:,215:235],axis=1),c='C2')
plt.show()
pdb.set_trace()