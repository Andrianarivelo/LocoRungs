import platform
import h5py
from skimage import io
import tifffile as tiff
import numpy as np
import glob
# import sima
# import sima.segment
import time
import pdb
import cv2
import pickle
import re
import matplotlib.pyplot as plt
from skimage import io
#import sys
#import os

from tools.h5pyTools import h5pyTools
import tools.googleDocsAccess as googleDocsAccess
from tools.pyqtgraph.configfile import *
from ScanImageTiffReader import ScanImageTiffReader


class extractSaveData:
    def __init__(self, mouse):
        self.mouse = mouse
        self.h5pyTools = h5pyTools()

        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' or platform.node() == 'thinkpadX1B':
            laptop = True
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/mgraupe/.miniconda2/envs/suite2p/bin/python'
        elif platform.node() == 'otillo':
            laptop = False
            self.analysisBase = '/media/paris_data/'
            self.suite2pPath = '/home/mgraupe/anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'yamal' or platform.node() == 'cerebellum-HP' or platform.node() == 'andry-ThinkPad-X1-Carbon-2nd' or platform.node() == 'OptiPlex-7070':
            laptop = False
            self.analysisBase = '/media/HDnyc_data/'
            self.suite2pPath = '/home/andry/anaconda3/envs/suite2p/bin/python'
        elif platform.node() == 'bs-analysis':  # andry-ThinkPad-X1-Carbon-2nd
            laptop = False
            self.analysisBase = '/home/mgraupe/nyc_data/'
        else:
            print('Run this script on a server or laptop. Otherwise, adapt directory locations.')
            sys.exit(1)
        # print('test1')
        self.listOfAllExpts = googleDocsAccess.getExperimentSpreadsheet()
        # print(self.listOfAllExpts['201007_t00'])
        # extract recording dates of the specific animal
        dates = []
        for d in self.listOfAllExpts[self.mouse]['dates']:
            dates.append(d)

        if dates[0] >= '181018':
            self.dataBase = '/media/invivodata2/'
        else:
            self.dataBase = '/media/invivodata/'

        # check if directory is mounted
        if not os.listdir(self.dataBase):
            os.system('mount %s' % self.dataBase)

        if not os.listdir(self.analysisBase):
            os.system('mount %s' % self.analysisBase)

        self.dataPCLocation = OrderedDict(
            [('behaviorPC', 'behaviorPC_data/dataMichael/'), ('2photonPC', ('altair_data/dataMichael/' if int(self.mouse[:6]) >= 170829 else 'altair_data/experiments/data_Michael/acq4/')),
                ('', ('altair_data/dataMichael/' if int(self.mouse[:6]) >= 170829 else 'altair_data/experiments/data_Michael/acq4/')), ])

        self.computerDict = OrderedDict(
            [('behaviorPC', 'behaviorPC'), ('2photonPC', '2photonPC'),('', '2photonPC')])

        self.analysisLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % mouse
        self.figureLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/%s/' % mouse

        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.analysisLocation)
        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.figureLocation)

        fName = self.analysisLocation + 'analysis.hdf5'
        # if os.path.isfile(fName):
        self.f = h5py.File(fName, 'a')
        self.mouse = mouse

        # experiments stored under this name were recorded as sequence with a certain number of repetitions
        self.listOfSequenceExperiments = ['locomotionTriggerSIAndMotor', 'locomotionTriggerSIAndMotorJin', 'locomotionTriggerSIAndMotor60sec', 'locomotion_recording_setup2']

    ############################################################
    def __del__(self):
        self.f.flush()
        # self.f.close()
        print('on exit')

    ############################################################
    def getMotioncorrectedStack(self, folder, rec, suffix):
        allFiles = []
        for file in glob.glob(self.analysisLocation + '%s_%s_%s*_%s.tif' % (self.mouse, folder, rec, suffix)):
            allFiles.append(file)
            print(file)
        if len(allFiles) > 1:
            print('more than one matching image file')
            sys.exit(1)
        else:
            motionCoor = np.loadtxt(allFiles[0][:-3] + 'csv', delimiter=',', skiprows=1)
            imStack = io.imread(allFiles[0])
        return (imStack, motionCoor, allFiles[0])

    ############################################################
    def saveRungMotionData(self, mouse, date, rec, rungPositions):
        rec = rec.replace('/', '-')
        pickle.dump(rungPositions, open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb'))

    ############################################################
    def getRungMotionData(self, mouse, date, rec):
        rec = rec.replace('/', '-')
        rungPositions = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'rb'))
        return rungPositions

    ############################################################
    def extractRoiSignals(self, folder, rec, tifFile):

        self.simaPath = self.analysisLocation + '%s_%s_%s' % (self.mouse, folder, rec)
        print(self.simaPath)
        if os.path.isdir(self.simaPath + '.sima'):
            print('sima dir exists')
            dataSet = sima.ImagingDataset.load(self.simaPath + '.sima')
        else:
            print('create sima dir')
            sequences = [sima.Sequence.create('TIFF', tifFile)]
            dataSet = sima.ImagingDataset(sequences, self.simaPath, channel_names=['GCaMP6F'])

        img = dataSet.time_averages[0][:, :, 0]

        self.simaPath = self.simaPath + '.sima/'
        overwrite = True
        if os.path.isfile(self.simaPath + 'rois.pkl'):
            print('rois already exist')
            input_ = raw_input('rois traces exist already, do you want to overwrite? (type \'y\' to overwrite, any character if not) : ')
            if input_ != 'y':
                overwrite = False
        if overwrite:
            print('create rois with roibuddy')
            segmentation_approach = sima.segment.STICA(channel='GCaMP6F', components=1, mu=0.9,  # weighting between spatial - 1 - and temporal - 0 - information
                # spatial_sep=0.8
            )
            print('segmenting calcium image ... ', end=" ")
            dataSet.segment(segmentation_approach, 'GCaMP6F_signals', planes=[0])
            print('done')
            while True:
                input_ = raw_input('Please check ROIs in \'roibuddy\' (type \'exit\' to halt) : ')
                if input_ == 'exit':
                    sys.exit(1)
                else:
                    break
        dataSet = sima.ImagingDataset.load(self.simaPath)
        rois = dataSet.ROIs['GCaMP6F_signals']

        # Extract the signals.
        dataSet.extract(rois, signal_channel='GCaMP6F', label='GCaMP6F_signals')
        raw_signals = dataSet.signals('GCaMP6F')['GCaMP6F_signals']['raw']
        # dataSet.export_signals('example_signals.csv', channel='GCaMP6F',signals_label='GCaMP6F_signals')
        # pdb.set_trace()

        # dataSet.signals('GCaMP6F')['GCaMP6F_signals']['mean_frame']

        # pdb.set_trace()
        roiLabels = []
        # extrace labels
        for n in range(len(rois)):
            roiLabels.append(rois[n].label)  # print 'ROI label', n, rois[n].label

        return (img, rois, raw_signals)

    ############################################################
    def getRecordingsList(self, expDate='all', recordings='all'):
        if recordings == 'all910':
            recID = 'recs910'
        elif recordings == 'all820':
            recID = 'recs820'
        elif recordings == 'all':
            recID = ['recs910','recs820']

        self.config = readConfigFile('simplexAnimals.config')
        #pdb.set_trace()
        for i in range(len(self.config)):
            if self.config['%s' % i]['mouse'] == self.mouse:
                print('experiment dictionary of mouse exists')
                self.expDict = self.config['%s' % i]['days']
                dictExists = True
        #pdb.set_trace()
        folderRec = []
        if self.mouse in self.listOfAllExpts:
            # print mouse
            # print expDate, self.listOfAllExpts[mouse]['dates']
            expDateList = []
            # pdb.set_trace()
            # provide choice of which days to include in analysis
            if expDate == 'some':
                print('Dates when experiments where performed with animal %s :' % self.mouse)
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    print('  %s %d' % (d, didx))
                    didx += 1
                print('Choose the dates for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\' : ', end='')
                daysInput = input()
                daysInputIdx = [int(i) for i in daysInput.split(',')]  # print(daysInputIdx,daysInputIdx[0],type(daysInputIdx))
            elif expDate == 'all910' or expDate == 'all820':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        if recID in self.expDict[d]:
                            daysInputIdx.append(didx)
                    didx+=1
            elif expDate == 'all':
                didx = 0
                daysInputIdx = []
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if d in self.expDict.keys() :
                        daysInputIdx.append(didx)
                    didx+=1
            ########################################################
            # generate list of days to analyze
            if expDate == 'some' or expDate=='all910' or expDate=='all820' or expDate == 'all':
                didx = 0
                for d in self.listOfAllExpts[self.mouse]['dates']:
                    if didx in daysInputIdx:
                        # print(d)
                        expDateList.append(d)
                    didx += 1
            else:
                expDateList.append(expDate)
            print('Selected dates :', expDateList)
            #pdb.set_trace()
            #####################################################
            # chose recordings
            if recordings == 'all910':
                print('All 910 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs910' in self.expDict[eD]:
                        idx910 = self.expDict[eD]['recs910']['recordings']
                    else:
                        idx910 = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx910 is not None:
                            recInputIdx.append(recIdx+idx910)
                        recIdx+=len(recList)
                #pdb.set_trace()
            elif recordings == 'all820':
                print('All 820 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs820' in self.expDict[eD]:
                        idx820 = self.expDict[eD]['recs820']['recordings']
                    else:
                        idx820 = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx820 is not None:
                            recInputIdx.append(recIdx+idx820)
                        recIdx+=len(recList)
                #pdb.set_trace()
            elif recordings == 'some':
                # first show all recordings for a given date
                print('Choose recording to analyze')
                recIdx = 0
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    for fold in dataFolders:
                        print(' ', fold)
                        # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        #self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                        #print(self.dataLocation)
                        if not os.path.exists(self.dataLocation):
                            #    print('experiment %s exists' % fold)
                            # else:
                            print('Problem, experiment does not exist')
                        # recList = OrderedDict()
                        recList = self.getDirectories(self.dataLocation)
                        for r in recList:
                            print('    %s  %s' % (r, recIdx))
                            recIdx += 1
                print('Choose the recordings for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\' : ', end='')
                recInput = input()
                recInputIdx = [int(i) for i in recInput.split(',')]
            elif recordings == 'all':
                print('All 910 and 820 recordings will be analyzed')
                recIdx = 0
                recInputIdx = []
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                    if 'recs910' in self.expDict[eD]:
                        idx910 = self.expDict[eD]['recs910']['recordings']
                    else:
                        idx910 = None
                    if 'recs820' in self.expDict[eD]:
                        idx820 = self.expDict[eD]['recs820']['recordings']
                    else:
                        idx820 = None
                    for fold in dataFolders:
                        #print(' ', fold)
                        self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                        recList = self.getDirectories(self.dataLocation)
                        #print(recList)
                        if idx910 is not None:
                            recInputIdx.append(recIdx+idx910)
                        if idx820 is not None:
                            recInputIdx.append(recIdx + idx820)
                        recIdx+=len(recList)
            else:
                recInputIdx = [int(i) for i in recordings.split(',')]
            print('list of recordings : ', recInputIdx)
            #
            # pdb.set_trace()
            # then compile a list the selected recordings
            recIdx = 0
            for eD in expDateList:
                # print(expDateList, self.listOfAllExpts[mouse]['dates'], len(self.listOfAllExpts[mouse]['dates']))
                dataFolders = self.listOfAllExpts[self.mouse]['dates'][eD]['folders']
                # print(eD, self.listOfAllExpts[mouse]['dates'],dataFolders)
                for fold in dataFolders:
                    # self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                    # pdb.set_trace()
                    self.dataLocation = self.dataBase + self.dataPCLocation[dataFolders[fold]['recComputer']] + fold + '/'
                    self.recordingMachine = self.computerDict[dataFolders[fold]['recComputer']]
                    print(self.dataLocation)
                    if not os.path.exists(self.dataLocation):
                        # print('experiment %s exists' % fold)
                        # else:
                        print('Problem, experiment does not exist')
                    recList = self.getDirectories(self.dataLocation)
                    if recordings == 'all':
                        tempRecList = []
                        for r in recList:
                            # if r[:-4] == 'locomotionTriggerSIAndMotor' or r[:-4] == 'locomotionTriggerSIAndMotorJin' or r[:-4] == 'locomotionTriggerSIAndMotor60sec' :
                            if r[:-4] in self.listOfSequenceExperiments:
                                subFolders = self.getDirectories(self.dataLocation + '/' + r)
                                for i in range(len(subFolders)):
                                    if subFolders[i][0] == '0':
                                        tempRecList.append(r + '/' + subFolders[i])
                                    else:
                                        tempRecList.append(r)
                                        break
                            else:
                                tempRecList.append(r)
                        folderRec.append([fold, eD, tempRecList])  # folderRec.append([fold,eD,recList])
                    else: # recordings == 'some':
                        tempRecList = []
                        for r in recList:
                            # only add recordings which were previously selected
                            if recIdx in recInputIdx:
                                # if r[:-4] == 'locomotionTriggerSIAndMotor' or r[:-4] == 'locomotionTriggerSIAndMotorJin' or r[:-4] == 'locomotionTriggerSIAndMotor60sec':
                                if r[:-4] in self.listOfSequenceExperiments:
                                    subFolders = self.getDirectories(self.dataLocation + '/' + r)
                                    for i in range(len(subFolders)):
                                        if subFolders[i][0] == '0':
                                            tempRecList.append(r + '/' + subFolders[i])
                                        else:
                                            tempRecList.append(r)
                                            break
                                else:
                                    tempRecList.append(r)
                            recIdx += 1
                        folderRec.append([fold, eD, tempRecList])
        #pdb.set_trace()
        print('Data was recorded on %s' % self.recordingMachine)
        return (folderRec, dataFolders)

    ############################################################
    def getDirectories(self, location):
        # seqFolder = self.dataLocation + '/' + r
        subFolders = [os.path.join(o) for o in os.listdir(location) if os.path.isdir(os.path.join(location, o))]
        subFolders.sort()
        return subFolders

    ############################################################
    def checkIfDeviceWasRecorded(self, fold, eD, recording, device):
        # recLocation =  (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
        # print(recLocation,eD,int(eD))
        if os.path.exists(recLocation):
            print('%s constains %s , ' % (fold, recording), end=" ")
        else:
            print('Problem, recording does not exist')
        if device in ['CameraGigEBehavior', 'CameraPixelfly']:
            if int(eD) >= 191104:
                pathToFile = recLocation + device + '/' + 'video_000.ma'
            else:
                pathToFile = recLocation + device + '/' + 'frames.ma'
        elif device == 'PreAmpInput':
            pathToFile = recLocation + '%s.ma' % 'DaqDevice'
        elif device == 'frameTimes':
            pathToFile = recLocation + '%s/%s.ma' % ('CameraGigEBehavior', 'daqResult')
        elif device == 'SICaImaging':
            recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/'
            print(recLocation)
            tiffList = glob.glob(recLocation + '*tif')
            tiffList.sort()
            print(tiffList)
            if len(tiffList) > 0:
                print('Ca imaging was acquired with ScanImage')
                return (True, tiffList, recLocation)
            else:
                print('No Ca imaging with ScanImage here.')
                return (False, [], None)
        else:
            pathToFile = recLocation + '%s.ma' % device
        print(pathToFile)

        if os.path.isfile(pathToFile):
            fData = h5py.File(pathToFile, 'r')
            try:
                kk = fData['data']
            except KeyError:
                print('Device %s was acquired but NO data exists' % device)
                return (False, None)
            else:
                print('Device %s was acquired' % device)
                return (True, fData)
        else:
            print('Device %s was NOT acquired' % device)
            return (False, None)

    ############################################################
    def checkIfPawPositionWasExtracted(self, fold, eD, recording,DLCinst):

        rec = recording.replace('/', '-')
        fName = self.analysisLocation + '%s_%s_%s_*%s.h5' % (self.mouse, fold, rec, DLCinst)
        fList = glob.glob(fName)
        pdb.set_trace()
        if len(fList) > 1:
            print('more than one file exist matching the file pattern %s' % fName)
            return (False, None)
        elif len(fList) == 1:
            print('paw data extraced and saved in %s' % fList[0])
            return (True, fList[0])
        else:
            print('no extraced paw data found for %s' % fName)
            return (False, None)

    ############################################################
    def readRawData(self, fold, eD, recording, device, fData, readRawData=True):
        # recLocation = (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        recLocation = self.dataBase + self.dataPCLocation[self.listOfAllExpts[self.mouse]['dates'][eD]['folders'][fold]['recComputer']] + fold + '/' + recording + '/'
        # print(recLocation)
        if device == 'RotaryEncoder':
            # data from activity monitor
            if len(fData['data']) == 1:
                angles = fData['data'][()][0]
            # data during high-res recording
            else:
                angles = fData['data'][4]
            times = fData['info/1/values'][()]
            try:
                startTime = fData['info/2/DAQ/ChannelA'].attrs['startTime']
            except:
                startTime = os.path.getctime(recLocation)
                monitor = True
            else:
                monitor = False
            return (angles, times, startTime, monitor)

        elif device == 'AxoPatch200_2':
            current = fData['data'][()][0]
            # pdb.set_trace()
            ephysTimes = fData['info/1/values'][()]
            # imageMetaInfo = self.readMetaInformation(recLocation)
            return (current, ephysTimes)

        elif device == 'Imaging':
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = self.readMetaInformation(recLocation)
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'CameraGigEBehavior':
            print('reading raw GigE data ...')
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = self.readGigEMetaInformation(recLocation)
            print('done')
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'CameraPixelfly':
            print('reading raw Pixelfly data ...', end=" ")
            if readRawData:
                frames = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            xPixelSize = fData['info/3/pixelSize'].attrs['0']
            yPixelSize = fData['info/3/pixelSize'].attrs['1']
            xSize = fData['info/3/region'].attrs['2']
            ySize = fData['info/3/region'].attrs['3']
            imageMetaInfo = np.array([xSize * xPixelSize, ySize * yPixelSize, xPixelSize, yPixelSize])
            print('done')
            return (frames, frameTimes, imageMetaInfo)
        elif device == 'PreAmpInput' or device == 'frameTimes':
            values = fData['data'][()]
            valueTimes = fData['info/1/values'][()]
            return (values, valueTimes)

        elif device == 'pawTraces':
            pawF = h5py.File(fData, 'a')
            pawTraces = pawF['df_with_missing']['table'][()]
            pawTracesA = pawTraces.view((float, (len(pawTraces[0][1]) + 1)))
            pawTracesA[:, 0] = np.arange(len(pawTraces))

            pFileName = '%s*.pickle' % fData[:-3]
            pfList = glob.glob(pFileName)
            print(pfList)
            if len(pfList) == 1:
                #pdb.set_trace()
                pawMetaData = pickle.load(open(pfList[0], 'rb'))
            else:
                pawMetaData = None
            return (pawTracesA, pawMetaData)

    ############################################################
    def readMetaInformation(self, filePath):
        # convert to um
        conversion = 1.E6
        config = readConfigFile(filePath + '.index')
        pixWidth = config['.']['Scanner']['program'][0]['scanInfo']['pixelWidth'][0] * conversion
        pixHeight = config['.']['Scanner']['program'][0]['scanInfo']['pixelHeight'][0] * conversion
        dimensionXY = np.array(config['.']['Scanner']['program'][0]['roi']['size']) * conversion
        position = np.array(config['.']['Scanner']['program'][0]['roi']['pos']) * conversion

        if pixWidth == pixHeight:
            deltaPix = pixWidth
        else:
            print('Pixel height and width are not equal.')
            sys.exit(1)

        # print r'dimensions (x,y, pixelsize in um) : ', np.hstack((dimensionXY,deltaPix))
        return np.hstack((position, dimensionXY, deltaPix))  # self.h5pyTools.createOverwriteDS(dataGroup,dataSetName,hstack((dimensionXY,deltaPix)))

    ############################################################
    def readGigEMetaInformation(self, filePath):
        config = readConfigFile(filePath + '.index')
        starttime = config['.']['startTime']
        # pdb.set_trace()
        return starttime

    ############################################################
    def saveLEDPositionCoordinates(self, groupNames, coordinates):
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # pdb.set_trace()
        self.h5pyTools.createOverwriteDS(grpHandle, 'LEDcoordinates', np.column_stack((coordinates[1], coordinates[2])), [['nLED', coordinates[0]], ['circleRadius', coordinates[3]]])

    ############################################################
    def readLEDPositionCoordinates(self, currentGroupName):
        temp = self.f[currentGroupName + '/LEDcoordinates'][()]
        posX = temp[:, 0]
        posY = temp[:, 1]
        nLED = self.f[currentGroupName + '/LEDcoordinates'].attrs['nLED']
        circleRadius = self.f[currentGroupName + '/LEDcoordinates'].attrs['circleRadius']
        coordinates = np.array([nLED, posX, posY, circleRadius], dtype=object)  # type is object to allow for different data structure of the entries, i.e., single number and array
        return coordinates

    ############################################################
    # foldersRecordings[f][0], foldersRecordings[f][1], foldersRecordings[f][2], r
    def checkForLEDPositionCoordinates(self, date, folder, recordings, r):
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        currentGroupNames = [date, recordings[r], 'LEDinVideo']
        (currentGroupName, currentGrpHandle) = self.h5pyTools.getH5GroupName(self.f, currentGroupNames)
        # check if coordinates for current recording exist already
        try:
            currentLEDcoordinates = self.readLEDPositionCoordinates(currentGroupName)
        except KeyError:
            currentCoodinatesExist = False
        else:
            print('LED roi coordinates for current recording exist')
            currentCoodinatesExist = True
            return (currentCoodinatesExist, currentLEDcoordinates)
        if r > 0:
            previousGroupNames = [date, recordings[r - 1], 'LEDinVideo']
            (previousGroupName, previousGrpHandle) = self.h5pyTools.getH5GroupName(self.f, previousGroupNames)
            try:
                previousLEDcoordinates = self.readLEDPositionCoordinates(previousGroupName)  # self.f[previousGroupName+'/LEDcoordinates'][()]
            except KeyError:
                pass
            else:
                print('LED roi coordinates for previous recording exist')
                return (currentCoodinatesExist, previousLEDcoordinates)
        print('NO LED roi coordinates exist')
        return (currentCoodinatesExist, None)

    ############################################################
    def checkForErroneousFramesIdx(self, date, folder, recordings, r, determineAgain=False):
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        currentGroupNames = [date, recordings[r], 'erroneousFrames']
        # print(currentGroupNames)
        # pdb.set_trace()
        try:
            (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, currentGroupNames)
            idxExc = self.f[grpName + '/idxToExclude'][()]
            # print(type(idxExc))
            idxExc = np.array(idxExc, dtype=int)  # check if coordinates for current recording exist already
        except KeyError:
            excludeIdxExist = False
            canBeUsed = True
            return (excludeIdxExist, None, canBeUsed)
        else:
            print('idx of erroneous frames exist')
            excludeIdxExist = True
            try:
                canBeUsed = self.f[grpName + '/canBeUsed'][()][0]
            except KeyError:
                canBeUsed = True
            if np.array_equal(idxExc, np.array([-1])):
                idxExc = np.array([], dtype=int)
            if determineAgain:
                return (False, idxExc,canBeUsed)
            else:
                return (excludeIdxExist, idxExc,canBeUsed)

    ############################################################
    def saveErroneousFramesIdx(self, groupNames, idxToExclude,canBeUsed=True):
        # [foldersRecordings[f][0], foldersRecordings[f][2][r], 'behavior_video']
        # print(groupNames)
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # print(grpName,grpHandle)
        if len(idxToExclude) == 0:
            idxToExclude = np.array([-1])  # pdb.set_trace(
        self.h5pyTools.createOverwriteDS(grpHandle, 'idxToExclude', idxToExclude)
        self.h5pyTools.createOverwriteDS(grpHandle, 'canBeUsed', np.array([canBeUsed]))
        self.f.flush()
        print('saved successfully', idxToExclude)

    ############################################################
    # idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo
    def saveBehaviorVideoTimeData(self, groupNames, idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo):
        # framesDuringRecording, startEndFrameTime, startEndFrameIdx, imageMetaInfo)
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        # self.h5pyTools.createOverwriteDS(grpHandle,'behaviorFrames',len(frames))
        # pdb.set_trace()
        self.h5pyTools.createOverwriteDS(grpHandle, 'indexVideo', videoIdx, ['startTime', imageMetaInfo])
        self.h5pyTools.createOverwriteDS(grpHandle, 'indexTimePoints', idxTimePoints)
        self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposureTime', startEndExposureTime)
        self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposurepIndex', startEndExposurepIdx)
        self.h5pyTools.createOverwriteDS(grpHandle, 'frameDropExcludeSummary', frameSummary)

    ############################################################
    def readBehaviorVideoTimeData(self, groupNames):
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        videoIdx = self.f[grpName + '/indexVideo'][()]
        imageMetaInfo = self.f[grpName + '/indexVideo'].attrs['startTime']
        idxTimePoints = self.f[grpName + '/indexTimePoints'][()]
        startEndExposureTime = self.f[grpName + '/startEndExposureTime'][()]
        startEndExposurepIdx = self.f[grpName + '/startEndExposurepIndex'][()]
        frameSummary = self.f[grpName + '/frameDropExcludeSummary'][()]
        return (idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo)

    ############################################################
    def getBehaviorVideoFrames(self, groupNames):
        (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        print(grpName)
        #pdb.set_trace()  [date,rec,'behavior_video']
        try:
            firstLastRecordedFrame = self.f[grpName + '/firstLastRecordedFrame'][()]
        except:  # before, first and last name was stored under a different name
            groupNames[2] = 'behavior_video'
            (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
            firstLastRecordedFrame = self.f[grpName + '/firstLastRecordedFrame'][()]
        else:
            pass

        return firstLastRecordedFrame

    ############################################################
    # self.saveBehaviorVideoData([date,rec,'behavior_video'],  framesRaw,videoIdx,startEndExposureTime, imageMetaInfo)
    # [foldersRecordings[f][0], foldersRecordings[f][2][r],'behavior_video']
    def saveBehaviorVideoFrames(self, groupNames,  framesRaw, videoIdx ):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        #print(grpName)
        #self.h5pyTools.createOverwriteDS(grpHandle, 'startEndExposureTime', startEndExposureTime,['imageMetaInfo', imageMetaInfo])
        self.h5pyTools.createOverwriteDS(grpHandle, 'firstLastRecordedFrame', np.array((framesRaw[videoIdx[0]],framesRaw[videoIdx[-1]])))

    ############################################################
    def saveImageStack(self, frames, fTimes, imageMetaInfo, groupNames, motionCorrection=[]):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImaging', frames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImagingTime', fTimes)
        self.h5pyTools.createOverwriteDS(grpHandle, 'caImagingField', imageMetaInfo)
        if len(motionCorrection) > 1:
            self.h5pyTools.createOverwriteDS(grpHandle, 'motionCoordinates', motionCorrection)

    ############################################################
    def readImageStack(self, groupNames):
        (grpName, test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        frames = self.f[grpName + '/caImaging'][()]
        fTimes = self.f[grpName + '/caImagingTime'][()]
        imageMetaInfo = self.f[grpName + '/caImagingField'][()]
        return (frames, fTimes, imageMetaInfo)

    ############################################################
    def saveWalkingActivity(self, angularSpeed, linearSpeed, wTimes, angles, aTimes, startTime, monitor, groupNames):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle, 'angularSpeed', angularSpeed, ['monitor', monitor])
        self.h5pyTools.createOverwriteDS(grpHandle, 'linearSpeed', linearSpeed)
        self.h5pyTools.createOverwriteDS(grpHandle, 'walkingTimes', wTimes, ['startTime', startTime])
        self.h5pyTools.createOverwriteDS(grpHandle, 'anglesTimes', np.column_stack((aTimes, angles)), ['startTime', startTime])

    ############################################################
    def getWalkingActivity(self, groupNames):
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        print(grpName)
        angularSpeed = self.f[grpName + '/angularSpeed'][()]
        monitor = self.f[grpName + '/angularSpeed'].attrs['monitor']
        linearSpeed = self.f[grpName + '/linearSpeed'][()]
        wTimes = self.f[grpName + '/walkingTimes'][()]
        startTime = self.f[grpName + '/walkingTimes'].attrs['startTime']
        angleTimes = self.f[grpName + '/anglesTimes'][()]
        return (angularSpeed, linearSpeed, wTimes, startTime, monitor, angleTimes)

    ############################################################
    def getPawRungPickleData(self, date, rec):
        frontpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (self.mouse, date, rec), 'rb'))
        hindpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (self.mouse, date, rec), 'rb'))
        rungs = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (self.mouse, date, rec), 'rb'))
        return (frontpawPos, hindpawPos, rungs)

    ############################################################
    def readMetaDataFileAndReadSetting(self, tiffFile, keyWord):
        metData = ScanImageTiffReader(tiffFile).metadata()
        #pdb.set_trace()
        keyWordIdx = metData.find(keyWord)
        splitString = re.split('=|\n', metData[keyWordIdx:])
        keyWordParameter = float(splitString[1])
        return keyWordParameter

    ############################################################
    def readTimeStampOfFrame(self, tiffFileObj, nFrame):
        # frameNumberAcqMode
        # numeric, number of frame, counted from beginning of acquisition mode
        #
        # frameNumberAcq
        # numeric, number of frame in current acquisition
        #
        # acqNumber
        # numeric, number of current acquisition
        #
        # epochAcqMode
        # string, time of the acquisition of the acquisiton of the first pixel in the current acqMode; format: output of datestr(now) '25-Jul-2014 12:55:21'
        #
        # frameTimestamp
        # [s] time of the first pixel in the frame passed since acqModeEpoch
        #
        # acqStartTriggerTimestamp
        # [s] time of the acq start trigger for the current acquisition
        #
        # nextFileMarkerTimestamp
        # [s] time of the last nextFileMarker recorded. NaN if no nextFileMarker was recorded

        desc = tiffFileObj.description(nFrame)
        keyWordIdx = desc.find('epoch')  # string, time of the acquisition of the acquisiton of the first pixel in the current acqMode; format: output of datestr(now) '25-Jul-2014 12:55:21'
        dateString = re.split('\[|\]', desc[keyWordIdx:])
        dateIdv = dateString[1].split()
        # print(dateIdv)
        unixStartTime = int(datetime.datetime(int(dateIdv[0]), int(dateIdv[1]), int(dateIdv[2]), int(dateIdv[3]), int(dateIdv[4]), int(float(dateIdv[5]))).strftime('%s'))
        #
        keyWordIdx = desc.find('frameTimestamps_sec')  # [s] time of the first pixel in the frame passed since acqModeEpoch
        splitString = re.split('=|\n', desc[keyWordIdx:])
        frameTimestamps = float(splitString[1])

        keyWordIdx = desc.find('acqTriggerTimestamps_sec')  # [s] time of the acq start trigger for the current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        acqTriggerTimestamps = float(splitString[1])

        keyWordIdx = desc.find('frameNumberAcquisition')  # numeric, number of frame in current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        frameNumberAcquisition = int(splitString[1])

        keyWordIdx = desc.find('acquisitionNumbers')  # numeric, number of current acquisition
        splitString = re.split('=|\n', desc[keyWordIdx:])
        acquisitionNumbers = int(splitString[1])

        unixFrameTime = unixStartTime + frameTimestamps
        # print(tiffFile,unixTime)
        return ([frameNumberAcquisition, acquisitionNumbers, unixStartTime, unixFrameTime, frameTimestamps, acqTriggerTimestamps])
    ############################################################
    def getRawCalciumImagingData(self, tiffList,saveDir):
        imagingData = []
        timeStamps = []
        for i in range(len(tiffList)):
            # data = ScanImageTiffReader(tiffPaths[i]).data()
            tiffFileObject = ScanImageTiffReader(tiffList[i])
            data = tiffFileObject.data()
            # pdb.set_trace()
            fN = np.shape(data)[0]
            # frameNumbers.append(fN)
            for n in range(fN):
                timeStamps.append(self.readTimeStampOfFrame(tiffFileObject, n))
            timeStampsASingle = np.asarray(timeStamps)
            imagingData.append([i, data, timeStampsASingle])
        timeStampsA = np.asarray(timeStamps)
        return (imagingData, timeStampsA)  # np.save(saveDir+'/suite2p/plane0/timeStamps.npy',timeStampsA)

    ############################################################
    def getAnalyzedCaImagingData(self, analysisLocation, tiffList):

        timeStamps = []
        for i in range(len(tiffList)):
            zF = self.readMetaDataFileAndReadSetting(tiffList[i], 'scanZoomFactor')
            if i == 0:
                zFold = zF
            else:
                if zF != zFold:
                    print('scanZoomFactor is not the same between recordings!')
            tiffFileObject = ScanImageTiffReader(tiffList[i])
            timeS = self.readTimeStampOfFrame(tiffFileObject, 0)
            timeStamps.append(timeS[3])
        #
        nDirs = 0
        for name in glob.glob(analysisLocation+'_suite2p*'):
            caAnalysisLocation = name
            print(name)
            nDirs+=1
        if nDirs > 1:
            print('There are more than one matching directory!')
            pdb.set_trace()
        #caAnalysisLocation = eSD.analysisLocation+foldersRecordings[f][0]+'_suite2p/'
        if os.path.isdir(caAnalysisLocation):
            ops = np.load(caAnalysisLocation + '/suite2p/plane0/ops.npy',allow_pickle=True)
            ops = ops.item()
            nframes = ops['nframes']
            meanImg = ops['meanImg']
            meanImgE = ops['meanImgE']
            # pdb.set_trace()
            return (nframes, meanImg, meanImgE, zF, timeStamps)
        else:
            print('Ca imaging data has not been analyzed wiht suite2p yet!')

    ############################################################
    def extractAndSaveCaTimeStamps(self, dataDir, saveDir, tiffPaths):

        (_,timeStampsA) = self.getRawCalciumImagingData(tiffPaths, saveDir)
        np.save(saveDir + '/suite2p/plane0/timeStamps.npy', timeStampsA)

    ############################################################
    def getCaImagingRoiData(self, caAnalysisLocation, tiffList):
        # frameNumbers = []
        # timeStamps = []
        # for i in range(len(tiffList)):
        #    data = ScanImageTiffReader(tiffList[i]).data()
        #    fN = np.shape(data)[0]
        #    frameNumbers.append(fN)
        #    for n in range(fN):
        #        timeStamps.append(self.readTimeStampOfRecording(tiffList[i],n))

        # pdb.set_trace()

        if os.path.isdir(caAnalysisLocation):
            timeStamps = np.load(caAnalysisLocation + '/suite2p/plane0/timeStamps.npy')
            F = np.load(caAnalysisLocation + '/suite2p/plane0/F.npy')
            Fneu = np.load(caAnalysisLocation + '/suite2p/plane0/Fneu.npy')
            ops = np.load(caAnalysisLocation + '/suite2p/plane0/ops.npy',allow_pickle=True)
            ops = ops.item()
            iscell = np.load(caAnalysisLocation + '/suite2p/plane0/iscell.npy')
            stat = np.load(caAnalysisLocation + '/suite2p/plane0/stat.npy',allow_pickle=True)

            # pdb.set_trace()
            nRois = np.arange(len(F))
            realCells = (iscell[:, 0] == 1)
            nRois = nRois[realCells]
            Fluo = F[realCells] - 0.7 * Fneu[realCells]  # substract neuropil data
            stat = stat[realCells]
            # pdb.set_trace()
            return (Fluo, nRois, ops, timeStamps, stat)

    ############################################################
    def saveTif(self, frames, mouse, date, rec, norm=None):
        img_stack_uint8 = np.array(frames, dtype=np.uint8)
        if norm:
            tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack%s.tif' % (mouse, date, rec, norm), img_stack_uint8)
        else:
            tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def readTif(self, frames, mouse, date, rec):
        img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def readPawTrackingData(self, date, rec, DLCinstance):
        rec = rec.replace('/', '-')
        (grpName, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'pawTrackingData',DLCinstance])
        # pdb.set_trace()
        rawPawPositionsFromDLC = self.f[grpName + '/rawPawPositionsFromDLC'][()]
        # self.h5pyTools.createOverwriteDS(grpHandle, 'croppingParameters', np.array(cropping))
        croppingParameters = self.f[grpName + '/croppingParameters'][()]
        pawTrackingOutliers = []
        jointNamesFramesInfo = []
        pawSpeed = []
        rawPawSpeed = []
        cPawPos = []
        for i in range(4):
            pTTemp = self.f[grpName + '/pawTrackingOutliers%s' % i][()]
            pawTrackingOutliers.append(pTTemp)
            jNTemp = 'paw %s' % i #self.f[grpName + '/pawTrackingOutliers%s' % i].attrs['PawID']
            jointNamesFramesInfo.append(jNTemp)
            pStemp = self.f[grpName + '/clearedPawSpeed%s' % i][()]
            pawSpeed.append(pStemp)
            pPPtemp = self.f[grpName + '/clearedXYPos%s' % i][()]
            cPawPos.append(pPPtemp)
            rPStemp = self.f[grpName + '/rawPawSpeed%s' % i][()]
            rawPawSpeed.append(rPStemp)
            if i == 0:
                recStartTime = self.f[grpName + '/clearedPawSpeed%s' % i].attrs['recStartTime']

        return (rawPawPositionsFromDLC, pawTrackingOutliers, jointNamesFramesInfo, pawSpeed, recStartTime, rawPawSpeed, cPawPos, croppingParameters)

    ############################################################
    # savePawTrackingData(mouse,foldersRecordings[f][0],foldersRecordings[f][2][r],DLCinstance,pawTrackingOutliers,pawMetaData,startEndExposureTime,imageMetaInfo,generateVideo=False)
    def savePawTrackingData(self, mouse, date, rec, DLCinstance, pawPositions, pawTrackingOutliers, pawMetaData, startEndExposureTime, startTime, generateVideo=True):
        # pdb.set_trace()
        jointNames = pawMetaData['data']['DLC-model-config file']['all_joints_names']
        jointIdx = pawMetaData['data']['DLC-model-config file']['all_joints']
        cropping = pawMetaData['data']['cropping_parameters']
        print('cropping parameters', cropping)
        # pdb.set_trace()
        rec = rec.replace('/', '-')
        (test, grpHandle) = self.h5pyTools.getH5GroupName(self.f, [date, rec, 'pawTrackingData', DLCinstance])
        self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawPositionsFromDLC', pawPositions)
        self.h5pyTools.createOverwriteDS(grpHandle, 'croppingParameters', np.array(cropping))
        timeArray = np.average(startEndExposureTime,axis=1) # use the 'middle' of the exposure time as time-point of the frame
        for i in range(4):
            pawMask = pawTrackingOutliers[i][3]
            #pdb.set_trace()
            rawPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)])) ** 2) / np.diff(timeArray)
            rawSpeedTime = (timeArray[:-1] + timeArray[1:]) / 2.
            clearedPawSpeed = np.sqrt((np.diff(pawPositions[:, (i * 3 + 1)][pawMask])) ** 2 + (np.diff(pawPositions[:, (i * 3 + 2)][pawMask])) ** 2) / np.diff(timeArray[pawMask])
            clearedPawXSpeed = np.diff(pawPositions[:, (i * 3 + 1)][pawMask]) / np.diff(timeArray[pawMask])
            clearedPawYSpeed = np.diff(pawPositions[:, (i * 3 + 2)][pawMask]) / np.diff(timeArray[pawMask])
            clearedSpeedTime = (timeArray[pawMask][:-1] + timeArray[pawMask][1:]) / 2.
            clearedPosIdx = np.arange(len(pawPositions))[pawMask]
            clearedXYPos = np.column_stack((timeArray[pawMask], pawPositions[:, (i * 3 + 1)][pawMask], pawPositions[:, (i * 3 + 2)][pawMask], clearedPosIdx))
            clearedSpeedIdx = np.array((clearedPosIdx[:-1] + clearedPosIdx[1:]) / 2., dtype=int)
            #pdb.set_trace()
            self.h5pyTools.createOverwriteDS(grpHandle, 'pawTrackingOutliers%s' % i, pawTrackingOutliers[i][3],[['PawID', jointNames[i]],['pawTrackingOutliers',np.array([ pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]])]])#, pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
                                             #['PawID', [jointNames[i], pawTrackingOutliers[i][1], pawTrackingOutliers[i][2]]])
            self.h5pyTools.createOverwriteDS(grpHandle, 'rawPawSpeed%s' % i, np.column_stack((rawSpeedTime, rawPawSpeed)), ['recStartTime', startTime])
            self.h5pyTools.createOverwriteDS(grpHandle, 'clearedPawSpeed%s' % i, np.column_stack((clearedSpeedTime, clearedPawSpeed, clearedPawXSpeed, clearedPawYSpeed, clearedSpeedIdx)),
                                             ['recStartTime', startTime])
            self.h5pyTools.createOverwriteDS(grpHandle, 'clearedXYPos%s' % i, clearedXYPos)  # pdb.set_trace()
        if generateVideo:
            fps = 80
            width = 800
            heigth = 600
            colors = [(255, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 255)]
            indicatorPositions = [(270, 15), (270, 35), (240, 35), (240, 15)]
            sourceVideoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
            outputVideoFileName = self.analysisLocation + '%s_%s_%s_paw_tracking.avi' % (mouse, date, rec)
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
            # print('1',fourcc,fps,width,heigth)
            out = cv2.VideoWriter(outputVideoFileName, fourcc, fps, (width, heigth))
            source = cv2.VideoCapture(sourceVideoFileName)
            nFrame = 0
            while (source.isOpened()):
                ret, frame = source.read()
                if ret == True:
                    # cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i],4), (10,20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
                    for i in range(4):
                        # print(int(pawPositions[nFrame, 3 * i + 1]+0.5),int(pawPositions[nFrame, 3 * i + 2]+0.5))
                        cv2.drawMarker(frame, (cropping[0] + int(pawPositions[nFrame, 3 * i + 1] + 0.5), cropping[2] + int(pawPositions[nFrame, 3 * i + 2] + 0.5)), colors[i],
                                       cv2.MARKER_CROSS, 20, 2)
                        if nFrame in pawTrackingOutliers[i][3]:
                            cv2.circle(frame, indicatorPositions[i], 7, (0, 255, 0), -1)
                        else:
                            cv2.circle(frame, indicatorPositions[i], 7, (0, 0, 255), -1)
                    out.write(frame)
                else:
                    break
                # print(nFrame)
                nFrame += 1
            out.release()
            source.release()

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def saveBehaviorVideo(self, mouse, date, rec, framesRaw, idxTimePoints, startEndExposureTime, startEndExposurepIdx, videoIdx, frameSummary, imageMetaInfo):
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
        midFrameTimes = (startEndExposureTime[:, 0] + startEndExposureTime[:, 1]) / 2.
        #pdb.set_trace()
        # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior.avi' % (mouse, date, rec)
        # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))

        vLength = np.shape(framesRaw)[0]
        width = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        # print('number of frames :', vLength, width, heigth)
        fps = 80
        # pdb.set_trace()
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width, heigth))
        # pdb.set_trace()
        for i in np.arange(len(videoIdx)):
            frame8bit = np.array(np.transpose(framesRaw[videoIdx[i]]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (videoIdx[i], (vLength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            out.write(frame)
        # Release everything if job is finished
        # cap.release()
        out.release()
        cv2.destroyAllWindows()

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def saveBehaviorVideoWithCa(self, mouse, date, rec, framesRaw, expStartTime, expEndTime, imageMetaInfo, angles, aTimes):
        # [foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        midFrameTimes = (expStartTime + expEndTime) / 2.
        # pdb.set_trace()
        # img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        # tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/', '-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior_withCa.avi' % (mouse, date, rec)
        # cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))
        caImg = io.imread('/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/190101_f15/2019.03.21_000_suite2p_reg/suite2p/plane0/reg_tif/AVG2_output_1-902C.tif')
        # pdb.set_trace()
        caImg = (caImg - np.min(caImg)) * 255. / (np.max(caImg) - np.min(caImg))
        # caImg = (caImg - 15.)*255./(30.-15.)
        vLength = np.shape(framesRaw)[0]
        width = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        print('number of frames :', vLength, width, heigth)
        fps = 250
        fBehavior = 200
        fCa = 15
        afterEvery = fBehavior / fCa
        # pdb.set_trace()
        ##########################################################
        # create walking dynamics figure instance
        fig0 = plt.figure(figsize=(6.56, 3.5))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.23)
        # x1 = 0.
        # y1 = 0.
        ax0 = fig0.add_subplot(1, 1, 1)
        ax0 = plt.axes(xlim=(0, 30), ylim=(-10, 130))
        line0, = ax0.plot([], [], 'k-', lw=2)
        ax0.set_ylabel('position (cm)', fontsize=18)
        ax0.set_xlabel('time (s)', fontsize=18)

        ax0.spines['top'].set_visible(False)
        ax0.spines['right'].set_visible(False)
        #
        ax0.spines['bottom'].set_position(('outward', 10))
        ax0.xaxis.set_ticks_position('bottom')
        #
        # if xyInvisible[1]:
        # ax.spines['left'].set_visible(False)
        # ax.yaxis.set_visible(False)
        # else:
        ax0.spines['left'].set_position(('outward', 10))
        ax0.yaxis.set_ticks_position('left')

        ##########################################################
        # create calcium dynamics figure instance
        fig = plt.figure(figsize=(6.56, 3.5))
        plt.subplots_adjust(left=0.18, right=0.95, top=0.97, bottom=0.23)
        # x1 = 0.
        # y1 = 0.
        ax = fig.add_subplot(1, 1, 1)
        ax = plt.axes(xlim=(0, 30), ylim=(45, 170))
        line1, = ax.plot([], [], 'g-', lw=2)
        line2, = ax.plot([], [], 'b-', lw=2)
        line3, = ax.plot([], [], 'r-', lw=2)
        ax.set_ylabel('fluorescence (a.u.)', fontsize=18)
        ax.set_xlabel('time (s)', fontsize=18)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        #
        ax.spines['bottom'].set_position(('outward', 10))
        ax.xaxis.set_ticks_position('bottom')
        #
        # if xyInvisible[1]:
        # ax.spines['left'].set_visible(False)
        # ax.yaxis.set_visible(False)
        # else:
        ax.spines['left'].set_position(('outward', 10))
        ax.yaxis.set_ticks_position('left')
        # ax.set_ylabel('fluorescence (a.u.)')
        # Define the codec and create VideoWriter object
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        # fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        # fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width + 512, heigth + 350))
        nCa = 1
        ttime = []
        ffluo = [[], [], []]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        rois = [[464, 452, 11], [329, 99, 11], [318, 235, 11],  # [446,389,11]
                ]
        nPos = 0
        for i in np.arange(len(framesRaw)):
            output = np.zeros((heigth + 350, width + 512, 3), dtype="uint8")
            frame8bit = np.array(np.transpose(framesRaw[i]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            # cv2.circle(frame, (100, 100), 50, (0, 255, 0), -1)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i], 4), (10, 20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (i, (vLength - 1)), (10, 40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            output[0:heigth, 0:width, :] = frame
            ttime.append(midFrameTimes[i])

            #####################################
            mask0 = aTimes < midFrameTimes[i]
            line0.set_data(aTimes[mask0], angles[mask0] * 80. / 360.)
            fig0.canvas.draw()
            img0 = np.fromstring(fig0.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img0 = img0.reshape(fig0.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
            output[600:, :656, :] = img0
            ####################################
            # treatment of ca image
            frameCa8bit = np.array(caImg[nCa - 1], dtype=np.uint8)
            frameCa = cv2.cvtColor(frameCa8bit, cv2.COLOR_GRAY2BGR)

            for n in range(len(rois)):
                cv2.circle(frameCa, (rois[n][0], rois[n][1]), rois[n][2], colors[n], 2)
            cv2.putText(frameCa, '50 um', (35, 480), cv2.QT_FONT_NORMAL, 0.55, color=(220, 220, 220))
            cv2.line(frameCa, (30, 490), (94, 490), (220, 220, 220), 4)
            output[44:(44 + 512), (width):(width + 512), :] = frameCa
            # plot roi and add to movie
            for n in range(len(rois)):
                mask = np.zeros(shape=frameCa.shape, dtype="uint8")
                cv2.circle(mask, (rois[n][0], rois[n][1]), rois[n][2], (255, 255, 255), -1)
                maskedImg = cv2.bitwise_and(src1=frameCa, src2=mask)
                ffluo[n].append(np.mean(maskedImg[maskedImg > 0]))

            #########################################
            line1.set_data(ttime, ffluo[0])
            line2.set_data(ttime, ffluo[1])
            line3.set_data(ttime, ffluo[2])
            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            ##########################################
            output[600:, 656:(656 + 656), :] = img
            # pdb.set_trace()
            #
            out.write(output)
            if (i > nCa * afterEvery) and nCa < (len(caImg)):
                nCa += 1
                print(i, nCa, nCa * afterEvery)
        # Release everything if job is finished
        # cap.release()
        out.release()
        cv2.destroyAllWindows()
