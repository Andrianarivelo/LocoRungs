import platform
import h5py
from skimage import io
import tifffile as tiff
import numpy as np
import glob
#import sima
#import sima.segment
import time
import pdb
import cv2
import pickle

from tools.h5pyTools import h5pyTools
import tools.googleDocsAccess as googleDocsAccess
from tools.pyqtgraph.configfile import *


class extractSaveData:
    def __init__(self, mouse):

        self.h5pyTools = h5pyTools()

        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' or platform.node() == 'thinkpadX1B':
            laptop = True
            self.analysisBase = '/media/HDnyc_data/'
        elif platform.node() == 'otillo':
            laptop = False
            self.analysisBase = '/media/paris_data/'
        elif platform.node() == 'yamal' or platform.node() == 'cerebellum-HP':
            laptop = False
            self.analysisBase = '/media/HDnyc_data/'
            
        elif platform.node() == 'bs-analysis':
            laptop = False
            self.analysisBase = '/home/mgraupe/nyc_data/'
        else:
            print('Run this script on a server or laptop. Otherwise, adapt directory locations.')
            sys.exit(1)

        self.listOfAllExpts = googleDocsAccess.getExperimentSpreadsheet()

        # lilith backup
        self.dataBase     = '/media/invivodata/'
        # spetses backup
        self.dataBase2     = '/media/invivodata2/'

        # check if directory is mounted
        if not os.listdir(self.dataBase):
            os.system('mount %s' % self.dataBase)
        if not os.listdir(self.dataBase2):
            os.system('mount %s' % self.dataBase2)
        if not os.listdir(self.analysisBase):
            os.system('mount %s' % self.analysisBase)

        if int(mouse[:6]) >= 170829 :
            self.dataBase +=  'altair_data/dataMichael/'
        else:
            self.dataBase += 'altair_data/experiments/data_Michael/acq4/'
        self.dataBase2 += 'altair_data/dataMichael/'

        self.analysisLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % mouse
        self.figureLocation   = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/%s/' % mouse

        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.analysisLocation)
        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.figureLocation)

        fName = self.analysisLocation+'analysis.hdf5'
        #if os.path.isfile(fName):
        self.f = h5py.File(fName,'a')
        self.mouse = mouse

    ############################################################
    def __del__(self):
        self.f.flush()
        #self.f.close()
        print('on exit')

    ############################################################
    def getMotioncorrectedStack(self,folder,rec,suffix):
        allFiles = []
        for file in glob.glob(self.analysisLocation+'%s_%s_%s*_%s.tif' % (self.mouse,folder,rec,suffix)):
            allFiles.append(file)
            print(file)
        if len(allFiles)>1:
            print('more than one matching image file')
            sys.exit(1)
        else:
            motionCoor = np.loadtxt(allFiles[0][:-3]+'csv',delimiter=',',skiprows=1)
            imStack = io.imread(allFiles[0])
        return (imStack,motionCoor,allFiles[0])

    ############################################################
    def saveRungMotionData(self,mouse,date,rec,rungPositions):
        rec = rec.replace('/', '-')
        pickle.dump(rungPositions, open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'wb'))

    ############################################################
    def getRungMotionData(self,mouse,date,rec):
        rec = rec.replace('/', '-')
        rungPositions = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (mouse, date, rec), 'rb'))
        return rungPositions

    ############################################################
    def extractRoiSignals(self,folder,rec,tifFile):

        self.simaPath = self.analysisLocation+'%s_%s_%s' % (self.mouse, folder, rec)
        print(self.simaPath)
        if os.path.isdir(self.simaPath+'.sima'):
            print('sima dir exists')
            dataSet = sima.ImagingDataset.load(self.simaPath+'.sima')
        else:
            print('create sima dir')
            sequences = [sima.Sequence.create('TIFF', tifFile)]
            dataSet = sima.ImagingDataset(sequences, self.simaPath, channel_names=['GCaMP6F'])

        img = dataSet.time_averages[0][:,:,0]

        self.simaPath = self.simaPath+'.sima/'
        overwrite = True
        if os.path.isfile(self.simaPath + 'rois.pkl'):
            print('rois already exist')
            input_ = raw_input('rois traces exist already, do you want to overwrite? (type \'y\' to overwrite, any character if not) : ')
            if input_ != 'y':
                overwrite = False
        if overwrite:
            print('create rois with roibuddy')
            segmentation_approach = sima.segment.STICA(
                channel='GCaMP6F',
                components=1,
                mu=0.9,# weighting between spatial - 1 - and temporal - 0 - information
                #spatial_sep=0.8
            )
            print('segmenting calcium image ... ', end =" ")
            dataSet.segment(segmentation_approach, 'GCaMP6F_signals',planes=[0])
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
        #dataSet.export_signals('example_signals.csv', channel='GCaMP6F',signals_label='GCaMP6F_signals')
        #pdb.set_trace()

        #dataSet.signals('GCaMP6F')['GCaMP6F_signals']['mean_frame']

        # pdb.set_trace()
        roiLabels = []
        # extrace labels
        for n in range(len(rois)):
            roiLabels.append(rois[n].label)
            #print 'ROI label', n, rois[n].label

        return (img,rois,raw_signals)

    ############################################################
    def getRecordingsList(self,mouse,expDate='all',recordings='all'):

        folderRec = []
        if mouse in self.listOfAllExpts:
            #print mouse
            #print expDate, self.listOfAllExpts[mouse]['dates']
            expDateList = []
            #pdb.set_trace()
            # provide choice of which days to include in analysis
            if expDate=='some':
                print('Dates when experiments where performed with animal %s :' % mouse)
                didx = 0
                for d in self.listOfAllExpts[mouse]['dates']:
                    print('  %s %d' % (d,didx))
                    didx+=1
                print('Choose the dates for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\' : ', end='')
                daysInput = input()
                daysInputIdx = [int(i) for i in daysInput.split(',')]
                #print(daysInputIdx,daysInputIdx[0],type(daysInputIdx))
            # generate list of days to analyze
            if expDate == 'all' :
                for d in self.listOfAllExpts[mouse]['dates']:
                    #print(d)
                    expDateList.append(d)
            elif expDate=='some':
                didx = 0
                for d in self.listOfAllExpts[mouse]['dates']:
                    if didx in daysInputIdx:
                        #print(d)
                        expDateList.append(d)
                    didx+=1
            else:
                expDateList.append(expDate)
            print('Selected dates :', expDateList)
            #pdb.set_trace()
            if recordings=='some':
                # first show all recordings for a given date
                print('Choose recording to analyze')
                recIdx = 0
                for eD in expDateList:
                    dataFolders = self.listOfAllExpts[mouse]['dates'][eD]['folders']
                    for fold in dataFolders:
                        print(' ',fold)
                        self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                        #print(self.dataLocation)
                        if not os.path.exists(self.dataLocation):
                            #    print('experiment %s exists' % fold)
                            #else:
                            print('Problem, experiment does not exist')
                        # recList = OrderedDict()
                        recList = self.getDirectories(self.dataLocation)
                        for r in recList:
                            print('    %s  %s' %(r,recIdx))
                            recIdx+=1
                print('Choose the recordings for analysis by typing the index, e.g, \'1\', or \'0,1,3,5\' : ', end='')
                recInput = input()
                recInputIdx = [int(i) for i in recInput.split(',')]
            #pdb.set_trace()
            # then compile a list the selected recordings
            recIdx = 0
            for eD in expDateList:
                #print(expDateList, self.listOfAllExpts[mouse]['dates'], len(self.listOfAllExpts[mouse]['dates']))
                dataFolders = self.listOfAllExpts[mouse]['dates'][eD]['folders']
                #print eD, self.listOfAllExpts[mouse]['dates']
                for fold in dataFolders:
                    self.dataLocation = (self.dataBase2 + fold + '/') if eD >= '181018' else (self.dataBase + fold + '/')
                    if not os.path.exists(self.dataLocation):
                        #print('experiment %s exists' % fold)
                        #else:
                        print('Problem, experiment does not exist')
                    recList = self.getDirectories(self.dataLocation)
                    if recordings=='some':
                        tempRecList = []
                        for r in recList:
                            # only add recordings which were previously selected
                            if recIdx in recInputIdx:
                                if r[:-4] == 'locomotionTriggerSIAndMotor':
                                    subFolders = self.getDirectories(self.dataLocation+'/'+r)
                                    for i in range(len(subFolders)):
                                        tempRecList.append(r+'/'+subFolders[i])
                                else:
                                    tempRecList.append(r)
                            recIdx += 1
                        folderRec.append([fold,eD,tempRecList])
                    elif recordings=='all':
                        tempRecList = []
                        for r in recList:
                            if r[:-4] == 'locomotionTriggerSIAndMotor':
                                subFolders = self.getDirectories(self.dataLocation + '/' + r)
                                for i in range(len(subFolders)):
                                    tempRecList.append(r + '/' + subFolders[i])
                            else:
                                tempRecList.append(r)
                        folderRec.append([fold,eD,tempRecList])
                        #folderRec.append([fold,eD,recList])
        return (folderRec,dataFolders)

    ############################################################
    def getDirectories(self, location):
        #seqFolder = self.dataLocation + '/' + r
        subFolders = [os.path.join(o) for o in os.listdir(location) if os.path.isdir(os.path.join(location, o))]
        subFolders.sort()
        return subFolders


    ############################################################
    def checkIfDeviceWasRecorded(self,fold,eD,recording, device):
        recLocation =  (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        #print(recLocation)
        if os.path.exists(recLocation):
            print('%s constains %s , ' % (fold, recording), end =" ")
        else:
            print('Problem, recording does not exist')
            
        if device in ['CameraGigEBehavior','CameraPixelfly']:
            pathToFile = recLocation + device + '/' + 'frames.ma'
        elif device is 'PreAmpInput':
            pathToFile = recLocation + '%s.ma' % 'DaqDevice'
        elif device is 'frameTimes':
            pathToFile = recLocation + '%s/%s.ma' % ('CameraGigEBehavior','daqResult')
        else:
            pathToFile = recLocation + '%s.ma' % device
        print(pathToFile)

        if os.path.isfile(pathToFile):
            fData = h5py.File(pathToFile,'r')
            try:
                kk = fData['data']
            except KeyError:
                print('Device %s was acquired but NO data exists' % device)
                return (False,None)
            else:
                print('Device %s was acquired' % device)
                return (True,fData)
        else:
            print('Device %s was NOT acquired' % device)
            return (False,None)

    ############################################################
    def readRawData(self, fold, eD, recording, device, fData , readRawData = True):
        recLocation = (self.dataBase2 + '/' + fold + '/' + recording + '/') if eD >= '181018' else (self.dataBase2 + '/' + fold + '/' + recording + '/')
        #print(recLocation)
        if device == 'RotaryEncoder':
            # data from activity monitor
            if len(fData['data'])==1:
                angles = fData['data'][()][0]
            # data during high-res recording
            else:
                angles = fData['data'][4]
            times  = fData['info/1/values'][()]
            try :
                startTime = fData['info/2/DAQ/ChannelA'].attrs['startTime']
            except :
                startTime = os.path.getctime(recLocation)
                monitor = True
            else:
                monitor = False
            return (angles,times,startTime,monitor)

        elif device == 'AxoPatch200_2':
            current  = fData['data'][()][0]
            #pdb.set_trace()
            ephysTimes = fData['info/1/values'][()]
            #imageMetaInfo = self.readMetaInformation(recLocation)
            return (current,ephysTimes)

        elif device == 'Imaging':
            if readRawData:
                frames     = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = self.readMetaInformation(recLocation)
            return (frames,frameTimes,imageMetaInfo)
        elif device == 'CameraGigEBehavior':
            print('reading raw GigE data ...', end =" ")
            if readRawData:
                frames     = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            imageMetaInfo = [None] #self.readMetaInformation(recLocation)
            print('done')
            return (frames,frameTimes,imageMetaInfo)
        elif device == 'CameraPixelfly':
            print('reading raw Pixelfly data ...', end =" ")
            if readRawData :
                frames     = fData['data'][()]
            else:
                frames = np.empty([2, 2])
            frameTimes = fData['info/0/values'][()]
            xPixelSize = fData['info/3/pixelSize'].attrs['0']
            yPixelSize = fData['info/3/pixelSize'].attrs['1']
            xSize = fData['info/3/region'].attrs['2']
            ySize = fData['info/3/region'].attrs['3']
            imageMetaInfo = np.array([xSize*xPixelSize,ySize*yPixelSize,xPixelSize,yPixelSize])
            print('done')
            return (frames,frameTimes,imageMetaInfo)
        elif device =='PreAmpInput' or device=='frameTimes':
            values = fData['data'][()]
            valueTimes = fData['info/1/values'][()]
            return (values,valueTimes)

    ############################################################
    def readMetaInformation(self,filePath):
        # convert to um
        conversion = 1.E6
        config = readConfigFile(filePath + '.index')
        pixWidth  = config['.']['Scanner']['program'][0]['scanInfo']['pixelWidth'][0]*conversion
        pixHeight = config['.']['Scanner']['program'][0]['scanInfo']['pixelHeight'][0]*conversion
        dimensionXY =  np.array(config['.']['Scanner']['program'][0]['roi']['size'])*conversion
        position = np.array(config['.']['Scanner']['program'][0]['roi']['pos'])*conversion

        if pixWidth == pixHeight:
            deltaPix = pixWidth
        else:
            print('Pixel height and width are not equal.')
            sys.exit(1)

        #print r'dimensions (x,y, pixelsize in um) : ', np.hstack((dimensionXY,deltaPix))
        return np.hstack((position,dimensionXY,deltaPix))
        #self.h5pyTools.createOverwriteDS(dataGroup,dataSetName,hstack((dimensionXY,deltaPix)))

    ############################################################
    def saveBehaviorVideoData(self,groupNames,frames,expStartTime,expEndTime,imageMetaInfo):
        # self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        (test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,groupNames)
        #self.h5pyTools.createOverwriteDS(grpHandle,'behaviorFrames',len(frames))
        self.h5pyTools.createOverwriteDS(grpHandle,'startExposure', expStartTime)
        self.h5pyTools.createOverwriteDS(grpHandle,'endExposure', expEndTime)


    ############################################################
    def saveImageStack(self,frames,fTimes,imageMetaInfo,groupNames,motionCorrection=[]):
        (test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle,'caImaging',frames)
        self.h5pyTools.createOverwriteDS(grpHandle,'caImagingTime', fTimes)
        self.h5pyTools.createOverwriteDS(grpHandle,'caImagingField', imageMetaInfo)
        if len(motionCorrection)>1:
            self.h5pyTools.createOverwriteDS(grpHandle,'motionCoordinates', motionCorrection)

    ############################################################
    def readImageStack(self, groupNames):
        (grpName, test) = self.h5pyTools.getH5GroupName(self.f,groupNames)
        frames = self.f[grpName + '/caImaging'][()]
        fTimes = self.f[grpName + '/caImagingTime'][()]
        imageMetaInfo = self.f[grpName + '/caImagingField'][()]
        return (frames,fTimes,imageMetaInfo)

    ############################################################
    def saveWalkingActivity(self,angularSpeed,linearSpeed,wTimes,angles,aTimes,startTime,monitor,groupNames):
        (test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])
        self.h5pyTools.createOverwriteDS(grpHandle,'anglesTimes', np.column_stack((aTimes,angles)), ['startTime',startTime])

    ############################################################
    def getWalkingActivity(self,groupNames):
        (grpName,test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        print(grpName)
        angularSpeed = self.f[grpName+'/angularSpeed'][()]
        monitor = self.f[grpName+'/angularSpeed'].attrs['monitor']
        linearSpeed = self.f[grpName+'/linearSpeed'][()]
        wTimes = self.f[grpName+'/walkingTimes'][()]
        startTime = self.f[grpName+'/walkingTimes'].attrs['startTime']
        angleTimes = self.f[grpName+'/anglesTimes'][()]
        return (angularSpeed,linearSpeed,wTimes,startTime,monitor,angleTimes)

    ############################################################
    def getPawRungPickleData(self,date,rec):
        frontpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (self.mouse,date,rec), 'rb'))
        hindpawPos = pickle.load(open(self.analysisLocation + '%s_%s_%s_hindpawLocations.p' % (self.mouse,date,rec), 'rb'))
        rungs = pickle.load(open(self.analysisLocation + '%s_%s_%s_rungPositions.p' % (self.mouse,date,rec), 'rb'))
        return (frontpawPos,hindpawPos,rungs)

    ############################################################
    def saveTif(self,frames,mouse,date,rec,norm=None):
        img_stack_uint8 = np.array(frames, dtype=np.uint8)
        if norm:
            tiff.imsave(self.analysisLocation+'%s_%s_%s_ImageStack%s.tif' % (mouse, date, rec,norm), img_stack_uint8)
        else:
            tiff.imsave(self.analysisLocation+'%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def readTif(self,frames,mouse,date,rec):
        img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        tiff.imsave(self.analysisLocation+'%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    # (mouse, foldersRecordings[f][0], foldersRecordings[f][2][r], framesDuringRecording, expStartTime, expEndTime, imageMetaInfo)
    def saveBehaviorVideo(self, mouse, date, rec, framesRaw, expStartTime, expEndTime, imageMetaInfo):
        #[foldersRecordings[f][0],foldersRecordings[f][2][r],'walking_activity']
        self.saveBehaviorVideoData([date,rec,'behavior_video'], framesRaw,expStartTime, expEndTime, imageMetaInfo)
        midFrameTimes = (expStartTime + expEndTime)/2.
        #pdb.set_trace()
        #img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        #tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        # replace possible backslashes from subdirectory structure and
        rec = rec.replace('/','-')
        videoFileName = self.analysisLocation + '%s_%s_%s_raw_behavior_hfyu.avi' % (mouse, date, rec)
        #cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))


        vLength = np.shape(framesRaw)[0]
        width  = np.shape(framesRaw)[1]
        heigth = np.shape(framesRaw)[2]

        #print('number of frames :', vLength)
        fps    = 80
        # Define the codec and create VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        #fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # (*'XVID')
        # M J P G is working great !! 184 MB per video
        # H F Y U is working great !! 2.7 GB per video
        # M P E G has issues !! DON'T USE (frames are missing)
        # X V I D : frame 3001 missing and last nine frames are screwed
        # 0 (no compression) : frame 3001 missing last 2 frames are the same
        fourcc = cv2.VideoWriter_fourcc('H','F','Y','U')
        #fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # cv2.VideoWriter_fourcc(*'MPEG') # 'HFYU' is a lossless codec, alternatively use 'MPEG'
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (width, heigth))

        for i in np.arange(len(framesRaw)):
            frame8bit = np.array(np.transpose(framesRaw[i]), dtype=np.uint8)
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2BGR)
            cv2.putText(frame, 'time %s sec' % round(midFrameTimes[i],4), (10,20), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            cv2.putText(frame, 'frame %04d / %s' % (i,(vLength-1)), (10,40), cv2.QT_FONT_NORMAL, 0.6, color=(220, 220, 220))
            out.write(frame)
        # Release everything if job is finished
        #cap.release()
        out.release()
        cv2.destroyAllWindows()