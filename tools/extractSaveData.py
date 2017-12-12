import platform
import h5py
from skimage import io
import tifffile as tiff
import numpy as np
import glob
import sima
import sima.segment
import time
import pdb
import cv2

from tools.h5pyTools import h5pyTools
import tools.googleDocsAccess as googleDocsAccess
from tools.pyqtgraph.configfile import *


class extractSaveData:
    def __init__(self, mouse):

        self.h5pyTools = h5pyTools()

        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' :
            laptop = True
            self.analysisBase = '/media/HDnyc_data/'
        elif platform.node() == 'otillo':
            laptop = False
            self.analysisBase = '/media/mgraupe/nyc_data/'
        elif platform.node() == 'bs-analysis':
            laptop = False
            self.analysisBase = '/home/mgraupe/nyc_data/'
        else:
            print 'Run this script on a server or laptop. Otherwise, adapt directory locations.'
            sys.exit(1)

        self.dataBase     = '/media/invivodata/'
        # check if directory is mounted
        if not os.listdir(self.dataBase):
            os.system('mount %s' % self.dataBase)
        if not os.listdir(self.analysisBase):
            os.system('mount %s' % self.analysisBase)

        self.analysisLocation = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % mouse
        self.figureLocation   = self.analysisBase + 'data_analysis/in_vivo_cerebellum_walking/LocoRungsFigures/%s/' % mouse

        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.analysisLocation)
        if not os.path.isdir(self.analysisLocation):
            os.system('mkdir %s' % self.figureLocation)

        self.listOfAllExpts = googleDocsAccess.getExperimentSpreadsheet()

        fName = self.analysisLocation+'analysis.hdf5'
        #if os.path.isfile(fName):
        self.f = h5py.File(fName,'a')
        self.mouse = mouse

    ############################################################
    def __del__(self):
        self.f.flush()
        self.f.close()
        print 'on exit'

    ############################################################
    def getMotioncorrectedStack(self,folder,rec,suffix):
        allFiles = []
        for file in glob.glob(self.analysisLocation+'%s_%s_%s*_%s.tif' % (self.mouse,folder,rec,suffix)):
            allFiles.append(file)
            print(file)
        if len(allFiles)>1:
            print 'more than one matching image file'
            sys.exit(1)
        else:
            motionCoor = np.loadtxt(allFiles[0][:-3]+'csv',delimiter=',',skiprows=1)
            imStack = io.imread(allFiles[0])
        return (imStack,motionCoor,allFiles[0])

    ############################################################
    def extractRoiSignals(self,folder,rec,tifFile):

        self.simaPath = self.analysisLocation+'%s_%s_%s' % (self.mouse, folder, rec)
        print self.simaPath
        if os.path.isdir(self.simaPath+'.sima'):
            print 'sima dir exists'
            dataSet = sima.ImagingDataset.load(self.simaPath+'.sima')
        else:
            print 'create sima dir'
            sequences = [sima.Sequence.create('TIFF', tifFile)]
            dataSet = sima.ImagingDataset(sequences, self.simaPath, channel_names=['GCaMP6F'])

        img = dataSet.time_averages[0][:,:,0]

        self.simaPath = self.simaPath+'.sima/'
        overwrite = True
        if os.path.isfile(self.simaPath + 'rois.pkl'):
            print 'rois already exist'
            input_ = raw_input('rois traces exist already, do you want to overwrite? (type \'y\' to overwrite, any character if not) : ')
            if input_ != 'y':
                overwrite = False
        if overwrite:
            print 'create rois with roibuddy'
            segmentation_approach = sima.segment.STICA(
                channel='GCaMP6F',
                components=1,
                mu=0.9,# weighting between spatial - 1 - and temporal - 0 - information
                #spatial_sep=0.8
            )
            print 'segmenting calcium image ... ',
            dataSet.segment(segmentation_approach, 'GCaMP6F_signals',planes=[0])
            print 'done'
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
        dataSet.signals('GCaMP6F')['GCaMP6F_signals']['mean_frame']

        # pdb.set_trace()
        roiLabels = []
        # extrace labels
        for n in range(len(rois)):
            roiLabels.append(rois[n].label)
            #print 'ROI label', n, rois[n].label

        return (img,rois,raw_signals)

    ############################################################
    def getRecordingsList(self,mouse,expDate):

        if mouse in self.listOfAllExpts:
            print mouse
            print expDate, self.listOfAllExpts[mouse]['dates']
            if expDate in self.listOfAllExpts[mouse]['dates']:
                dataFolder = self.listOfAllExpts[mouse]['dates'][expDate]['folder']
                print expDate, dataFolder

        self.dataLocation = self.dataBase + 'altair_data/dataMichael/' + dataFolder + '/'

        if os.path.exists(self.dataLocation):
            print 'experiment %s exists' % dataFolder
        else:
            print 'Problem, experiment does not exist'
        #recList = OrderedDict()
        recList = [os.path.join(o) for o in os.listdir(self.dataLocation) if os.path.isdir(os.path.join(self.dataLocation,o))]
        #recList = glob(self.dataLocation + '*')
        recList.sort()
        return (recList,dataFolder)

    ############################################################
    def checkIfDeviceWasRecorded(self,recording, device):
        recLocation = self.dataLocation + '/' + recording + '/'
        
        if os.path.exists(self.dataLocation):
            print '%s exists' % recording
        else:
            print 'Problem, recording does not exist'
            
        if device != 'CameraGigEBehavior':
            pathToFile = recLocation + '%s.ma' % device
        else:
            pathToFile = recLocation + device + '/' + 'frames.ma'
        #print pathToFile

        if os.path.isfile(pathToFile):
            fData = h5py.File(pathToFile,'r')
            return (True,fData)
        else:
            print 'device %s was not acquired during recording %s' % (device,recording)
            return (False,None)

    ############################################################
    def readRawData(self, recording, device, fData):

        recLocation = self.dataLocation + '/' + recording + '/'
        if device == 'RotaryEncoder':
            # data from activity monitor
            if len(fData['data'])==1:
                angles = fData['data'].value[0]
            # data during high-res recording
            else:
                angles = fData['data'][4]
            times  = fData['info/1/values'].value
            try :
                startTime = fData['info/2/DAQ/ChannelA'].attrs['startTime']
            except :
                startTime = os.path.getctime(recLocation)
                monitor = True
            else:
                monitor = False
            return (angles,times,startTime,monitor)
        elif device == 'Imaging':
            frames     = fData['data'].value
            frameTimes = fData['info/0/values'].value
            imageMetaInfo = self.readMetaInformation(recLocation)
            return (frames,frameTimes,imageMetaInfo)
        elif device == 'CameraGigEBehavior':
            frames     = fData['data'].value
            frameTimes = fData['info/0/values'].value
            imageMetaInfo = self.readMetaInformation(recLocation)
            return (frames,frameTimes,imageMetaInfo)

    ############################################################
    def readMetaInformation(self,filePath):
        # convert to um
        conversion = 1.E6
        config = readConfigFile(filePath + '.index')
        pixWidth  = config['.']['Scanner']['program'][0]['scanInfo']['pixelWidth'][0]*conversion
        pixHeight = config['.']['Scanner']['program'][0]['scanInfo']['pixelHeight'][0]*conversion
        dimensionXY =  np.array(config['.']['Scanner']['program'][0]['roi']['size'])*conversion

        if pixWidth == pixHeight:
            deltaPix = pixWidth
        else:
            print 'Pixel height and width are not equal.'
            sys.exit(1)

        #print r'dimensions (x,y, pixelsize in um) : ', np.hstack((dimensionXY,deltaPix))
        return np.hstack((dimensionXY,deltaPix))
        #self.h5pyTools.createOverwriteDS(dataGroup,dataSetName,hstack((dimensionXY,deltaPix)))

    ############################################################
    def saveImageStack(self,frames,fTimes,imageMetaInfo,groupName,motionCorrection=[]):
        grp_name = self.f.require_group(groupName)
        self.h5pyTools.createOverwriteDS(grp_name,'caImaging',frames)
        self.h5pyTools.createOverwriteDS(grp_name,'caImagingTime', fTimes)
        self.h5pyTools.createOverwriteDS(grp_name,'caImagingField', imageMetaInfo)
        if len(motionCorrection)>1:
            self.h5pyTools.createOverwriteDS(grp_name,'motionCoordinates', motionCorrection)

    ############################################################
    def saveWalkingActivity(self,angularSpeed,linearSpeed,wTimes,startTime,monitor,groupNames):
        (test,grpHandle) = self.h5pyTools.getH5GroupName(self.f,groupNames)
        self.h5pyTools.createOverwriteDS(grpHandle,'angularSpeed',angularSpeed,['monitor',monitor])
        self.h5pyTools.createOverwriteDS(grpHandle,'linearSpeed', linearSpeed)
        self.h5pyTools.createOverwriteDS(grpHandle,'walkingTimes', wTimes, ['startTime',startTime])

    ############################################################
    def getWalkingActivity(self,groupNames):
        (grpName,test) = self.h5pyTools.getH5GroupName(self.f, groupNames)
        print grpName
        angularSpeed = self.f[grpName+'/angularSpeed'].value
        monitor = self.f[grpName+'/angularSpeed'].attrs['monitor']
        linearSpeed = self.f[grpName+'/linearSpeed'].value
        wTimes = self.f[grpName+'/walkingTimes'].value
        startTime = self.f[grpName+'/walkingTimes'].attrs['startTime']
        return (angularSpeed,linearSpeed,wTimes,startTime,monitor)


    ############################################################
    def saveTif(self,frames,mouse,date,rec):
        img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        tiff.imsave(self.analysisLocation+'%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)

    ############################################################
    def saveTif(self, framesRaw, mouse, date, rec):


        #img_stack_uint8 = np.array(frames[:, :, :, 0], dtype=np.uint8)
        #tiff.imsave(self.analysisLocation + '%s_%s_%s_ImageStack.tif' % (mouse, date, rec), img_stack_uint8)
        videoFileName = self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec)
        #cap = cv2.VideoCapture(self.analysisLocation + '%s_%s_%s_behavior.avi' (mouse, date, rec))

        width  = shape(framesRaw)[1]
        heigth = shape(framesRaw)[2]
        fps    = 20.
        #w = 480
        #h = 640

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # (*'XVID')
        out = cv2.VideoWriter(videoFileName, fourcc, fps, (height, width))

        ret = True
        for i in range(len(framesRaw)):
            #frameRaw = np.random.rand(w, h) * 255.

            frame8bit = np.array(framesRaw, dtype=np.uint8)
            # ret, frame = cap.read()
            frame = cv2.cvtColor(frame8bit, cv2.COLOR_GRAY2RGB)
            #if ret == True:
                # frame = cv2.flip(frame,0)

                # write the flipped frame
            out.write(frame)

            #   cv2.imshow('frame', frame)
            #    if cv2.waitKey(1) & 0xFF == ord('q'):
            #        break
            #else:
            #    break

        # Release everything if job is finished
        #cap.release()
        out.release()
        cv2.destroyAllWindows()