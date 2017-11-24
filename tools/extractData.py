import pdb
import time
import platform
import os
import h5py

#from glob import glob


class extractData:
    def __init__(self,experiment):
        # determine location of data files and store location
        if platform.node() == 'thinkpadX1' :
            laptop = True
            analysisBase = '/media/HDnyc_data/'
        elif platform.node() == 'otillo':
            laptop = False
            analysisBase = '/media/mgraupe/nyc_data/'
        elif platform.node() == 'bs-analysis':
            laptop = False
            analysisBase = '/home/mgraupe/nyc_data/'
        else:
            print 'Run this script on a server or laptop. Otherwise, adapt directory locations.'
            sys.exit(1)
            
        dataBase     = '/media/invivodata/'
        # check if directory is mounted
        if not os.listdir(dataBase):
            os.system('mount %s' % dataBase)
        if not os.listdir(analysisBase):
            os.system('mount %s' % analysisBase)
        self.dataLocation  = dataBase + 'altair_data/dataMichael/' + experiment + '/'
        self.analysisLocation = analysisBase+'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'
        
        if os.path.exists(self.dataLocation):
            print 'experiment exists'
        else:
            print 'Problem, experiment does not exist'
    
    def getRecordingsList(self):
        recList = [os.path.join(o) for o in os.listdir(self.dataLocation) if os.path.isdir(os.path.join(self.dataLocation,o))]
        #recList = glob(self.dataLocation + '*')
        return recList
        
    def readData(self,recording, device):
        recLocation = self.dataLocation + '/' + recording + '/'
        
        if os.path.exists(self.dataLocation):
            print 'recroding exists'
        else:
            print 'Problem, recording does not exist'
            
        if device != 'CameraGigEBehavior':
            pathToFile = recLocation + '%s.ma' % device
        else:
            pathToFile = recLocation + device + '/' + 'frames.ma'
        print pathToFile
        if os.path.isfile(pathToFile):
            fData = h5py.File(pathToFile,'r')
        else:
            print 'file does not exist'
            
        return fData
            
