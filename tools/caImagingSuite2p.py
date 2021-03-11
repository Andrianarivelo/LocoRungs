#import cv2
#import sys
#import pdb
#import pickle
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
#import glob
#import json

#import suite2p
#import suite2p as s2p
import os
from ScanImageTiffReader import ScanImageTiffReader

#import tools.dataAnalysis as dataAnalysis


class caImagingSuite2p:
    def __init__(self, analysisLoc, figureLoc, showI = False):
        #  "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/170606_f37/170606_f37_2017.07.12_001_behavingMLI_000_raw_behavior.avi"
        self.analysisLocation = analysisLoc
        self.figureLocation = figureLoc
        #self.f = ff
        self.showImages = showI

        #self.ops = np.load('tools/ops1.npy').item()


    ############################################################
    def __del__(self):
        print('suite2p : on exit')

    ############################################################
    def decideWhichTiffFilesToUse(self,dataDir,tiffFiles,defaultNumber=5):
        specificList = []
        availableIdxs = []
        # first show available tiff files with index
        print('List of available tiff files with index :')
        for n in range(len(tiffFiles)):
            print(' ',tiffFiles[n],n)
        # in case of the default number of tiff-files per recording, no hand-selection is required
        if len(tiffFiles)==defaultNumber:
            ll = []
            nn = ''
            for n in range(len(tiffFiles)):
                ll.append(tiffFiles[n])
                nn+= str(tiffFiles[n][-5:-4])
            specificList.append([ll,nn])
        else: # pick the lists by hand in case of more than the default number of tiff-files
            nn = ''
            listIdxs = input('Enter indices of the tiff files to be analyzed together separated by a space, different lists are separated by coma (e.g. 1 2 3 4 5, 6):')   # Python 3
            separateLists = listIdxs.split(',')
            for i in range(len(separateLists)):
                sl = [int(s) for s in separateLists[i].split()]
                ll = []
                nn = ''
                for j in sl:
                    ll.append(tiffFiles[j])
                    nn+= str(tiffFiles[j][-5:-4])
                specificList.append([ll,nn])
        return specificList
    ############################################################
    def runSuite2pPipeline(self,pythonPath,dataDir,saveDir,tiffPaths=None):
        #fName = dataDir + '*.tif'
        #tiffsList = glob.glob(fName)
        # only run on specified tiffs
        os.system(pythonPath + ' tools/runSuite2p.py %s %s ' % (dataDir, saveDir) + ' '.join(tiffPaths))

    ############################################################
    def generateOverviewFigure(self,suite2pPath,tiffList,mouseID,recFolder):
        ops = np.load(suite2pPath+'suite2p/plane0/ops.npy',allow_pickle=True).item()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('animal %s, rec: %s' % (mouseID,recFolder))
        ax.imshow(ops['meanImg'],vmax=max(ops['meanImg'].flatten())*0.3)

        plt.savefig(suite2pPath + 'animal_%s_rec_%s.pdf' %(mouseID,recFolder))

    ############################################################
    def generateSummaryFigures(self):
        pass



