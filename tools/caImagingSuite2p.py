import cv2
import sys
import pdb
import pickle
import numpy as np
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import glob
import json

import suite2p
from suite2p.run_s2p import run_s2p
from ScanImageTiffReader import ScanImageTiffReader

import tools.dataAnalysis as dataAnalysis


class caImagingSuite2p:
    def __init__(self, analysisLoc, figureLoc, ff, showI = False):
        #  "/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/170606_f37/170606_f37_2017.07.12_001_behavingMLI_000_raw_behavior.avi"
        self.analysisLocation = analysisLoc
        self.figureLocation = figureLoc
        self.f = ff
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
    def setSuite2pParameters(self,dataDir,saveDir,tiffPaths=None):
        #fName = dataDir + '*.tif'
        #tiffsList = glob.glob(fName)
        # only run on specified tiffs

        if tiffPaths is not None:
            tiffList = [tP.split('/')[-1] for tP in tiffPaths]
            print(dataDir,saveDir,tiffList)
        else:
            print(dataDir,saveDir)
            tiffList = []

        self.db = {
              'h5py': [], # a single h5 file path
              'h5py_key': 'caData',
              'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
              'data_path': [dataDir],
                                    # a list of folders with tiffs
                                    # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
              'save_path0': saveDir,
              'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
              'fast_disk': '/tmp/', # string which specifies where the binary file will be stored (should be an SSD)
              'tiff_list': tiffList, #tiffList # list of tiffs in folder * data_path *!
              'input_format': 'tif'
            }
        print(dataDir + tiffList[0])
        reader = ScanImageTiffReader(dataDir + tiffList[0])
        metaD = reader.metadata()
        sE2 = self.extractInfoFromMetaData(metaD,'SI.hChannels.channelSave')
        #print(sE1[0],sE2)
        #pdb.set_trace()
        if sE2 == '[1;2]':
            nChan = 2
            print('2 channels saved')
        elif sE2 == '1':
            nChan = 1
            print('1 channel saved')
        #with ScanImageTiffReader(dataDir + tiffList[0]) as reader:
        #    o = json.loads(reader.metadata())
        #    pdb.set_trace()
        #    print(o["RoiGroups"]["imagingRoiGroup"]["rois"]["scanfields"]["affine"])
        # set your options for running
        # overwrites the run_s2p.default_ops
        self.ops = {
            'fast_disk': '/tmp/', # used to store temporary binary file, defaults to save_path0 (set as a string NOT a list)
            'save_path0': saveDir, # stores results, defaults to first item in data_path
            'delete_bin': False, # whether to delete binary file after processing
            # main settings
            'nplanes' : 1, # each tiff has these many planes in sequence
            'nchannels' : nChan, # each tiff has these many channels per plane
            'functional_chan' : 1, # this channel is used to extract functional ROIs (1-based)
            'diameter':12, # this is the main parameter for cell detection, 2-dimensional if Y and X are different (e.g. [6 12])
            'tau':  0.7, # this is the main parameter for deconvolution : 0.7 is the value for GCaMP6f
            'fs': 30.,  # sampling rate (total across planes)
            # output settings
            'save_mat': False, # whether to save output as matlab files
            'combined': True, # combine multiple planes into a single result /single canvas for GUI
            # parallel settings
            'num_workers': 0, # 0 to select num_cores, -1 to disable parallelism, N to enforce value
            'num_workers_roi': -1, # 0 to select number of planes, -1 to disable parallelism, N to enforce value
            # registration settings
            'do_registration': True, # whether to register data
            'nimg_init': 200, # subsampled frames for finding reference image
            'batch_size': 200, # number of frames per batch
            'maxregshift': 0.1, # max allowed registration shift, as a fraction of frame max(width and height)
            'align_by_chan' : 1, # when multi-channel, you can align by non-functional channel (1-based)
            'reg_tif': False, # whether to save registered tiffs
            'subpixel' : 10, # precision of subpixel registration (1/subpixel steps)
            # cell detection settings
            'connected': True, # whether or not to keep ROIs fully connected (set to 0 for dendrites)
            'navg_frames_svd': 5000, # max number of binned frames for the SVD
            'nsvd_for_roi': 1000, # max number of SVD components to keep for ROI detection
            'max_iterations': 20, # maximum number of iterations to do cell detection
            'ratio_neuropil': 6., # ratio between neuropil basis size and cell radius
            'anatomical_only': 0, #
            'ratio_neuropil_to_cell': 3, # minimum ratio between neuropil radius and cell radius
            'tile_factor': 1., # use finer (>1) or coarser (<1) tiles for neuropil estimation during cell detection
            'threshold_scaling': 1., # adjust the automatically determined threshold by this scalar multiplier
            'max_overlap': 0.75, # cells with more overlap than this get removed during triage, before refinement
            'inner_neuropil_radius': 2, # number of pixels to keep between ROI and neuropil donut
            'outer_neuropil_radius': np.inf, # maximum neuropil radius
            'min_neuropil_pixels': 350, # minimum number of pixels in the neuropil
            # deconvolution settings
            'baseline': 'maximin', # baselining mode
            'win_baseline': 60., # window for maximin
            'sig_baseline': 10., # smoothing constant for gaussian filter
            'prctile_baseline': 8.,# optional (whether to use a percentile baseline)
            'neucoeff': .7,  # neuropil coefficient
          }

    ############################################################

    def runSuite2pPipeline(self):

        #print(tiffList)
        # pdb.set_trace()
        opsEnd = run_s2p(ops=self.ops, db=self.db)

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

    ############################################################
    def extractInfoFromMetaData(self,metaD,keyWord):
        #print(metaD)
        startIdx = metaD.find(keyWord)
        #eElem = metaD[startIdx]
        sE1 = metaD[(startIdx+len(keyWord)+2):].split('\n')
        #pdb.set_trace()
        sE2 = sE1[0].strip()
        return sE2

