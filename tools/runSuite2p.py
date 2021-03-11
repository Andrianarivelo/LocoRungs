import numpy as np
import sys
import pdb
import suite2p as s2p
from ScanImageTiffReader import ScanImageTiffReader

#import tools.dataAnalysis as dataAnalysis

#print(sys.argv)
#pdb.set_trace()
############################################################
def extractInfoFromMetaData(metaD,keyWord):
    #print(metaD)
    startIdx = metaD.find(keyWord)
    #eElem = metaD[startIdx]
    sE1 = metaD[(startIdx+len(keyWord)+2):].split('\n')
    #pdb.set_trace()
    sE2 = sE1[0].strip()
    return sE2


dataDir = sys.argv[1]
saveDir = sys.argv[2]
tiffPaths = sys.argv[3:]

print(dataDir)
print(saveDir)
print(tiffPaths)

if tiffPaths is not None:
    tiffList = [tP.split('/')[-1] for tP in tiffPaths]
    print(dataDir,saveDir,tiffList)
else:
    print(dataDir,saveDir)
    tiffList = []

db = {
      'h5py': [], # a single h5 file path
      'h5py_key': 'caData',
      'look_one_level_down': False, # whether to look in ALL subfolders when searching for tiffs
      'data_path': [dataDir],
                            # a list of folders with tiffs
                            # (or folder of folders with tiffs if look_one_level_down is True, or subfolders is not empty)
      'save_path0': saveDir,
      'subfolders': [], # choose subfolders of 'data_path' to look in (optional)
      'fast_disk': saveDir, # string which specifies where the binary file will be stored (should be an SSD)
      'tiff_list': tiffList, #tiffList # list of tiffs in folder * data_path *!
      'input_format': 'tif'
    }

print(dataDir + tiffList[0])
reader = ScanImageTiffReader(dataDir + tiffList[0])
metaD = reader.metadata()
sE2 = extractInfoFromMetaData(metaD,'SI.hChannels.channelSave')
#print(sE1[0],sE2)
#pdb.set_trace()
if sE2 == '[1;2]':
    nChan = 2
    print('2 channels saved')
elif sE2 == '1':
    nChan = 1
    print('1 channel saved')


ops = s2p.default_ops() # load default ops

ops['nchannels'] = nChan # each tiff has these many channels per plane
ops['tau'] =  0.7 # this is the main parameter for deconvolution : 0.7 is the value for GCaMP6f
ops['fs'] = 30.  # sampling rate (total across planes)
ops['anatomical_only'] = 2 #


opsEnd = s2p.run_s2p(ops=ops, db=db)


