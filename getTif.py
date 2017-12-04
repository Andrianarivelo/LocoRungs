import tools.extractData as extractData
import tifffile as tiff
import numpy as np
import pdb

experiment = '2017.11.14_002'
rec  = '2pScan_2.0umPix_Behavior_006'

eD = extractData.extractData(experiment)
rL = eD.getRecordingsList()
if rec in rL:
    data = eD.readData(rec,'Imaging')

frames     = data['data'].value 
frameTimes = data['info/0/values'].value

img_stack_uint8 = np.array(frames[:,:,:,0],dtype=np.uint8)
tiff.imsave('mouseImageStack_%s_%s.tif' % (experiment,rec), img_stack_uint8)
