import h5py
import numpy as np

f = h5py.File('/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/190409_st2/analysis.hdf5', 'r')
dset = f['2019.04.09_001']

max_speed = []
for i in range(len(list(dset.keys()))):
    max_speed.append(max(list(dset['locomotion_motor_00%s/walking_activity/linearSpeed' % i])))
max_speed = np.asarray(max_speed)
max_speed = np.vstack((np.asarray([10, 15, 20, 25, 30, 35, 40]), max_speed))
value = np.interp(9, max_speed[1], max_speed[0])  # Value in arduino to get 9 cm.s^-1

