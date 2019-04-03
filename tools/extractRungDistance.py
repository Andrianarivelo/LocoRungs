import numpy as np
import matplotlib.pyplot as plt
import pdb
import scipy

imgIdx = [0,1000,4000,5000]
mouse = '190101_f15'
path = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/%s/' % mouse
rungs = []
for i in range(len(imgIdx)):
    rungs.append(np.loadtxt(path+'rungs_img0_2019.03.19_000_locomotionTriggerSIAndMotor_000-000_image#%s.txt' % imgIdx[i] ,delimiter=';',skiprows=1))



distBelow = []
distAbove = []
posBelow = []
posAbove = []

for i in range(len(imgIdx)):
    if i == 0:
        above = rungs[i][0,5]
        below = rungs[i][0,2]
    distBelow.extend(np.diff(rungs[i][:,0]))
    distBelow.extend(np.diff(rungs[i][:,1]))
    distAbove.extend(np.diff(rungs[i][:,3]))
    distAbove.extend(np.diff(rungs[i][:,4]))
    posBelow.extend(rungs[i][:-1,0])
    posBelow.extend(rungs[i][:-1,1])
    posAbove.extend(rungs[i][:-1,3])
    posAbove.extend(rungs[i][:-1,4])

distBelow = np.asarray(distBelow)
distAbove = np.asarray(distAbove)
posBelow  = np.asarray(posBelow)
posAbove  = np.asarray(posAbove)

# sposBelow = np.copy(dd0[:,0])
# sposBelow  = np.concatenate((sposBelow,dd1[:,0]))
# sposBelow  = np.concatenate((sposBelow,dd2[:,0]))
# angle0 = np.arctan(dd0[:,2]-dd0[:,0])/(dd0[:,3]-dd0[:,1])
# angle1 = np.arctan(dd1[:,2]-dd1[:,0])/(dd1[:,3]-dd1[:,1])
# angle2 = np.arctan(dd2[:,2]-dd2[:,0])/(dd2[:,3]-dd2[:,1])
#
# angle = np.concatenate((angle0,angle1))
# angle = np.concatenate((angle,angle2))




maskBelow = posBelow > 0
maskAbove = posAbove > 0
#pdb.set_trace()
polycoeffsBelow = scipy.polyfit(posBelow[maskBelow], distBelow[maskBelow], 4)
print('below %s pix : ' % below ,polycoeffsBelow)
# [ 2.00710807, 1.09204496]
belowData = np.linspace(np.min(posBelow),np.max(posBelow),1000)
yFitBelow = scipy.polyval(polycoeffsBelow, belowData)

polycoeffsAbove = scipy.polyfit(posAbove[maskAbove], distAbove[maskAbove], 4)
print('above %s pix : ' % above ,polycoeffsAbove)
# [ 2.00710807, 1.09204496]
midData = np.linspace(np.min(posAbove),np.max(posAbove),1000)
yFitMid = scipy.polyval(polycoeffsAbove, midData)

fig = plt.figure()
ax0 = fig.add_subplot(1,1,1)
ax0.plot(belowData,yFitBelow)
ax0.plot(posBelow,distBelow,'o')
ax0.plot(midData,yFitMid)
ax0.plot(posAbove,distAbove,'o')

#ax1 = fig.add_subplot(2,1,2)
#ax1.plot(sposBelow,angle,'o')

plt.show()
