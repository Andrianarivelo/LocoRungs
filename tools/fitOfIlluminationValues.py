import numpy as np
import matplotlib.pyplot as plt
import pickle
import pdb

import scipy.optimize
#import matplotlib as mpl
#mpl.rcParams['agg.path.chunksize'] = 10000000


ill = pickle.load( open('../illuminatoinValues.p', 'rb' ) )

allValues = []

for i in range(len(ill)-1,len(ill)):
    allValues.extend(ill[i])
    plt.plot(ill[i][:,0],ill[i][:,1],'o',ms=1)


allValues = np.asarray(allValues)
allValues = np.sort(allValues,axis=0)
fitfunc = lambda p, x: x**p[0]
errfunc = lambda p, x, y: (fitfunc(p,x)-y)**2
p0 = 0.5
p1, success = scipy.optimize.leastsq(errfunc, p0,args=(allValues[:,0],allValues[:,1]))
rangeOfValues = np.linspace(min(allValues[:,0]),max(allValues[:,0]),1000)
corrfit = fitfunc(p1, rangeOfValues)
print('fitted exponent is :', p1)

plt.plot(rangeOfValues,corrfit)

plt.show()
pdb.set_trace()
