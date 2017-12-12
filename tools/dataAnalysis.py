import time
import numpy as np

class dataAnalysis:
    def __init__(self):
        self.time = time.time()

    def getSpeed(self,angles,times):
        angleJumps = angles[np.concatenate((([False]),np.diff(angles)>0.))]
        timePoints = times[np.concatenate((([False]),np.diff(angles)>0.))]
        speed = np.diff(angleJumps)/np.diff(timePoints)
        speedTimes = timePoints[1:]
        return (speed,speedTimes)

