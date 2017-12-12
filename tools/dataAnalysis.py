import time
import numpy as np


def getSpeed(angles,times,circumsphere):
    angleJumps = angles[np.concatenate((([False]),np.diff(angles)!=0.))]
    timePoints = times[np.concatenate((([False]),np.diff(angles)!=0.))]
    angularSpeed = np.diff(angleJumps)/np.diff(timePoints)
    linearSpeed = angularSpeed*circumsphere/360.
    speedTimes = timePoints[1:]
    return (angularSpeed,linearSpeed,speedTimes)

