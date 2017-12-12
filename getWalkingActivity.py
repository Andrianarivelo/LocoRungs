import tools.extractData as extractData
import tools.dataAnalysis as dataAnalysis
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys

mouse = '170927_m68'
expDate = '171115'

eD      = extractData.extractData()
expList = eD.getExperimentSpreadsheet()

dA      = dataAnalysis.dataAnalysis()

if mouse in expList:
    if expDate in expList[mouse]['dates']:
        recordings = eD.getRecordingsList(expList[mouse]['dates'][expDate]['folder'])
        print  expList[mouse]['dates'][expDate]
        print expList[mouse]['dates'][expDate]['folder']

tracks = []
for rec in recordings:
    data = eD.readData(rec,'RotaryEncoder')
    if data:
        (angles,aTimes) = eD.extractData(data,'RotaryEncoder')
        (speed, sTimes) = dA.getSpeed(angles,aTimes)
        tracks.append([rec,sTimes,speed])
        #plt.plot(sTimes,speed)





