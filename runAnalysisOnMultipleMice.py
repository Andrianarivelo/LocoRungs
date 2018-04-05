import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import os
import commands
import time
import pickle

saveDir = 'scriptRunHistory/'

mouseList = ['180107_m27',
             '180107_f28',
             '180107_f29',
             '171227_m96',
             ]

script = 'getRawBehaviorImagesSaveVideo'

commandHist = []

# loop over mice
for m in mouseList:
    eSD         = extractSaveData.extractSaveData(m)
    (recordings,dataFolder) = eSD.getRecordingsList(m) # get recordings for specific mouse and date
    # loop over recording dates of the current mouse
    for n in range(len(recordings)):
        comandString = 'python %s.py -m %s -d %s' % (script,m,recordings[n][1])
        print comandString
        #tp = os.system('pwd')
        #print tp
        (out,err) = commands.getstatusoutput(comandString)
        commandHist.append([comandString,out,err])
        #pdb.set_trace()

ttt = time.strftime("%y-%m-%d")
sname = os.path.basename(__file__)
pickle.dump( commandHist, open( saveDir+"%s_runOf_%s.p" % (ttt,sname), "wb" ) )