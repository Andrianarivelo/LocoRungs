import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import os
import commands
import time
import pickle
import multiprocessing

saveDir = 'scriptRunHistory/'

mouseList = [#'180107_m27',
             #'180107_f28',
             #'180107_f29',
             #'171227_m96',
             '171215_m2',
             '180112_m33',
             '180112_f36 ',
             '180124_f1',
             '180131_f2',
             '180201_f48',
             '180201_f49',
             '180203_f40',
             '180203_f41',
             '180201_m42',
             '180201_m43',
             '180201_m44',
             '180201_m45',
             ]

script = 'getRawBehaviorImagesSaveVideo'

commandHist = []

# loop over mice
for m in mouseList:
    eSD         = extractSaveData.extractSaveData(m)
    (recordings,dataFolder) = eSD.getRecordingsList(m) # get recordings for specific mouse and date
    # loop over recording dates of the current mouse
    for n in range(len(recordings)):
        # two or more recordings on the same day appear as two entries in recordings, can be skipped here
        if (n>0) and (recordings[n-1][1] == recordings[n][1]):
            continue
        else:
            comandString = 'python %s.py -m %s -d %s' % (script,m,recordings[n][1])
            print comandString
            #tp = os.system('pwd')
            #print tp
            (out,err) = commands.getstatusoutput(comandString)
            commandHist.append([comandString,out,err])
            #pdb.set_trace()

ttt = time.strftime("%y-%m-%d")
sname = os.path.basename(__file__)
pickle.dump( commandHist, open( saveDir+"%s_%s_script-%s.p" % (ttt,sname,script), "wb" ) )