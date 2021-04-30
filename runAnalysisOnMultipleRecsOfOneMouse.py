import tools.extractSaveData as extractSaveData
#import tools.dataAnalysis as dataAnalysis
#import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import os
import subprocess
import time
import pickle
import multiprocessing

saveDir = 'scriptRunHistory/'

runList = {0:{'mouse':'210120_m85','days':{'210323':{'recs910':'0'},\
                                           '210324':{'recs910':'0', 'recs820':'1'},\
                                           '210325':{'recs910':'0'},\
                                           '210326':{'recs910':'0'},\
                                           '210329':{'recs910':'0'},\
                                           '210330':{'recs910':'0'},\
                                           '210331':{'recs910':'0'},\
                                           '210401':{'recs910':'0', 'recs820':'1'},\
                                           '210402':{'recs910':'0', 'recs820':'1'},\
                                           }
              }
           }

script = 'extractPawTrackingOutliers'
do820analysis = False
commandHist = []

# loop over mice
for key,value in runList.items():
    print('mouse : ', value['mouse'])
    mouse = value['mouse']
    # loop over all recording days of one mouse
    for d,r in value['days'].items():
        recordings910 = [int(i) for i in r['recs910'].split(',')]
        #print(value['mouse'], d, '910rec:', r['recs910'])
        commandString9 = 'python %s.py -m %s -d %s -r %s' % (script,mouse,d,r['recs910'])
        print('running ', commandString9)
        (out,err) = subprocess.getstatusoutput(commandString9)
        print(out,err)
        commandHist.append([commandString9, out, err])
        if do820analysis:
            try:
                r['recs820']
            except:
                pass #print()
            else:
                recordings820 = [int(i) for i in r['recs820'].split(',')]
                #print('820rec:',r['recs820'])
                commandString8 = 'python %s.py -m %s -d %s -r %s' % (script, mouse, d, r['recs820'])
                print(commandString8)


ttt = time.strftime("%y-%m-%d")
sname = os.path.basename(__file__)
pickle.dump( commandHist, open( saveDir+"%s_%s_script-%s.p" % (ttt,sname,script), "wb" ) )