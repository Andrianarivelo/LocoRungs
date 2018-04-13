from oauth2client import tools
tools.argparser.add_argument("-m","--mouse", help="specify name of the mouse", required=False)
tools.argparser.add_argument("-d","--date", help="specify name of the mouse", required=False)
args = tools.argparser.parse_args()

import tools.extractSaveData as extractSaveData
import tools.dataAnalysis as dataAnalysis
import tools.openCVImageProcessingTools as openCVImageProcessingTools
import pdb
import pickle


mouseD = '180107_m27'
expDateD = '180214'
#mouse = '171126_m90'
#expDate = '180118'
#mouse = '171218_f8'
#expDate = '180123'


# in case mouse, and date were specified as input arguments
if args.mouse == None:
    mouse = mouseD
else:
    mouse = args.mouse

if args.date == None:
    expDate = expDateD
else:
    expDate = args.date



miceList = [    \
'180107_m27',   \
'180107_f28',   \
'180107_f29',   \
'171227_m96',   \
'180112_m33',   \
'180124_f1',    \
'180131_f2',    \
'180203_f41',   \
'180201_m42',   \
'180201_m43',   \
'180201_m44',   \
'180201_m45']

#miceList = ['180107_m27']
BadVids = {}
WheelMask = [1500, 1205, 1625]
RungsLoc = [190, 0]
stop = 0

try:
    with open("badVideos.f", 'rb') as BadVideos_File:
        my_depickler = pickle.Unpickler(BadVideos_File)
        BadVids = my_depickler.load()
except: pass

for i, mouse in enumerate(miceList):
    print "############### MOUSE : %s" % mouse
    eSD         = extractSaveData.extractSaveData(mouse)
    (foldersRecordings,dataFolder) = eSD.getRecordingsList(mouse) # get recordings for specific mouse and date

    cv2Tools = openCVImageProcessingTools.openCVImageProcessingTools(eSD.analysisLocation,eSD.figureLocation,eSD.f,showI=True)

    for j, Session in enumerate(foldersRecordings) :
        
        folder =Session[0]
        
        for k, rec in enumerate(Session[2]): # for r in recordings[f][1]:
            
            #print rec
            (existence,fileHandle) = eSD.checkIfDeviceWasRecorded(Session[0],rec,'CameraGigEBehavior')
            #print existence
            if existence:
                try:
                    
                    with open(eSD.analysisLocation + '%s_%s_%s_frontpawLocations.p' % (mouse, folder, rec),'rb') as FrontPawloc_File: #Loads the data as a pickle file
                        my_depickler = pickle.Unpickler(FrontPawloc_File)
                        FrontPaws = my_depickler.load()
                        if len(FrontPaws) >= 800:
                            print "Video Already Analysed"
                            
                        elif len(FrontPaws) < 800 and not [folder, rec] in BadVids[mouse]: 
                            raise Exception("Error : the Video wasn't fully analysed (Too few values)")
                        
                        else: # [folder, rec] in BadVids[mouse]:
                            print "/!\ Video was marked as bad"

                except:
                    

                    stop, WheelMask, RungsLoc, badVideo = cv2Tools.trackPawsAndRungs(mouse,folder,rec, WheelMask=WheelMask, RungsLoc= RungsLoc)
                    
                    if badVideo:
                        print "\n\nVideo %s / %s was marked as badly tracked\n\n" % (folder, rec)
                        try : BadVids[mouse].append([folder, rec])
                        except: BadVids[mouse] = [[folder, rec]]

            if stop:
                break
        if stop:
            break
    if stop:
        break

print ('\n\nDone. Videos marked as badly tracked : ', BadVids)


with open("badVideos.f", 'wb') as BadVideos_File:
    my_pickler = pickle.Pickler(BadVideos_File)
    my_pickler.dump(BadVids)

