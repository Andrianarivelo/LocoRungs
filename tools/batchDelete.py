import glob
import os

path = '/media/otillo/analysis/DLC2_Projects/2021Jun_PawExtraction_m13_m14_f99_f70-Clarisse-2021-06-23/labeled-data/'
filesToDelete = '*.png'


dirList = glob.glob(path+'*/')


for i in range(len(dirList)):
    print(dirList[i])
    fList = glob.glob(dirList[i]+filesToDelete)
    for n in range(len(fList)):
        print('rm %s' % fList[n])
        os.system('rm %s' % fList[n])
    #os.system('mv %s %s' % (fList[i],fList[i].replace(oldNamePart,toBeReplacedWith)))
    #print()

