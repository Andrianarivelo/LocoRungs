import glob
import os

oldNamePart = 'f84Apr23shuffle1'
toBeReplacedWith = 'f84Apr23shuffle2'
path = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/210122_f84/'


fList = glob.glob(path+'*'+oldNamePart+'*')

for i in range(len(fList)):
    print(fList[i])
    print(fList[i].replace(oldNamePart,toBeReplacedWith))
    os.system('mv %s %s' % (fList[i],fList[i].replace(oldNamePart,toBeReplacedWith)))
    print()

