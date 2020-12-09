#import ezodf
import numpy as np

class h5pyTools:
    def __init__(self):
            pass

    ############################################################
    # creates a new data-set if it doesn't exist, otherwise overwrites
    ############################################################
    def createOverwriteDS(self,grp,dsname,data,at=None):
        try:
            rec = grp.require_dataset(dsname, shape = np.shape(data), dtype = data.dtype, exact=False ) 
        # either different shape or not exisiting
        except TypeError:
            try:
                del grp[dsname]
            except KeyError:
                grp.create_dataset(dsname,data=data,dtype = data.dtype)
            else:
                grp.create_dataset(dsname,data=data,dtype = data.dtype)
        else:
            rec[:] = data
        # save attributes
        if at:
            if len(np.shape(at))==1:
                grp[dsname].attrs[at[0]]=at[1]
            else:
                for i in range(len(at)):
                    grp[dsname].attrs[at[i][0]]=at[i][1]

    ############################################################
    def getH5GroupName(self,f,groupNames):

        current_group = ''
        for i in range(len(groupNames)):
            if i == 0:
                grpHandle = f.require_group(groupNames[i])
            else:
                grpHandle = f[current_group].require_group(groupNames[i])
            current_group += groupNames[i] + '/'
        return (current_group[:-1],grpHandle)
