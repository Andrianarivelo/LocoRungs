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
            grp[dsname].attrs[at[0]]=at[1]

