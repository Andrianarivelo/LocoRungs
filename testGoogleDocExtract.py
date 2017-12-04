import tools.extractData as extractData
import tifffile as tiff
import numpy as np
import pdb

experiment = '2017.11.14_002'

eD = extractData.extractData(experiment)
eL = eD.getExperimentSpreadsheet()


