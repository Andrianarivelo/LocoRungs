# Locomotion data analysis

(start : Oct 2017)

The scripts and classes are used to extract/analyze/display locomotion and electrophysiological data recorded in the Cerebellum of mice walking on a runged treadmill. The recordings are performed in the Isabel Llano's lab starting Autum 2017.

The raw data has been recoreded with ACQ4 and is stored on lilith ('/home/labo/2pinvivo/backup/altair_data/') and backed up to spetses. Information about the experiments are furthermore stored in an google document ``Experiments Cerebellum awake : April 2017 onwards`` .

## Data extraction scripts

### Calcium imaging experiments

`getRawCalciumActivityStack.py` - The script reads the image stack recorded using the 2p scanning microscope. The raw stack is saved as tif file for subsequenc image registration using ImageJ.

`getCalciumTracesDetermineRois.py` - The script reads the motion corrected image stack (previously generated with `getRawCalciumActivityStack.py`). The motion
corrected image stack is saved to the hdf5 file. Furthermore, roibuddy is used to extract fluorescent traces of individual rois. The fluorescent traces
are plotted in a figure.


### Experiments with walking

`getWalkingActivity.py` - The script extracts the Rotary encoder data and saves it to hdf5 file.



## Plotting scripts

`getRawCalciumActivityStack.py` - The script reads the image stack recorded using the 2p scanning microscope. The raw stack is saved as tif file for subsequenc image registration using ImageJ.

`getCalciumTracesDetermineRois.py` - The script reads the motion corrected image stack (previously generated with `getRawCalciumActivityStack.py`). The motion
corrected image stack is saved to the hdf5 file. Furthermore, roibuddy is used to extract fluorescent traces of individual rois. The fluorescent traces
are plotted in a figure.

`getWalkingActivity.py` - The script extracts the Rotary encoder data and saves it to hdf5 file.

