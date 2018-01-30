# Locomotion data analysis

(start : Oct 2017)

The scripts and classes are used to extract/analyze/display locomotion and electrophysiological data recorded in the Cerebellum of mice walking on a runged treadmill. The recordings are performed in the Isabel Llano's lab starting Autum 2017.

The raw data has been recoreded with ACQ4 and is stored on lilith ('/home/labo/2pinvivo/backup/altair_data/') and backed up to spetses. Information about the experiments are furthermore stored in an google document ``Experiments Cerebellum awake : April 2017 onwards`` .

-----
#### Calcium imaging experiments

`getRawCalciumActivityStack.py` - The script reads the image stack recorded using the 2p scanning microscope. The raw stack is saved as tif file for subsequenct image registration using ImageJ.

`MocoMarco.ijm` - ImageJ macro which performs automatic image registration using the Moco plugin.

`getCalciumTracesDetermineRois.py` - The script reads the motion corrected image stack (previously generated with `getRawCalciumActivityStack.py`). The motion
corrected image stack is saved to the hdf5 file. Furthermore, roibuddy is used to extract fluorescent traces of individual rois. The fluorescent traces
are plotted in a figure.


**Typical work-flow to analyze calcium imaging data**

1. run `getRawCalciumActivityStack.py` to extract raw calcium images and save as tif file.
1. run `MocoMarco.ijm` macro in ImageJ to perform image registration.
1. run `getCalciumTracesDetermineRois.py` to determine rois with roibuddy and extract calcium traces of rois.

-----
#### Experiments with walking behavior recordings

`getWalkingActivity.py` - The script extracts the Rotary encoder data and saves it to hdf5 file.

`getRawBehaviorImagesSaveVideo.py` - The script extracts images recorded with the high-speed camera. The extracted images are saved as avi video for the tracking procedures.


**Typical work-flow to analyze experiments with walking behavior**

1. run `getRawBehaviorImagesSaveVideo.py` turn behavior camera recordings into movie.
1. run `analyzePawMovement.py` to track paw and rungs.

-----
#### Experiments to access motion artefacts

`getPixelflyImageStack.py` - The script extracts images recorded with the Pixelfly camera and saves them to a stacked tif file.

`plotMotionArtefacts.py` - The script uses the motion correction coordinates from the ImageJ Moco plugin. A figure is generated with comprises several recordings.


**Typical work-flow to analyze experiments with walking behavior**
1. run `getPixelflyImageStack.py` extract Pixelfly images and save them as tif
1. run `ImageJ` with the `MocoMacro.ijm` macro : generates x, y coordinates for motion correction
1. run `plotMotionArtefacts.py` generate summary figure over several recordings





