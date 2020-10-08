# Locomotion data analysis

(start : Oct 2017)

The scripts and classes are used to extract/analyze/display locomotion, imaging and electrophysiological data recorded in the Cerebellum of mice walking on a
runged treadmill. The recordings are performed in the Isabel Llano's lab starting Autum 2017.

The raw data is been recorded with ACQ4 as well as ScanImage and is stored on lilith and spetses, the lab backup servers.

Information about the experiments are furthermore stored in an google document [***Experiments Cerebellum awake : April 2017 onwards***](https://docs.google.com/spreadsheets/d/14UbR4oYZLeGchwlGjw_znBwXQUvtaoW7-E-cmDQbr-c/edit?usp=sharing).
This google spreadsheet is read by the analysis scripts and used to access data of specific mice.

**Content of the documentation**

* [Analyze calcium imaging experiments](#calcium-imaging-experiments)

* [Analyze experiments with walking behavior recordings](#experiments-with-walking-behavior-recordings)

* [Experiments to access motion artefacts](#experiments-to-access-motion-artefacts)

* [Work-flow to generate overview figure](#work-flow-to-generate-overview-figure)

* [Extract paw positions with DeepLabCut](deepLabCut.md)

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

`plotRecordingOverviewPerAnimal.py` - The script uses the extracted Rotary encoder data and generates an overview figures.

`getRawBehaviorImagesSaveVideo.py` - The script extracts images recorded with the high-speed camera. The extracted images are saved as avi video for the tracking procedures.

`analyzePawMovement.py` - The script uses openCV to analyze the behavioral videos. Paw and rung locations are extracted.

**Typical work-flow to analyze global movement parameters during walking experiments**

1. run `getWalkingActivity.py` turn behavior camera recordings into movie.
1. run `plotRecordingOverviewPerAnimal.py` to track paw and rungs.

**Typical work-flow to analyze detailed paw movement during walking experiments**

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


-----
#### Work-flow to generate overview figure

The experiment overview figure contains information from the rotary enconder-, the video- and the calcium imaging recordings. It serves to have 
quick overview of the recordings, their implementation and the number of days an animal has been recorded. 
The overview figure can be generated with the `plotRecordingOverviewPerAnimal.py` script. Before this script can be run, data has to be extracted 
and some pre-analysis is required. In particular, the following scripts need to be run before generating the overiew : 

* `getWalkingActivity.py` : extract data from the rotary encoder recording and calculates speed of the wheel. 
* `getRawBehaviorImagesSaveVideo.py` : extract data from the high-speed video recordings, generates videos and saves timing information of the video recording.
* `getCalciumTracesDetermineRois.py` : runs Suite2p on the calcium-imaging recordings obtained with ScanImage. 

To complete the initial analysis, some more scripts need to be run on the data. 
* `extractRungLocation.py` : extracts the location of the rungs from the video generated with `getRawBehaviorImagesSaveVideo.py`
* `extractPawTrackingOutliers.py` : Uses the paw tracking data generated with DeepLabCut and remove mis-tacked paw positions based on the paw displacement 
between frames. Large, unrealistic displacements are removed. 
* `extractSwingStancePhase.py` : uses information from the wheel speed, paw position and rung position to separate paw trajectoris into swing and stance 
phases


