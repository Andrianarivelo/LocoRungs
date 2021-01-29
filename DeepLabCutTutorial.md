# Tutorial on how to extract paw position using DeepLabCut

(written by Jeremy Gabillet 29-Sept-2020, adapted by Michael Graupner Jan 2021)

DeepLabCut website ([DeepLabCut — adaptive motor control lab](https://www.mousemotorlab.org/deeplabcut)) is very resourceful. 
However, the features and tutorials may differ greatly from the installed version 
at the time (2.0.2). The tutorial can be customized with the help of the website 
depending on your preferences and the version you’re using.

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

The analysis below concerns experiments during which animal behavior has been recorded through the rotary recording on 
the treadmill and/or the animal was filmed with the high-speed camera (GigE). Global movement parameter is the overall 
walking progress of the animal during the recording. 

File | What it does
----- | ------
`getWalkingActivity.py` | The script extracts the Rotary encoder data and saves it to hdf5 file.
`plotRecordingOverviewPerAnimal.py` | The script uses the extracted Rotary encoder data and generates an overview figures.
`extractBehaviorCameraTiming.py` | Sets the ROI location for the LED used for synchronization and determines the exact timing of individual frames.  
`getRawBehaviorImagesSaveVideo.py` | The script extracts images recorded with the high-speed camera. The extracted images are saved as avi video for the tracking procedures.
`analyzePawMovement.py` | The script uses openCV to analyze the behavioral videos. Paw and rung locations are extracted.


**Typical work-flow to analyze global movement parameters during walking experiments**

1. run `getWalkingActivity.py` to read and save rotary encoder data. 
1. run `plotRecordingOverviewPerAnimal.py` to generate an overview figure. 

**Typical work-flow to analyze detailed paw movement during walking experiments**

1. run `extractBehaviorCameraTiming.py` sets the ROI location on the LED, extracts the luminosity trace and determines frame times. 
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


