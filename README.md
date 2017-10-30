# Locomotion data analysis
==============================
(start : Oct 2017)

The scripts and classes are used to extract/analyze/display locomotion and later electrophysiological data recorded in the Cerebellum of mice walking on a runged treadmill. The recordings are performed in the Isabel Llano's lab starting Autum 2017. 

The raw data has been recoreded with ACQ4 and is stored on lilith ('/home/labo/2pinvivo/backup/altair_data/') and backed up to spetses. Information about the experiments are furthermore stored in an google document **Experiments Cerebellum awake : April 2017 onwards** .


## analyzeExperiment.py
----------------------------------

The script analyzes one experiments.. 

* analyzes pulse-train, pure-tones, AM noise and spontaneous recordings



## Example session 
---------------

Find here an analysis example of one experiment '141008_04' including 15 recordings.

    python analyzeExperiment.py 141008_04


or equivalently for the last line:
'''
python launch_analyze_experiment.py 574,589
'''
