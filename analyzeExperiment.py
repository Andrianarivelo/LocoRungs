import pdb
import time
import platform
import os
import h5py

#from membranePotential.membranePotential import *
#from membranePotential.tuning import *
#from membranePotential.extractSoundTraces import *
#from membranePotential.correlations import *
#from membranePotential.ivCurve import *
#from tools.openDocumentTools import *
from tools.pyqtgraph.configfile import *
from videoAnalysis import *


# default parameter , please note that changes here don't take effect
# if config file already exists
params= OrderedDict([
    ('ivCurveParameters',{ # iv-curve analysis
        'tGap': 0.010,     # leave gap in sec
        'tAverage': 0.02,  # duration in sec to average voltage
        }),
    ('artifactRemovalParameters', {    # artifact removal 
        'artifactJumpThreshold' : 20., # in mV
        'maxartifactDuration' : 0.1 ,  # period in sec before artifact used to determine mean and STD for the inserted white noise
        }),
    ('detrendingFilteringParameters', { # detrending and filtering
        'detrend' : 'new',              # 'new' for re-calculating, 'read' for reading formerly calculated traces
        'normalizationRange' : 0.05,    # in sec
        'normalizationRangeSlow' : 0.8, # in sec
        'cutoff_freq' : 150.,           # low-pass frequency in Hz, remove high-frequency noise by low-pass filtering recording
        }),
    ('extractSoundOnAndOffsetParameters', { # extract sound on- and offset
        'tBefore' : 0.15,                # extract time in sec before sound
        'tAfter' : 0.15,                 # extrac time in sec after sound
        'extractionCriterion' : None, # criterion deciding how the trigger signal is encodes sound stimuli presentations
        }),
    ('crossCorrelationParameters', { # calculate cross-correlation
        'correlationRange' : 2.,      # in sec +/-
        'fast' : True,                # if true, corrrelations are not calculated; speeds up script enormously
        }),
    ('fitIndividualResponsesParameters', { # fitting individual responses
        'timeFit' : 0.020 ,       # in sec, time before onset and after maximum to be included in fit
        'pThreshold' : 0.5,       # correlation threshold : above -> response, below -> faliure
        }),
    ('maxMinWindowAverageResponseParameters', { # determine maximum and mininum window of average response
        # average in [maximum +/- maxWindow] will be used to extract tone-evoked response in single recordings
        'maxWindow' : 0.004 ,     # in sec
        'minWindow' : 0.004,       # in sec
        'nDurationMult' : 2,        # look for maximum in window of n*toneDuration
        }),
    ])
    



    
###############################################################################
def analyzeVideo(ExptDay,trial,dataLocation,fData):
    
    ##################################################
    eV = videoAnalysis(ExptDay,trial,dataLocation)
    eV.storeAsTiffFile(analysisLocation) # input argument Nframes = 
    eV.storeAsVideoFile(analysisLocation)

###############################################################################
def analyzePureTones(expData,dataChannel):
    
    ##################################################
    # general membrane potential operations, "treat_raw_data" before
    vm = extractHighSpeedVideo(expData,dataChannel)
    
    vm.findAndReplaceArtifacts(config['artifactRemovalParameters']['artifactJumpThreshold'],config['artifactRemovalParameters']['maxartifactDuration'])
    vm.detrendMembranePotential(config['detrendingFilteringParameters']['detrend'],config['detrendingFilteringParameters']['normalizationRange'],config['detrendingFilteringParameters']['normalizationRangeSlow'])
    vm.findUpAndDownStateTransitions()
    vm.lowPassFiltering(config['detrendingFilteringParameters']['cutoff_freq'])
    vm.calculatePowerAndCoherence()
    
    ##################################################
    extract = extractSoundTraces(expData,dataChannel,config['extractSoundOnAndOffsetParameters']['tBefore'],config['extractSoundOnAndOffsetParameters']['tAfter'],config['extractSoundOnAndOffsetParameters']['extractionCriterion'])
    
    extract.findSoundOnAndOffsets()
    extract.extractTracesBeforeDuringAfterSound()
    
    ##################################################
    # extract properties with respect to tuning to sound stimuli
    tune = tuning(expData,dataChannel)
    
    tune.averageEvokedToneResponses()
    tune.determineOnsetAndMaximumInToneAverage()
    tune.clusterTestSignificantResponse()
    tune.clusterTestSignificantResponsePerTone()
    tune.fitIndividualResponseToAverageToneResponse(config['fitIndividualResponsesParameters']['timeFit'],config['fitIndividualResponsesParameters']['pThreshold'],config['maxMinWindowAverageResponseParameters']['maxWindow'],config['maxMinWindowAverageResponseParameters']['minWindow'],config['maxMinWindowAverageResponseParameters']['nDurationMult'])
    tune.eventDetectionToCharacterizeResponses(config['fitIndividualResponsesParameters']['timeFit'],config['fitIndividualResponsesParameters']['pThreshold'],config['maxMinWindowAverageResponseParameters']['maxWindow'],config['maxMinWindowAverageResponseParameters']['minWindow'],config['maxMinWindowAverageResponseParameters']['nDurationMult'])
    
    #################################################
    # calculate correlations
    corr = correlations(expData,dataChannel)

    corr.globalCorrelations(config['crossCorrelationParameters']['correlationRange'],config['crossCorrelationParameters']['fast'])
    corr.correlationsBetweenToneEpochs()
    
###############################################################################
def analyzeNoise(expData,dataChannel):
    noiseType = 'noise'
    analyzeAMNoise(expData,dataChannel,noiseType)

###############################################################################
def analyzeFlatNoise(expData,dataChannel):
    noiseType = 'flatNoise'
    analyzeAMNoise(expData,dataChannel,noiseType)
    
###############################################################################
def analyzeAMNoise(expData,dataChannel,noiseType='AMnoise'):
       
    ##################################################
    # general membrane potential operations, "treat_raw_data" before
    vm = membranePotential(expData,dataChannel)
    
    vm.findAndReplaceArtifacts(config['artifactRemovalParameters']['artifactJumpThreshold'],config['artifactRemovalParameters']['maxartifactDuration'])
    vm.detrendMembranePotential(config['detrendingFilteringParameters']['detrend'],config['detrendingFilteringParameters']['normalizationRange'],config['detrendingFilteringParameters']['normalizationRangeSlow'])
    vm.findUpAndDownStateTransitions()
    vm.lowPassFiltering(config['detrendingFilteringParameters']['cutoff_freq'])
    vm.calculatePowerAndCoherence()
    
    ##################################################
    # extract AM noise on- and offset as well as traces
    extract = extractSoundTraces(expData,dataChannel,config['extractSoundOnAndOffsetParameters']['tBefore'],config['extractSoundOnAndOffsetParameters']['tAfter'],config['extractSoundOnAndOffsetParameters']['extractionCriterion'],noiseType)
    
    extract.findSoundOnAndOffsets()
    extract.extractTracesBeforeDuringAfterSound()
    
    ##################################################
    # extract properties with respect to tuning to sound stimuli
    tune = tuning(expData,dataChannel,noiseType)
    
    tune.averageEvokedNoiseResponses()
    tune.clusterTestSignificantResponse()
    
    #################################################
    # calculate correlations
    corr = correlations(expData,dataChannel,noiseType)

    corr.globalCorrelations(config['crossCorrelationParameters']['correlationRange'],config['crossCorrelationParameters']['fast'])
    corr.correlationsBetweenNoiseEpochs()
    

###############################################################################
def analyzeSpontaneous(expData,dataChannel):
    
    ##################################################
    # general membrane potential operations, "treat_raw_data" before
    vm = membranePotential(expData,dataChannel)
    
    vm.findAndReplaceArtifacts(config['artifactRemovalParameters']['artifactJumpThreshold'],config['artifactRemovalParameters']['maxartifactDuration'])
    vm.detrendMembranePotential(config['detrendingFilteringParameters']['detrend'],config['detrendingFilteringParameters']['normalizationRange'],config['detrendingFilteringParameters']['normalizationRangeSlow'])
    vm.findUpAndDownStateTransitions()
    vm.lowPassFiltering(config['detrendingFilteringParameters']['cutoff_freq'])
    vm.calculatePowerAndCoherence()
    
    #################################################
    # calculate correlations
    corr = correlations(expData,dataChannel)

    corr.globalCorrelations(config['crossCorrelationParameters']['correlationRange'],config['crossCorrelationParameters']['fast'])


###############################################################################
## main program 
###############################################################################
if __name__=="__main__":
    ##########################################################
    # start time of the analysis script
    start_time = time.time()
    
    ExptDay = '2017.10.09_000'
    trials = ['behavingMLI_000']
    
    ##########################################################
    # read in all the input arguments
    #if len(sys.argv[1]) > 1:
    #    expt = sys.argv[1]                  # the date of the recording in YYMMDD format
    #else:
    #   print "No experiment specified!"
    #    sys.exit(0)
    
    #try:
    #    sys.argv[2]
    #except IndexError:
    #    recordings = 'all'
    #else:
    #    recordingsInput = sys.argv[2]        # number of recording, can also be a list of recordings
    #    # split of the recording numbers in single integers
    #    recordingsStr = recordingsInput.split(',')
    #    recordings = [('rec_%02d' % int(x)) for x in recordingsStr]
    
    ##########################################################
    # determine location of data files and store location
    if platform.node() == 'thinkpadX1' :
        laptop = True
        dataBase     = '/media/invivodata/'
        analysisBase = '/media/HDnyc_data/'
    elif platform.node() == 'michael-server':
        laptop = False
        dataBase = '/media/mgraupe/nyc_data/'
    elif platform.node() == 'bs-analysis':
        laptop = False
        dataBase = '/home/mgraupe/nyc_data/'
    else:
        print 'Run this script on a server or laptop. Otherwise, adapt directory locations.'
        sys.exit(1)
    
    dataLocation  = dataBase + 'altair_data/dataMichael/'
    analysisLocation = analysisBase+'data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'

    ##########################################################
    # open the appropriate hdf5 file, config file and directory for the experiment
    fName = analysisLocation+str(ExptDay)+'_data.hdf5'
    
    if os.path.isfile(fName):
        # open existing file
        f = h5py.File(fName,'r+')
    else:
        # crate new file
        f = h5py.File(fName,'w')
    
    ###########################################################
    # extract number of recordings, loop over individual recordings
    #nExperiments = len(f.keys())
    # analyze all recordings in one experiment if no one is specified as an input parameter
    #if recordings == 'all':
    #    recordings = f.keys()
    
    nAnalyzed = 0
    nTotal = 0
    # loop over all recordings
    for t in trials:
        print ExptDay, t
        dL = dataLocation + str(ExptDay) + '/' + str(t) + '/'
        #print dL
        if os.path.isfile(dL + 'RotaryEncoder.ma'):
            print '==',t, 'RotaryEncoder ','======'
            
        if os.path.isdir(dL + 'CameraGigEBehavior'):
            print '==',t, 'CameraGigEBehavior ','======'
            analyzeVideo(ExptDay,t,dL+'CameraGigEBehavior/',f)
    
    duration = time.time() - start_time
    
    print "Analysis time:", duration, "seconds"

#del(everything)
