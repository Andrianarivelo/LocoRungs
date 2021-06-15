# Tutorial on how to extract paw position using DeepLabCut

(written by [Jeremy Gabillet 29-Sept-2020](https://docs.google.com/document/d/1xdx-A_suzs5PG9FkaE4_w0-k_6pSUb6Wge6VVINpkmA/edit#heading=h.6mc7g4r1euo),
adapted by Michael Graupner Jan 2021)

DeepLabCut website ([DeepLabCut — adaptive motor control lab](https://www.mousemotorlab.org/deeplabcut)) is very resourceful. In short 
tutorial on how to use Deeplabcut can be found [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/UseOverviewGuide.md). A 
more in-depth guide with details on all the function parameters is [here](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/standardDeepLabCut_UserGuide.md).  
However, the features and tutorials may differ greatly from the installed version. Our current **DeepLabCut version is 2.1.10**. 




-----
#### Workflow of paw extraction 

1. **Connect to our computing server otillo:** <br>
   On your machine, open the terminal and connect to the computing unit through SSH with X11 forwarding. 
   This method displays DeepLabCut’s GUI faster and lighter than TeamViewer but changing window size can mess with its disposition. 
   Use -C flag for data compression if using a low bandwidth connection (not necessary at the lab): <br>
   `ssh -X mgraupe@otillo`    

1. **Activate Deeplabcut enviornment and move to folder :** Open a terminal and start the DeepLabCut conda 
   environment called **DLC-GPU** : <br>
   `conda activate DLC-GPU` <br>
   Move to the folder `analysis/DLC2_Projects/` where currently all DeepLabCut projects are stored : <br>
   `cd /home/mgraupe/analysis/DLC2_Projects/` <br>
   This folder contains a file called `dlc_project.py` which contains all typically used function calls. 
   
1. **Start Python and load DeepLabCut :** Start in **ipython** session and load the requires python libraries : <br>
   `ipython ` <br>
   `import deeplabcut` <br>
   `import glob` <br>
   `import pdb` <br>
   
1. **Assemble videos to be analyzed :** If you analyze different animals, their extracted video files are usually 
   placed in seperate folders on the analysis drive (`/media/paris_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData`) server. 
   Create list of string paths that contains all videos of the animals you want to analyze. DeepLabCut only accepts 
   paths under the format [‘path’]. [Glob](https://docs.python.org/3/library/glob.html) is a convenient tool to get all paths using wildcards. 
   ```
   videoBase = '/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'
   animals = ['201017_m99']
   videos = []

   for n in range(len(animals)):
       vids = glob.glob(videoBase+animals[n]+'/*.avi')
       videos.extend(vids)
   ```

1. **Start the DeepLabCut project :** 
   1. Start a **new** project with : <br>
    `config_path = deeplabcut.create_new_project('2021-Jan_PawExtractionTestMG','MichaelG', videos, copy_videos=False)` <br>
      The `config_path` variable contains the absolute path to the configure file  `config.yaml` which is located in the project folder.
   2. Work on an existing project with by creating the variable `config_path` pointing ot the existing config file : <br>
    `config_path = /home/mgraupe/analysis/DLC2_projects/[project]/config.yaml`
      
1. **Configure the Project :** Open the `config.yaml` file in a text editior and edit the project parameters.  
   1. In  particular you must add the list of bodyparts (or poins of interests) that you want to track. We use **Front left**, 
   **Front right**, **Hind right** and **Hind left** for the four paws. The location of the paws in the video image is 
   as shown below.
   1. Set the numframes2pick (number of frames extracted per individual video) variable. The total a number of images should 
   be of the order 100-200 frames (i.e., numframes2pick=20 for 10 videos). Too much is not that more effective
   1.  Also set the cropping parameters per video `[video_file_path] : crop: 21, 795, 175, 525` in the `config.yaml` file. 
       The cropping window can be determined with <br> 
   `deeplabcut.extract_frames(config_path, mode='manual', algo='uniform', crop=True)` <br>
   This command launches the manual frame extraction and allows to draw the cropping window. The routine can then be 
       aborted but the cropping parameters are still preserved, this will change the cropping parameters only for the loaded video. 
       Those parameters can be set as cropping parameters for remaining videos and as cropping parameters for videos training/analysis after setting cropping=true (cropping=true affect training speed) in the config.yaml file. <br>
       

1. **Extract frames for labelling :** A good training dataset should consist of a sufficient number of frames that capture
   the breadth of the behavior. This ideally implies to select the frames from different (behavioral) sessions, different 
   lighting and different animals, if those vary substantially (to train an invariant, robust feature detector).
   The function ``extract_frames`` extracts frames from all the videos in the project configuration file in order 
   to create a training dataset.  Launch the frame extraction with : <br>
   `deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', crop=True, userfeedback=False)` <br>
   The extracted frames can be inspected in the `labeled-data` subfolder which contains a folder for each video.
   
1. **Label Frames:** The toolbox provides a function label_frames which helps the user to easily label all the 
   extracted frames using an interactive graphical user interface (GUI). Launch with : <br>
   `deeplabcut.label_frames(config_path)` <br>
   A window appears, click on Load frames, where you and chose one after another the directories (per vidoe) of the 
   previously extracted frames. Right click to place and left click to move the label positions on the paws. Note
   to keep the same order of labelling for each frame (i.e., FL, FR, HR, HL). Check that the labels are correctly 
   saved by going to the next frame and back. <br>
   **Important :** <br>
   In general, invisible or occluded points should not be labeled by the user. They can simply be skipped by not applying the label anywhere on the frame. 
   The user needs to save the labels after all the frames from one of the videos are labeled by clicking the save button at the bottom right.
   It is advisable to consistently label similar spots (e.g., on a wrist that is very large, try to label the same location).
   
1. **Create Training Dataset :** Run this step on the machine where you are going to train the network. If you labelled the project on your computer, move the project folder   to otillo, then run the step below on that platform. This function combines the labeled datasets from all 
   the videos and splits them to create train and test datasets. The training data will be used to train the network, 
   while the test data set will be used for evaluating the network. <br>
   `deeplabcut.create_training_dataset(config_path, num_shuffles=1)` <br>
    At this step, for create_training_dataset you select the network you want to use, and any additional data augmentation 
    (beyond the defaults). You can set net_type and augmenter_type when you call the function. Specify an integer value in the num_shuffles parameters to keep track of DLC performance.
    In the `pose_config.yaml` (in the `dlc-models/.../train` subdirectory), you can modify some parameters :
    save_iters saves every n iteration the network state. If the training is stopped between 2 saved iterations, the last one 
      is saved (or none if before the first saved one). Bigger number means less space occupied but less snapshots.
    You can change the cropping variables if needed.
   
1. **Train the Network :** The function ‘train_network’ helps the user in training the network. It is used as follows: <br>
   `deeplabcut.train_network(config_path,shuffle=1,max_snapshots_to_keep=5, displayiters=1000,saveiters=10000,maxiters=400000)` <br>
   ● shuffle=Integer value specifying the shuffle index for the model to be trained. The default is set to 1. 
   
   ● displayiters. This sets how often the network iterations are displayed in the terminal. This variable is set in pose_config.yaml (called display_iters). However, you can overwrite it by
passing a variable. If None, the value from pose_config.yaml is used; otherwise, it is overwritten..
   
   ● saveiters. This sets how many iterations to save; every 50,000 is the default. This variable is set in pose_config.yaml (called save_iters). However, you can overwrite it by passing a
variable. If None, the value from the pose_config.yaml file is used; otherwise, it is overwritten.
   
   ● maxiters. This variable sets how many iterations to train for. This variable is set in pose_config.yaml. However, you can overwrite it by passing a variable. If None, the value from
there is used; otherwise it is overwritten. Default: None.
   
1. **Evaluate the Network:** It is important to evaluate the performance of the trained network. 
   This performance is measured by computing the mean average Euclidean error (MAE; which is proportional to the 
   average root mean square error) between the manual labels and the ones predicted by DeepLabCut. Evaluation is run with :<br>
   `deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)`
   
1. **Video Analysis:** The trained network can be used to analyze new videos. The user needs to first choose a checkpoint 
   with the best evaluation results for analyzing the videos. <br>
   `: deeplabcut.analyze_videos(config_path, shuffle=1, videos ,save_as_csv=True)` <br>
   The analyzed videos do not have to be the one extracted and you can use any other video with the trained network. 
   Ubuntu accepts only files under 144 characters long, the pickle file generated from the videos might be a problem so 
   be sure the videos don’t have a too long name (remove useless parts of the names but not the dates). 
   DeepLabCut is creating files in the same path for every iteration, thinking that the video has already been analyzed, 
   so you have to delete them for new video analysis. You can change the iteration number in config.yaml for a previous 
   one if the results are getting worse (in case of overtraining).
   Create labelled videos (for figures, presentations or checking for instance) with : <br>
   `deeplabcut.create_labeled_video(config_path, videos, save_frames = False)`

1. **Refine Network:** A single training is usually not enough to have satisfying results. 
   Refining the dataset if a crucial part and can be done with : <br>
   `deeplabcut.extract_outlier_frames(config_path, videos)` <br>
   The number of frames extracted is based on the `numframes2pick` variable, so you might want to adjust it. 
   The frames extracted are based on the jump of a label between 2 frames but other methods exist. <br>
   Refining the labels can then be done in the GUI : <br>
   `deeplabcut.refine_labels(config_path)` <br>
   The same way as during the labelling, a window opens, Load the frames. You can then move the labels to the desired location. 
   Even more than before, DeepLabCut sometimes doesn’t save the position correctly so go a frame forward then 
   backward again to be sure. <br>
   Merge the datasets with : <br>
   `deeplabcut.merge_datasets(config_path)` <br>
   Now you can run `deeplabcut.create_training_dataset`, then `deeplabcut.train_network`, etc.

1. **Adding new videos to analyze:** You can use the trained network to analyze new set of videos. 
   If you want to analyze videos from different animals, you should remove already analyzed videos from the project's videos folder. 
   Adding new videos to the project can be done with : 
   ```
   videoBase ='/media/HDnyc_data/data_analysis/in_vivo_cerebellum_walking/LocoRungsData/'
   
   animals = ['201017_m99'] 
   
   videos = []

   for n in range(len(animals)):
       vids = glob.glob(videoBase+animals[n]+'/*.avi')
       videos.extend(vids) 

   deeplabcut.add_new_videos(config_path,videos)
     ```
   Multiple animals can be added by separating the animal by a ',' `(animals = ['201017_m99', '210122_f83', '210214'])`
   Now you can run `deeplabcut.extract_frames(config_path, mode='automatic', algo='kmeans', crop=True, userfeedback=False)`, then `deeplabcut.label_frames(config_path)`, etc



