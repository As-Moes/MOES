
# Libraries
import os
import yaml
from fabric import task

from src.preprocess import SampleAugment
from src.preprocess import Keypoints2Dataset

from src.keypoints.mpLoader import MediaPipeLoader
from src.keypoints import KeypointsDetector

# Read tasks and paths from config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

#-------------------------------------------------------------------
    
@task
def ProcessVideos(c):
    dataset_videos_path = config['tasks']['ProcessVideos']['dataset_videos_path'] 
    dataset_frames_path = config['tasks']['ProcessVideos']['dataset_frames_path']  
    number_of_frames    = 30          # Number of frames to sample from each video
    size                = (640, 480)  # Size to resize the frames
    augment_factor      = 9           # Number of augmented frames per original frame 
    SampleAugment.process_videos(dataset_videos_path, dataset_frames_path, number_of_frames, size, augment_factor, show=False)
     
@task
def ExtractKeypoints(c):
    dataset_frames_path = config['tasks']['ExtractKeypoints']['dataset_frames_path']
    dataset_path        = config['tasks']['ExtractKeypoints']['dataset_path'] 
    modes               = ["coordinates", "angles_distances"]
    media_pipe_loader   = MediaPipeLoader() 
    Keypoints2Dataset.process_frames(dataset_frames_path, dataset_path, modes[0], media_pipe_loader, show=False)

#-------------------------------------------------------------------

@task
def LiveHandTrack(c):
    window_size        = (1080, 720)
    media_pipe_loader  = MediaPipeLoader()
    KeypointsDetector.live_hands_tracking(window_size, media_pipe_loader)
 
