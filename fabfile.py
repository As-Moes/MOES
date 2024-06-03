
# Libraries
import os

import yaml
from fabric import task

from src.keypoints import KeypointsDetector
from src.keypoints.mpLoader import MediaPipeLoader
from src.preprocess import CutFrames, SampleAugment, Keypoints2Dataset
from src.train import Trainer

# Read tasks and paths from config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

#-------------------------------------------------------------------

@task
def CutLastFrame(c):
    problematic_videos = [
        "data/CrossDatasetVideos/melancia/1/signbank_MELANCIA.mp4",
        "data/CrossDatasetVideos/misturar/1/signbank_MISTURAR.mp4",
        "data/CrossDatasetVideos/nadar/1/signbank_NADAR.mp4",
        "data/CrossDatasetVideos/patins/1/signbank_PATINS.mp4"
    ]
    CutFrames.cut_problematic_frame(problematic_videos)

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

@task 
def SplitDataset(c):
    dataset_path        = config['tasks']['SplitDataset']['dataset_path'] 
    train_percentage    = 0.8
    val_percentage      = 0.1
    test_percentage     = 0.1
    Trainer.split_dataset(dataset_path, train_percentage, val_percentage, test_percentage)

#-------------------------------------------------------------------

@task
def TrainModel(c):
    train_dataset_path = config['tasks']['TrainModel']['dataset_path'] + config['tasks']['TrainModel']['train']
    val_dataset_path   = config['tasks']['TrainModel']['dataset_path'] + config['tasks']['TrainModel']['val']  
    test_dataset_path  = config['tasks']['TrainModel']['dataset_path'] + config['tasks']['TrainModel']['test'] 
    output_path        = config['tasks']['TrainModel']['output_folder_path']
    Trainer.train(train_dataset_path, val_dataset_path, test_dataset_path, output_path)

@task
def TestModel(c):
    train_dataset_path = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['train']
    val_dataset_path   = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['val']  
    test_dataset_path  = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['test'] 
    model_path         = config['tasks']['TestModel']['model_path']
    Trainer.test(train_dataset_path, val_dataset_path, test_dataset_path, model_path)

#-------------------------------------------------------------------

@task
def LiveHandTrack(c):
    window_size        = (1080, 720)
    media_pipe_loader  = MediaPipeLoader()
    KeypointsDetector.live_hands_tracking(window_size, media_pipe_loader)
    
@task
def LiveSignDetect(c):
    window_size        = (1080, 720) # Size to render image
    frame_size         = (640, 480)  # Size to resize the frames
    threshold          = 0.5         # Minimum confidence to predict sign
    media_pipe_loader  = MediaPipeLoader()
    model_path         = config['tasks']['TestModel']['model_path']
    train_dataset_path = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['train']
    val_dataset_path   = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['val']  
    test_dataset_path  = config['tasks']['TestModel']['dataset_path'] + config['tasks']['TestModel']['test'] 
    KeypointsDetector.live_sign_detection(window_size, frame_size, threshold, media_pipe_loader, model_path, train_dataset_path, val_dataset_path, test_dataset_path)
