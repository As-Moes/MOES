
# Libraries
import os

import yaml
from fabric import task

from src.keypoints import KeypointsDetector
from src.keypoints.mpLoader import MediaPipeLoader
from src.preprocess import CutFrames, SampleAugment, Keypoints2Dataset

from src.train import DatasetLoader, Trainer, Tester

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
    random_state        = 77796983 
    show                = False
    SampleAugment.process_videos(dataset_videos_path,dataset_frames_path,number_of_frames,size,augment_factor,random_state,show)
     
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
    series_size         = 30  # Number of frames Samples
    train_percentage    = 0.8
    val_percentage      = 0.1
    test_percentage     = 0.1
    random_state        = 77796983
    DatasetLoader.split_dataset(dataset_path, train_percentage, val_percentage, test_percentage, series_size, random_state)

#-------------------------------------------------------------------

@task
def TrainModel(c):
    train_dataset_path   = config['tasks']['TrainModel']['train_dataset_path']
    val_dataset_path     = config['tasks']['TrainModel']['val_dataset_path']
    output_folder_path   = config['tasks']['TrainModel']['output_folder_path']
    series_size          = 30 
    Trainer.train(train_dataset_path, val_dataset_path, output_folder_path, series_size)

@task
def TestModel(c):
    test_dataset_path  = config['tasks']['TestModel']['test_dataset_path']
    model_path         = config['tasks']['TestModel']['model_path'] 
    series_size        = 30 
    Tester.test(test_dataset_path, model_path, series_size)

#-------------------------------------------------------------------

@task
def LiveHandTrack(c):
    window_size        = (1080, 720)
    media_pipe_loader  = MediaPipeLoader()
    KeypointsDetector.live_hands_tracking(window_size, media_pipe_loader)
   

    
