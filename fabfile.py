
# Libraries
import os
import yaml
from fabric import task

from src.preprocess import SampleAugment
from src.scripts import hand_tracker

# Read tasks and paths from config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

@task
def ProcessVideos(c):
    dataset_videos_path = config['tasks']['ProcessVideos']['dataset_videos_path'] 
    dataset_frames_path = config['tasks']['ProcessVideos']['dataset_frames_path']  
    SampleAugment.process_videos(dataset_videos_path, dataset_frames_path, show=True)
     
@task
def TrackHands(c):
    width = 1080
    height = 900
    hand_tracker.track_hands(width, height)
