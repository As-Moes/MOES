
# Libraries
import os
import yaml
from fabric import task

from src.preprocess import prepare_data

# Read tasks and paths from config file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

@task
def PrepareData(c):
    dataset_videos_path        = config['tasks']['PrepareData']['dataset_videos_path'] 
    dataset_landmarks_pathpath = config['tasks']['PrepareData']['dataset_landmarks_path']  
    prepare_data.convert_videos_to_csv(dataset_videos_path, dataset_landmarks_pathpath, show=False)
