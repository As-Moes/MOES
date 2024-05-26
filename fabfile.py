
# Libraries
import os
import yaml
from fabric import task

from src.preprocess import prepare_data
from src.scripts import hand_tracker

@task
def PrepareData(c):
    prepare_data.convert_videos_to_csv()
 

@task
def TrackHands(c):
    width = 1080
    height = 900
    hand_tracker.track_hands(width, height)
