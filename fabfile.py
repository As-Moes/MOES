
# Libraries
import os
import yaml
from fabric import task

from src.preprocess import prepare_data

@task
def PrepareData(c):
    prepare_data.convert_videos_to_csv()
 

