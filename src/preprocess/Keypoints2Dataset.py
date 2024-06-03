
import os
import cv2
import numpy as np
import pandas as pd

from ..keypoints.mpLoader import MediaPipeLoader
from ..keypoints import KeypointsDetector

from tqdm import tqdm
from ..utils import utils_files
from ..utils import utils_cv

#-----------------------------------------------------------------------------------
# Generate column names depending on mode
def columns_coordinates(dataset_path, data, labels):    
    handmark_names  = ["WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", 
                        "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP",
                        "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP",
                        "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP", "RING_FINGER_MCP",
                        "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
                        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP"] 
    columns = []
    for frame in range(len(data[0])):
        for handmark in handmark_names:
            columns.append(f"{handmark}_x_left_frame{frame}")
            columns.append(f"{handmark}_y_left_frame{frame}")
            columns.append(f"{handmark}_z_left_frame{frame}")
        for handmark in handmark_names: 
            columns.append(f"{handmark}_x_right_frame{frame}")
            columns.append(f"{handmark}_y_right_frame{frame}")
            columns.append(f"{handmark}_z_right_frame{frame}")
    columns.append("label")
    return columns
 
def columns_angles_distances(dataset_path, data, labels):
    columns = []
    for frame in range(len(data[0])):
        for i in range(26):
            columns.append(f"angle_{i}_left_frame{frame}")
        for i in range(26):
            columns.append(f"angle_{i}_right_frame{frame}")
        for i in range(37):
            columns.append(f"distance_{i}_frame{frame}") 
    columns.append("label")
    return columns
 
#-------------------------------------------------------------------------------------------------------

# Create dataset as csv and save it
def create_dataset(dataset_path, data, labels, columns):
    data = np.array(data) 
    print(data.shape)

    # Create dataset as csv and save it
    data    = data.reshape(len(data), -1) 
    dataset = np.hstack((data, np.array(labels).reshape(-1, 1).astype(int))) 
    df      = pd.DataFrame(dataset, columns=columns)
    
    if(os.path.exists(dataset_path) == False): os.makedirs(dataset_path)
  
    # Save full dataset
    full_dataset_path = os.path.join(dataset_path, "full_dataset.csv")  
    df.to_csv(full_dataset_path, index=False)
 
# Process all frames in dataset, extract keypoints and save them
# as a csv file
def process_frames(dataset_frames_path, dataset_path, mode, media_pipe_loader, show=False):

    if(mode != "coordinates" and mode != "angles_distances"):
        raise Exception("Mode must be 'coordinates' or 'angles_distances'")

    # Read all classes folders
    folder_paths = utils_files.read_folders(dataset_frames_path)
    folder_paths.sort()
   
    # Iterate over all classes folders 
    # For each sample get all features
    data   = []
    labels = [] 
    for i in tqdm(range(len(folder_paths))):
        folder_path     = folder_paths[i]
        subfolder_paths = utils_files.read_folders(folder_path)
        subfolder_paths.sort()
        print("Processing folder: ", folder_path) 
        for j in tqdm(range(len(subfolder_paths))):
            subfolder_path = subfolder_paths[j] 
            frames_paths, _ = utils_files.read_all_images(subfolder_path)
            features     = [] 
            for frame_path in frames_paths:
                frame = utils_cv.read(frame_path) 
                if mode == "coordinates":
                    frame_features = KeypointsDetector.extract_keypoints_raw(frame, media_pipe_loader, show) 
                elif mode == "angles_distances":
                    frame_features = KeypointsDetector.extract_angles_distances(frame, media_pipe_loader, show)
                features.append(frame_features)
            data.append(features)
            labels.append(i)
 
    # Generate column names
    columns = []
    if mode == "coordinates":
        columns      = columns_coordinates(dataset_path, data, labels)
        dataset_path = os.path.join(dataset_path, "coordinates")
    elif mode == "angles_distances":
        columns = columns_angles_distances(dataset_path, data, labels)
        dataset_path = os.path.join(dataset_path, "angles_distances")

    # Create dataset as csv and save it
    create_dataset(dataset_path, data, labels, columns)     
