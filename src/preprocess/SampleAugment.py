
import cv2
import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from ..utils import utils_files
from ..utils import utils_cv

#-----------------------------------------------------------------------------------

# Sample frames from a video using a normal distribution
def sample_frames(rng, video_path, num_samples=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get total number of frames
    total_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps     = cap.get(cv2.CAP_PROP_FPS)
    
    # Generate N frame indices using a normal distribution
    # From Amanda Thesis
    mean          = total_frames / 2
    std_dev       = mean * 0.4
    frame_indices = rng.normal(loc=mean, scale=std_dev, size=num_samples)
    frame_indices = np.clip(frame_indices, 0, total_frames - 1).astype(int)
    frame_indices.sort()
 
    # Ensure that all frame indices are unique
    for i in range(1, len(frame_indices)):
        if frame_indices[i] <= frame_indices[i - 1]:
            frame_indices[i] = min(frame_indices[i - 1] + 1, total_frames - 1) 
 
    # print(len(frame_indices), total_frames, video_fpsj) 
    # Extract the frames corresponding to these indices
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # utils_cv.show(frame)
            frames.append(frame)
   
    # print(len(frames))
    if(len(frames) != num_samples):
        raise ValueError("Error sampling frames from video")

    # Release the video capture object
    cap.release()
    return frames

# Resize all frames
def resize_frames(frames, new_size):
    resized_frames = []  
    for frame in frames:
        resized_frame = utils_cv.resize_to(frame, new_size)
        resized_frames.append(resized_frame)
    return resized_frames 

# Augment data using flip, rotation and translation
# Important to keep consistency within frames 
def augment_frames(rng, frames, augment_factor):
    
    augmented_frames = []
    augmented_frames.append(frames)
    
    for i in range(augment_factor):
        modified_frames = []

        doFlip = rng.integers(2)
        angle  = rng.uniform(-20, 20)
        tx     = rng.integers(-50, 51)
        ty     = rng.integers(-50, 51) 
        
        for frame in frames:
            frame = cv2.flip(frame, 1) if doFlip else frame

            rows, cols = frame.shape[:2]
            rotation_matrix          = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_frame            = cv2.warpAffine(frame, rotation_matrix, (cols, rows))

            translation_matrix       = np.float32([[1, 0, tx], [0, 1, ty]])
            translated_frame         = cv2.warpAffine(rotated_frame, translation_matrix, (cols, rows))

            modified_frames.append(translated_frame)
       
        augmented_frames.append(modified_frames)
    
    return augmented_frames

#-----------------------------------------------------------------------------------

# Display all frames
def display_frames_list(frames_list):
    for i in range(len(frames_list[0])):
        display = [frames_list[j][i] for j in range(len(frames_list))]
        utils_cv.show(cv2.hconcat(display))

# Process all videos
def process_videos(dataset_videos_path, dataset_frames_path, number_of_frames, size, augment_factor, random_state, show=False):
    # Set random seed
    rng = np.random.default_rng(seed=random_state)

    # Create dataset output folder if it doesn't exist
    if(os.path.exists(dataset_frames_path) == False): os.makedirs(dataset_frames_path)

    # Read the folders for all videos
    folder_paths = utils_files.read_folders(dataset_videos_path)
    folder_paths.sort() 

    # Iterate over all subfolders
    for i in tqdm(range(len(folder_paths))):
        folder_path = folder_paths[i]
        folder_name = folder_path.split('/')[-1]

        videos_paths, videos_names = utils_files.read_all_videos(folder_path+'/1')
        videos_names.sort()
        print()
        print(videos_names)
        print()
        continue
    
        # Create output folder 
        output_folder = os.path.join(dataset_frames_path, folder_name)
        if(os.path.exists(output_folder) == False): os.makedirs(output_folder)

        # Iterate over all videos 
        subfolder_index = 0
        print("Processing folder: ", folder_path)
        for j in tqdm(range(len(videos_paths))):
            video_path      = videos_paths[j]
            frames          = sample_frames(rng, video_path, number_of_frames)
            resized_frames  = resize_frames(frames, size) 
            augmented_frames_list = augment_frames(rng, resized_frames, augment_factor)
            if show: display_frames_list(augmented_frames_list) 
        
            # Save a set of augmented frames for each video 
            for frames in augmented_frames_list:
                output_subfolder = os.path.join(output_folder, str(subfolder_index))
                if(os.path.exists(output_subfolder) == False): os.makedirs(output_subfolder)
                    
                subfolder_index += 1 
                for frame_index in range(len(frames)):
                    output_path = os.path.join(output_subfolder, str(frame_index)+'.png')
                    utils_cv.write(frames[frame_index], output_path)                

