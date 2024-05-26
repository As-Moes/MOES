
import cv2
import os
import numpy as np

from ..utils import utils_files
from ..utils import utils_cv

# Sample frames from a video using a normal distribution
def sample_frames_from_video(video_path, num_samples=15):
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
    frame_indices = np.random.normal(loc=mean, scale=std_dev, size=num_samples)
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
 
# Augment data using flip, rotation and translation
# Important to keep consistency within frames 
def augment_frames(frames, augment_factor):
    
    augmented_frames = []
    augmented_frames.append(frames)
    
    for i in range(augment_factor):
        modified_frames = []

        doFlip = np.random.randint(2)
        angle  = np.random.uniform(-10, 10)
        tx     = np.random.randint(-20, 21)
        ty     = np.random.randint(-20, 21) 
        
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

# Resize all frames
def resize_frames(frames_collection, new_size):
    resized_frames_collection = []    
    for frames in frames_collection:
        resized_frames = []
        for frame in frames:
            resized_frame = utils_cv.resize_to(frame, new_size)
            # utils_cv.show(resized_frame)
            resized_frames.append(resized_frame)
        resized_frames_collection.append(resized_frames)
    return resized_frames_collection

# Convert videos dataset to landmarks dataset
def convert_videos_to_csv(dataset_videos_path, dataset_landmarks_path, show=False):
    folder_paths = utils_files.read_folders(dataset_videos_path)
    # folder_paths.sort() 
    for folder_path in folder_paths:
        videos_paths, videos_names = utils_files.read_all_videos(folder_path+'/1')
        print(f"Folder {folder_path} has {len(videos_paths)} videos")
      
        for video_path in videos_paths:
            print(f"Processing {video_path}")
            frames           = sample_frames_from_video(video_path, 30) 
            augmented_frames = augment_frames(frames, 10) 
            finished_frames  = resize_frames(augmented_frames, (640, 480))
