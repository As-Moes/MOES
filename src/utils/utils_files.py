
import sys
import os

from pathlib import Path

# Get path of all subfolders inside a folder
def read_folders(parent_folder):
    parent_folder = Path(parent_folder)
    if not parent_folder.is_dir():
        sys.exit("[ERROR] Invalid path: Please provide a valid folder path")

    folder_paths = [str(subfolder) for subfolder in parent_folder.iterdir() if subfolder.is_dir()]
    return folder_paths

# Read into an array all files paths with specified extensions inside a folder
def read_files(path, extensions, sort=False):
    path = Path(path) 
    if not path.is_dir():
        sys.exit("[ERROR reading files - Invalid path] Please provide a valid folder path")      
   
    files_paths = []
    files_names = []
    
    for ext in extensions:
        files = list(path.glob('*.' + ext))
        for file_path in files:
            files_paths.append(str(file_path))
            files_names.append(file_path.name)
    if sort:
        files_paths, files_names = zip(*sorted(zip(files_paths, files_names)))
        
    return files_paths, files_names

# Read into an array the paths all images inside a folder
def read_all_images(path, sort=False):
    extensions = ['png', 'jpg', 'jpeg', 'tiff']
    return read_files(path, extensions, sort)

# Read into an array the paths of all videos inside a folder
def read_all_videos(path):
    extensions = ['mp4', 'avi', 'mov', 'mkv']
    return read_files(path, extensions)

