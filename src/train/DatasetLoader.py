
import os
import pandas as pd

from ..utils import utils_files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
from torch.utils.data import DataLoader, TensorDataset

#-------------------------------------------------------------------------------------------------

def split_dataset_kfold(dataset_path, train_percentage, val_percentage, series_size, random_state):
    print(f"Splitting dataset {dataset_path}...") 

    if(train_percentage + val_percentage != 1):
        raise Exception("Sum of train and val percentages must be equal to 1")
    
    full_df         = pd.read_csv(dataset_path)
    origin_dataset  = full_df.iloc[:, -2]
    unique_origins  = origin_dataset.unique()

    folder = os.path.join(os.path.dirname(dataset_path), 'datasets_kfold')
    for origin in unique_origins:
        origin_dir = os.path.join(folder, f'origin_{origin}')
        os.makedirs(origin_dir, exist_ok=True)
   
        test           = full_df[origin_dataset == origin] 
        remaining_data = full_df[origin_dataset != origin]
        
        train, val     = train_test_split(remaining_data, test_size=val_percentage, random_state=42)
        
        train.to_csv(f'{origin_dir}/train.csv', index=False)
        val.to_csv(f'{origin_dir}/val.csv', index=False)
        test.to_csv(f'{origin_dir}/test.csv', index=False) 
    
    
# Split dataset in train, val and test
# Save number of classes and input size
def split_dataset(dataset_path, train_percentage, val_percentage, test_percentage, series_size, random_state):
    print(f"Splitting dataset {dataset_path}...") 

    if(train_percentage + val_percentage + test_percentage != 1):
        raise Exception("Sum of train, val and test percentages must be equal to 1")

    # Full dataset
    full_df = pd.read_csv(dataset_path)

    # Split dataset in train, val and test
    val_test_percentage   = val_percentage + test_percentage
    train_df, val_test_df = train_test_split(full_df, test_size=val_test_percentage, random_state=random_state) 
    val_proportion        = val_percentage/val_test_percentage
    val_df, test_df       = train_test_split(val_test_df, test_size=(1 - val_proportion), random_state=random_state) 
  
    # Save train, val and test datasets 
    dataset_folder = os.path.dirname(dataset_path)
    train_path     = os.path.join(dataset_folder, "train.csv")
    val_path       = os.path.join(dataset_folder, "val.csv")
    test_path      = os.path.join(dataset_folder, "test.csv") 
   
    print(f"Saving train, val and test datasets in {dataset_folder}...")
    train_df.to_csv(train_path, index=False)  
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Add number of classes and input size information to the dataset
    # on the top row
    num_classes  = len(set(full_df.iloc[:, -1].values)) 
    num_features = int((full_df.head(1).shape[1] - 2) / series_size)
    print(f"Number of classes: {num_classes} \nNumber of features: {series_size}") 
     
    utils_files.add_row_to_csv(train_path, [num_classes, num_features])
    utils_files.add_row_to_csv(val_path, [num_classes, num_features])
    utils_files.add_row_to_csv(test_path, [num_classes, num_features])

#-------------------------------------------------------------------------------------------------

# Get number of classes and input size
def get_dataset_info(dataset_path):
    num_classes, num_features  = utils_files.read_csv(dataset_path)[0]
    print(f"Number of classes: {num_classes} \nNumber of features: {num_features}")
    return int(num_classes), int(num_features)

# Get training data loaders
def get_training_loaders(train_dataset_path, val_dataset_path, batch_size=32, series_size=30):
    print("Getting training data...")

    # Convert to pandas DataFrame discarting the first row
    # the first row contains only the number of classes and the number of features    
    first_row = utils_files.read_csv(train_dataset_path)[0]    
    utils_files.remove_first_row_from_csv(train_dataset_path)
    utils_files.remove_first_row_from_csv(val_dataset_path)

    train_df = pd.read_csv(train_dataset_path)
    val_df   = pd.read_csv(val_dataset_path)

    utils_files.add_row_to_csv(train_dataset_path, first_row)
    utils_files.add_row_to_csv(val_dataset_path,   first_row)

    # Extract features and labels
    train_features = train_df.iloc[:, :-2].values
    train_labels   = train_df.iloc[:, -1].values
    val_features   = val_df.iloc[:, :-2].values
    val_labels     = val_df.iloc[:, -1].values

    # Reshape features to their original shape
    train_features = train_features.reshape(train_features.shape[0], series_size, -1)
    val_features   = val_features.reshape(val_features.shape[0], series_size, -1)
    
    # One-hot encode labels
    one_hot_encoder     = OneHotEncoder(sparse_output=False)
    train_labels_onehot = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))
    val_labels_onehot   = one_hot_encoder.transform(val_labels.reshape(-1, 1))
    
    # Convert features and labels to torch tensors
    train_features = torch.tensor(train_features, dtype=torch.float32)
    train_labels   = torch.tensor(train_labels_onehot, dtype=torch.long)
    val_features   = torch.tensor(val_features, dtype=torch.float32)
    val_labels     = torch.tensor(val_labels_onehot, dtype=torch.long)

    # Create data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset   = TensorDataset(val_features, val_labels)
    val_loader    = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_loader(test_dataset_path, series_size=30):
    print("Getting test data...")

    # Convert to pandas DataFrame discarting the first row
    # the first row contains only the number of classes and the number of features    
    first_row = utils_files.read_csv(test_dataset_path)[0]    
    utils_files.remove_first_row_from_csv(test_dataset_path)
    test_df = pd.read_csv(test_dataset_path)
    utils_files.add_row_to_csv(test_dataset_path, first_row)

    # Extract features and labels
    test_features  = test_df.iloc[:, :-2].values
    test_labels    = test_df.iloc[:, -1].values    

    # Reshape features to their original shape
    test_features = test_features.reshape(test_features.shape[0], series_size, -1)
    
    # Convert features and labels to torch tensors
    test_features = torch.tensor(test_features, dtype=torch.float32)
    test_labels   = torch.tensor(test_labels, dtype=torch.long)
   
    # Create data loaders
    test_dataset  = TensorDataset(test_features, test_labels)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)    

    return test_loader
