
import os
import numpy as np
import pandas as pd

from ..utils import utils_files

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .LSTM import LSTMModel
from .LSTM import LSTMModelSimple

import matplotlib.pyplot as plt

#---------------------------------------------------------------
# Prepare Data 

def split_dataset(dataset_path, train_percentage, val_percentage, test_percentage):
    random_state = 77796983
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
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
 
def load_dataset(train_dataset_path, val_dataset_path, test_dataset_path, batch_size=32, use_all=False):
    print("Preparing datasets...")
    
    # Load from csv into pandas DataFrame
    train_df = pd.read_csv(train_dataset_path)
    val_df   = pd.read_csv(val_dataset_path)
    test_df  = pd.read_csv(test_dataset_path)

    # Extract features and labels
    train_features = train_df.iloc[:, :-1].values
    train_labels   = train_df.iloc[:, -1].values
    val_features   = val_df.iloc[:, :-1].values
    val_labels     = val_df.iloc[:, -1].values
    test_features  = test_df.iloc[:, :-1].values
    test_labels    = test_df.iloc[:, -1].values

    # Reshape features to their original shape
    train_features = train_features.reshape(train_features.shape[0], 30, -1)
    val_features   = val_features.reshape(val_features.shape[0], 30, -1)
    test_features  = test_features.reshape(test_features.shape[0], 30, -1)

    # One-hot encode labels
    one_hot_encoder     = OneHotEncoder(sparse_output=False)
    train_labels_onehot = one_hot_encoder.fit_transform(train_labels.reshape(-1, 1))
    val_labels_onehot   = one_hot_encoder.transform(val_labels.reshape(-1, 1))
    test_labels_onehot  = one_hot_encoder.transform(test_labels.reshape(-1, 1))
 
    # Convert to PyTorch tensors
    train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
    train_labels_tensor   = torch.tensor(train_labels_onehot, dtype=torch.float32)
    val_features_tensor   = torch.tensor(val_features, dtype=torch.float32)
    val_labels_tensor     = torch.tensor(val_labels_onehot, dtype=torch.float32)
    test_features_tensor  = torch.tensor(test_features, dtype=torch.float32)
    test_labels_tensor    = torch.tensor(test_labels_onehot, dtype=torch.float32)

    # Create TensorDataset objects
    train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
    val_dataset   = TensorDataset(val_features_tensor, val_labels_tensor)
    test_dataset  = TensorDataset(test_features_tensor, test_labels_tensor)

    # Create DataLoader objects 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Get net parameters
    input_size  = train_features.shape[2]
    num_classes = train_labels_onehot.shape[1]
 
    return [train_loader, val_loader, test_loader], input_size, num_classes

#---------------------------------------------------------------
# Train Model

def create_lstm_simple_model(input_size, num_classes, device):
    hidden_size = 128
    num_layers  = 2 
    model = LSTMModelSimple(input_size, hidden_size, num_layers, num_classes).to(device)  
    return model
    
def create_lstm_model(input_size, num_classes, device):
    hidden_size1 = 64
    hidden_size2 = 128
    hidden_size3 = 64
    fc1_size     = 64
    fc2_size     = 32
    dropout_prob = 0.4
    model = LSTMModel(input_size,hidden_size1,hidden_size2,hidden_size3,fc1_size,fc2_size,num_classes,dropout_prob).to(device)    
    return model

def train(train_dataset_path, val_dataset_path, test_dataset_path, output_path): 
    # Load dataset
    batch_size    = 16 
    
    # Hyperparameters
    num_epochs    = 100
    learning_rate = 0.0001
    patience      = np.inf
  
    # Load dataset
    loaders, input_size, num_classes = load_dataset(train_dataset_path, val_dataset_path, test_dataset_path, batch_size) 
    train_loader, val_loader, test_loader = loaders

    # Create model 
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model = create_lstm_model(input_size, num_classes, device)

    # Define loss function and optimizer
    loss_func     = nn.CrossEntropyLoss()
    optimizer     = optim.Adam(model.parameters(), lr=learning_rate)

    # Train model 
    train_losses = []
    val_losses   = [] 

    # Define a variable to track the best validation loss
    best_val_loss      = np.inf
    best_model_weights = None
    no_improvement_epochs = 0 
    for epoch in range(num_epochs):
        # print(len(train_loader)) 
        model.train()
        
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs  = inputs.to(device)
            labels  = labels.squeeze(1).long().to(device)
            outputs = model(inputs)
            loss    = loss_func(outputs, torch.max(labels, 1)[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() 
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs        = model(inputs)
                loss           = loss_func(outputs, torch.max(labels, 1)[1])
                val_loss      += loss.item() 
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            # print("Validation loss decreased. Saving the model weights...")
            best_val_loss      = val_loss
            best_model_weights = model.state_dict()
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= patience:
                print(f'Validation loss did not improve for {patience} consecutive epochs. Stopping training.')
                break

    # Create folder to save training results
    if(not os.path.exists(output_path)): os.makedirs(output_path)
    folders = utils_files.read_folders(output_path)
    results_folder = os.path.join(output_path, "run"+str(len(folders) + 1))
    if(not os.path.exists(results_folder)): os.makedirs(results_folder)

    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(results_folder, "losses.png"))
    plt.show()
    
    # Save train and val losses
    train_losses_file_path = os.path.join(results_folder, "train_losses.npy")
    val_losses_file_path = os.path.join(results_folder, "val_losses.npy") 
    np.save(train_losses_file_path, np.array(train_losses)) 
    np.save(val_losses_file_path, np.array(val_losses)) 

    # Save best model
    final_model_path = os.path.join(results_folder, "model_final.pth")
    best_model_path = os.path.join(results_folder, "model_best.pth")
    torch.save(model.state_dict(), final_model_path)
    torch.save(best_model_weights, best_model_path)

    # Save model hyperparameter
    hyperparameters_file_path = os.path.join(results_folder, "hyperparameters.txt")
    with open(hyperparameters_file_path, "w") as f:
        f.write(f"dataset: {os.path.dirname(train_dataset_path)}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"optimizer: {str(type(optimizer))}\n")
        f.write(f"loss_func: {str(type(loss_func))}\n")

def test(train_dataset_path, val_dataset_path, test_dataset_path, output_path): 
    # Load dataset
    loaders, input_size, num_classes = load_dataset(train_dataset_path, val_dataset_path, test_dataset_path, batch_size) 
    train_loader, val_loader, test_loader = loaders
    
    pass
