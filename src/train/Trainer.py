
import os
import numpy as np
import pandas as pd

from . import DatasetLoader

from ..utils import utils_files

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from .LSTM import LSTMAmanda

import matplotlib.pyplot as plt

#---------------------------------------------------------------
# Plot training and validation loss
def plot_loss(train_losses, val_losses, results_folder):
    # Plot the training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(results_folder, "losses.png"))
    plt.show()
   
#---------------------------------------------------------------
# Train
def train(train_dataset_path, val_dataset_path, output_path, series_size):  
    # Hyperparameters
    batch_size    = 32
    num_epochs    = 300
    learning_rate = 0.0001

    hidden_size   = 512
    mask_prob     = 0.2
    dropout_prob  = 0.4
    l1_lambda     = 0.001
    
    patience      = np.inf
  
    # Load dataset
    num_classes, num_features = DatasetLoader.get_dataset_info(train_dataset_path) 
    train_loader, val_loader  = DatasetLoader.get_training_loaders(train_dataset_path, val_dataset_path, batch_size, series_size)

    # Create model, loss function and optimizer 
    device        = 'cuda' if torch.cuda.is_available() else 'cpu' 
    model         = LSTMAmanda(num_features, num_classes, hidden_size, mask_prob, dropout_prob, l1_lambda).to(device) 
    loss_func     = nn.CrossEntropyLoss()
    optimizer     = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)

    # Define a variable to track the validation loss
    best_val_loss         = np.inf
    best_model_weights    = None
    no_improvement_epochs = 0 

    # Actuallly Train model 
    train_losses  = []
    val_losses    = []  
    
    for epoch in range(num_epochs):
        model.train() 
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs  = inputs.to(device)
            labels  = labels.to(device)
            outputs = model(inputs)

            # Compute loss
            labels = torch.max(labels, 1)[1]
            loss    = loss_func(outputs, labels)
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
      
    # Plot train and val losses
    plot_loss(train_losses, val_losses, results_folder)

    # Save train and val losses
    train_losses_file_path = os.path.join(results_folder, "train_losses.npy")
    val_losses_file_path = os.path.join(results_folder, "val_losses.npy") 
    np.save(train_losses_file_path, np.array(train_losses)) 
    np.save(val_losses_file_path, np.array(val_losses)) 

    # Save final and best model
    final_model_path = os.path.join(results_folder, "model_final.pth")
    best_model_path = os.path.join(results_folder, "model_best.pth")
    torch.save(model, final_model_path)
    model.load_state_dict(best_model_weights)
    torch.save(model, best_model_path)

    # Save model hyperparameter
    hyperparameters_file_path = os.path.join(results_folder, "hyperparameters.txt")
    with open(hyperparameters_file_path, "w") as f:
        f.write(f"dataset: {os.path.dirname(train_dataset_path)}\n")
        f.write(f"batch_size: {batch_size}\n")
        f.write(f"num_epochs: {num_epochs}\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"optimizer: {str(type(optimizer))}\n")
        f.write(f"loss_func: {str(type(loss_func))}\n")
