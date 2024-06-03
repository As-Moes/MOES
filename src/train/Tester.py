
import numpy as np 

import torch
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt

from . import DatasetLoader

#--------------------------------------------------------------- 
# Plot confusion matrix
def plot_confusion_matrix(cm, num_classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, cmap='Blues', fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show() 

#---------------------------------------------------------------
# Test
def test(test_dataset_path, model_path, series_size):

    # Load test dataset
    num_classes, num_features = DatasetLoader.get_dataset_info(test_dataset_path)  
    test_loader = DatasetLoader.get_test_loader(test_dataset_path, series_size)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    model  = torch.load(model_path)
    model  = model.to(device)
    model.eval()

    # Test
    num_correct = 0
    all_preds   = []
    all_labels  = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs  = inputs.to(device)
            label   = labels[0]
            all_labels.append(label)
            
            outputs = model(inputs)
            outputs = outputs.cpu().numpy() 
            pred    = np.argmax(outputs)
            all_preds.append(pred)
            
            if pred == label:
                num_correct += 1
                
    accuracy = (num_correct / len(all_preds)) * 100
    print(f'Accuracy on test set: {accuracy:.2f}%')

    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    print(f'Balanced Accuracy: {balanced_acc:.2f}')

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, num_classes)
