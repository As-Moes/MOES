
import os
import numpy as np 

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt

from . import DatasetLoader

ACTIONS = ['ABACAXI', 'ACOMPANHAR', 'ACONTECER', 'ACORDAR', 'ACRESCENTAR', 'ALTO', 'AMIGO', 'ANO', 'ANTES', 'APAGAR', 'APRENDER', 'AR', 'BARBA', 'BARCO', 'BICICLETA', 'BODE', 'BOI', 'BOLA', 'BOLSA', 'CABELO', 'CAIR', 'CAIXA', 'CALCULADORA', 'CASAMENTO', 'CAVALO', 'CEBOLA', 'CERVEJA', 'CHEGAR', 'CHINELO', 'COCO', 'COELHO', 'COMER', 'COMPARAR', 'COMPRAR', 'COMPUTADOR', 'DESTRUIR', 'DIA', 'DIMINUIR', 'ELEFANTE', 'ELEVADOR', 'ESCOLA', 'ESCOLHER', 'ESQUECER', 'FLAUTA', 'FLOR', 'MELANCIA', 'MISTURAR', 'NADAR', 'PATINS']

#--------------------------------------------------------------- 
# Plot confusion matrix
def plot_confusion_matrix(cm, num_classes, save_path):
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_percentage, cmap='Blues', xticklabels=ACTIONS, yticklabels=ACTIONS, fmt='g')
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
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
                           

    accuracy     = 100 * accuracy_score(all_labels, all_preds)
    balanced_acc = 100 * balanced_accuracy_score(all_labels, all_preds) 
    precision    = 100 * precision_score(all_labels, all_preds, average='macro')
    recall       = 100 * recall_score(all_labels, all_preds, average='macro')
    f1           = 100 * f1_score(all_labels, all_preds, average='macro')

    # Print the results
    print(f"Accuracy: {accuracy:2f}")
    print(f"Balanced Accuracy: {balanced_acc:.2f}")
    print(f"Precision: {precision:2f}")
    print(f"Recall: {recall:2f}")
    print(f"F1-score: {f1:2f}")

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, num_classes, os.path.dirname(model_path))
