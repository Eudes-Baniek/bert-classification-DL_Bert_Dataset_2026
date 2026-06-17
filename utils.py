import torch
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f" Seed fixée à {seed}")

def compute_metrics(y_true, y_pred):
    """
    Calculer les métriques de classification.
    y_true (list): Labels vrais
    y_pred (list): Labels prédits
    dict: Dictionnaire avec accuracy et F1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # average='weighted' pour gérer les classes déséquilibrées
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_score': f1
    }


def plot_confusion_matrix(y_true, y_pred, classes=None):
    """
    Tracer une matrice de confusion.
    Utile pour voir quelles classes sont confondues.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Matrice de confusion')
    plt.ylabel('Label vrai')
    plt.xlabel('Label prédit')
    plt.tight_layout()
    return plt.gcf()


def plot_training_curves(history):
    """
    Tracer les courbes de loss et accuracy pendant l'entraînement.
    history (dict): Dictionnaire contenant train_loss, val_loss, etc.
    """
    # CORRECTION : On décommente cette ligne essentielle pour créer la figure et les axes !
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss par epoch')
    # Ajustement cosmétique pour afficher proprement 1, 2, 3 sur l'axe des époques
    axes[0].set_xticks(range(len(history['train_loss'])))
    axes[0].set_xticklabels([str(i+1) for i in range(len(history['train_loss']))])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_accuracy'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy par epoch')
    # Ajustement cosmétique pour afficher proprement 1, 2, 3 sur l'axe des époques
    axes[1].set_xticks(range(len(history['train_accuracy'])))
    axes[1].set_xticklabels([str(i+1) for i in range(len(history['train_accuracy']))])
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig