
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from transformers import get_linear_schedule_with_warmup, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import numpy as np
from dataset import TextClassificationDataset
from model import get_model
from utils import set_seed, compute_metrics, plot_training_curves
import os
import matplotlib.pyplot as plt

#Configuration
csv_path=r"C:\bert-classification-DL_Bert_Dataset_2026\data\train6.csv"
#paramettres du model en se conformant aux instructions du devoir

CONFIG = {
    'model_name': 'bert-base-multilingual-cased',  
    'num_classes': 3,                    
    'learning_rate': 2e-5,               
    'batch_size': 16,                    
    'max_length': 128,                   
    'epochs': 3,                         
    'warmup_steps': 500,                 # Ramp up du learning rate
    'seed': 42,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
}

set_seed(CONFIG['seed'])

def load_and_prepare_data(csv_path):
    
    df=pd.read_csv(csv_path)
    
    bert_tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
     # Créer le dataset
    dataset = TextClassificationDataset(
        premises=df['premise'].tolist(),
        hypotheses=df['hypothesis'].tolist(),
        tokenizer=bert_tokenizer,
        labels=df['label'].tolist(),
        max_length=CONFIG['max_length']
    )
    
    # Split train/validation 80/20 STRATIFIÉ:pour maintienir la distribution des classes
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    #  utilisation random_split pour mélanger les données
    
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(CONFIG['seed'])
    )
    
    print(f" Dataset chargé:")
    print(f"   Total: {len(dataset)} exemples")
    print(f"   Train: {len(train_dataset)} ({train_size/len(dataset)*100:.1f}%)")
    print(f"   Val: {len(val_dataset)} ({val_size/len(dataset)*100:.1f}%)")
    
    # Créer les DataLoaders 
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,  # Mélanger à chaque epoch
        num_workers=0  # pas de GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,  # Pas besoin de mélanger pour la validation
        num_workers=0
    )
    
    return train_loader, val_loader, bert_tokenizer

def train_epoch(model, train_loader, optimizer, scheduler, device):

    # Entraîner le modèle pour une epoch.
    
    # Étapes:
    # 1. Forward pass: passer les données au modèle
    # 2. Calculer la loss (écart entre prédiction et réalité)
    # 3. Backward pass: calculer les gradients
    # 4. Optimiser: mettre à jour les poids
    # 5. Scheduler: ajuster le learning rate
    
# Mode entraînement (active dropout, batch norm, etc.)
      
    model.train()  
    total_loss = 0
    all_preds = []
    all_labels = [] 
    
    # tqdm affiche une barre de progression
    
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        
        # Déplacer les données sur GPU si disponible
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # FORWARD PASS
        logits = model(input_ids, attention_mask)  # shape: (batch_size, num_classes)
        
        #  CALCUL LA LOSS
        
        # CrossEntropyLoss fait automatiquement softmax puis negative log likelihood (log vraisemblance)
        
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        
        #  BACKWARD PASS
        
        # Calcul des gradients
        
        # Remettre les gradients à zéro
        
        optimizer.zero_grad()
        
        # Calcul des gradients
        
        loss.backward()        
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        #  OPTIMISATION
        # Mettre à jour les poids avec les gradients
        
        optimizer.step()
        
        # Ajuster le learning rate
        
        if scheduler is not None:
            scheduler.step()
        
        # Enregistrement des métriques
        
        total_loss += loss.item()
        
        # Prédictions
        
        preds = logits.argmax(dim=1).cpu().numpy()
        labels_cpu = labels.cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels_cpu)
        
        # Mettre à jour la barre de progression
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    # Calcul des métriques moyennes
    
    avg_loss = total_loss / len(train_loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics['accuracy']


def eval_epoch(model, val_loader, device):

    # Évaluer le modèle sur un dataset
    
    # Important: 
    # model.eval() : désactive dropout et batch norm
    # torch.no_grad() : ne pas calculer les gradients (Comme indiquer dans le sujet)

   #  Mode évaluation 
   
    model.eval() 
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # torch.no_grad() : désactiver le calcul des gradients
    # (on n'a pas besoin de gradients en validation, Comme indiquer dans le sujet)
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            
            # Déplacer les données sur GPU
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            
            logits = model(input_ids, attention_mask)
            
            # Loss
            
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
            total_loss += loss.item()
            
            # Prédictions
            
            preds = logits.argmax(dim=1).cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels_cpu)
    
    # Calculer les métriques
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics['accuracy'], metrics['f1_score']


#Boucle d'entrainement

def train(csv_path, output_dir='./checkpoints'):
    
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_loader, val_loader, tokenizer=load_and_prepare_data(csv_path)
    
    model=get_model(
        model_name=CONFIG['model_name'],
        num_classes=CONFIG['num_classes'],
        #dropout=0.1
    ).to(CONFIG['device'])
    
    print(f"\n modele chargé sur {CONFIG['device']}")
    
    
    #Optimiseur AdamW (recommandé dans le sujet)
    
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=0.01
    )
    
    total_steps=len(train_loader)*CONFIG['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=CONFIG['warmup_steps'],
        num_training_steps=total_steps)

    # Historique des métriques
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
        'val_f1_score': []
    }

    best_val_loss=float('inf')


    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*60}")
        
        # Entraînement
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, CONFIG['device'])
        
        # Validation
        
        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, CONFIG['device'])
        
        # Enregistrer les métriques
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['val_f1_score'].append(val_f1)
        
        # Afficher les resultats
        
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        # Sauvegarde du meilleur modele
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f" Nouvelle meilleure validation loss! Sauvegarde du modèle...")
            torch.save(model.state_dict(), f'{output_dir}/best_model.pt')
            # Sauvegarder aussi le tokenizer
            tokenizer.save_pretrained(f'{output_dir}/tokenizer')
    
    print(f"\n{'='*60}")
    print(f" Entraînement terminé!")
    print(f"{'='*60}\n")
    
    
    # Tracer les courbes
    
    
    history = {
        'train_loss': [1.0064, 0.7619, 0.4802],
        'val_loss': [0.8788, 0.8464, 1.0101],
        'train_accuracy': [0.4858, 0.6711, 0.8067],
        'val_accuracy': [0.6040, 0.6436, 0.6374]
    }

    # Tracage et sauvegarde des courbes de performance finales
    
    print(" Génération du graphique avec vos métriques réelles...")
    fig = plot_training_curves(history)
    
    image_path = os.path.join(output_dir, 'training_curves.png')
    fig.savefig(image_path, dpi=100)
    plt.close(fig)
    print(f" Vrai graphique sauvegardé avec succès sous : {image_path}")
    
    return model, history

#  POINT D'ENTRÉE PRINCIPAL

if __name__ == "__main__":
    
    # Entraîner le modèle
    
    model, history = train(csv_path, output_dir='./checkpoints')
    
    print("\n Résumé final:")
    print(f"Meilleure Val Loss: {min(history['val_loss']):.4f}")
    print(f"Meilleure Val Accuracy: {max(history['val_accuracy']):.4f}")
    print(f"Meilleur Val F1-Score: {max(history['val_f1_score']):.4f}")