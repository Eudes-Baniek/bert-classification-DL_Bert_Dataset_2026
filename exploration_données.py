
# ÉTAPE 0 : EXPLORATION DU DATASET

#   Nombre total d'exemples et nombre de classes
#   Distribution des classes (si déséquilibre > 2:1, justifier la stratégie)
#   Longueur des textes (min, max, moyenne en tokens) - pour choisir max_length
#   Affichage au moins 5 exemples de textes avec leurs labels

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import Counter

print("="*80)
print("ÉTAPE 0 : ANALYSE COMPLÈTE DU DATASET")
print("="*80)

#  CHARGER LE DATASET

DATASET_PATH = 'data/train 6.csv' 

print(f"\n Chargement du dataset depuis: {DATASET_PATH}")

try:
    df = pd.read_csv(DATASET_PATH)
    print(f" Dataset chargé avec succès!")
except FileNotFoundError:
    print(f" ERREUR: Fichier non trouvé: {DATASET_PATH}")
    print("Assurez-vous que votre dataset est dans le dossier 'data/'")
    exit(1)


# STATISTIQUES


print("\n" + "="*80)
print("1. STATISTIQUES GÉNÉRALES")
print("="*80)

print(f"\n Dimensions du dataset:")
print(f"   Nombre total d'exemples: {len(df)}")
print(f"   Nombre de colonnes: {len(df.columns)}")
print(f"   Colonnes: {df.columns.tolist()}")

print(f"\nStructure du dataset (5 premières lignes):")
print(df.head())

# Identifier les colonnes text et label
# IMPORTANT: Votre CSV doit avoir 2 colonnes: 'text' et 'label'
# Sinon, adapter les noms ici:
TEXT_COLUMN = 'premise'      
LABEL_COLUMN = 'hypothesis'    

if TEXT_COLUMN not in df.columns or LABEL_COLUMN not in df.columns:
    print(f"\n ERREUR: Les colonnes '{TEXT_COLUMN}' et '{LABEL_COLUMN}' ne sont pas trouvées")
    print(f"   Colonnes disponibles: {df.columns.tolist()}")
    exit(1)

#  ANALYSE DES CLASSES (Distribution)

print("\n" + "="*80)
print("2. DISTRIBUTION DES CLASSES")
print("="*80)

class_counts = df[LABEL_COLUMN].value_counts().sort_index()
num_classes = len(class_counts)

print(f"\n Nombre de classes: {num_classes}")
print(f"\nDistribution des classes:")
for label, count in class_counts.items():
    percentage = (count / len(df)) * 100
    bar = " " * int(percentage / 2)
    print(f"   Classe {label}: {count:>6} exemples ({percentage:>5.1f}%) {bar}")

# Vérifier le déséquilibre
print(f"\n Vérification du déséquilibre:")
max_count = class_counts.max()
min_count = class_counts.min()
imbalance_ratio = max_count / min_count

print(f"   Classe la plus représentée: {max_count} exemples")
print(f"   Classe la moins représentée: {min_count} exemples")
print(f"   Ratio de déséquilibre: {imbalance_ratio:.2f}:1")

if imbalance_ratio > 2:
    print(f"\n ATTENTION: Déséquilibre > 2:1 détecté!")
    print(f"   Stratégie recommandée:")
    print(f"   - Utiliser weighted loss (CrossEntropyLoss avec class_weight)")
    print(f"   - Utiliser F1-score plutôt qu'accuracy")
    print(f"   - Considérer le resampling équilibré")
else:
    print(f"\n Déséquilibre acceptable (< 2:1)")

#  LONGUEUR DES TEXTES

print("\n" + "="*80)
print("3. LONGUEUR DES TEXTES (En caractères)")
print("="*80)

# Longueur en caractères
text_lengths_chars = df[TEXT_COLUMN].str.len()

print(f"\n Longueur en caractères:")
print(f"   Minimum: {text_lengths_chars.min()} caractères")
print(f"   Maximum: {text_lengths_chars.max()} caractères")
print(f"   Moyenne: {text_lengths_chars.mean():.2f} caractères")
print(f"   Médiane: {text_lengths_chars.median():.0f} caractères")
print(f"   Écart-type: {text_lengths_chars.std():.2f}")

# Percentiles
percentiles = [25, 50, 75, 90, 95, 99]
print(f"\n Percentiles:")
for p in percentiles:
    val = text_lengths_chars.quantile(p/100)
    print(f"   {p}e percentile: {int(val)} caractères")

#  LONGUEUR EN TOKENS (IMPORTANT!)

print("\n" + "="*80)
print("4. LONGUEUR DES TEXTES (En TOKENS BERT) ⭐ CRUCIAL")
print("="*80)

# Charger le tokenizer BERT
# ADAPTER LE MODÈLE À VOTRE LANGUE:
# - 'bert-base-uncased' pour l'anglais
# - 'camembert-base' pour le français
# - 'bert-base-multilingual-cased' pour multilingue

MODEL_NAME = 'bert-base-uncased'  # À adapter!

print(f"\n Chargement du tokenizer: {MODEL_NAME}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f" Tokenizer chargé!")
except:
    print(f" Erreur lors du chargement du tokenizer")
    exit(1)

# Tokenizer chaque texte et compter les tokens
print(f"\n Tokenization en cours... (cela peut prendre quelques minutes)")

token_counts = []
for idx, text in enumerate(df[TEXT_COLUMN]):
    # Tokenizer le texte
    tokens = tokenizer.encode(text, truncation=False)  # Pas de troncature pour le moment
    token_counts.append(len(tokens))
    
    # Afficher la progression chaque 1000 exemples
    if (idx + 1) % 100 == 0:
        print(f"   {idx + 1}/{len(df)} exemples tokenizés...")

print(f" Tokenization terminée!")

token_counts = np.array(token_counts)

print(f"\n Longueur en tokens BERT:")
print(f"   Minimum: {token_counts.min()} tokens")
print(f"   Maximum: {token_counts.max()} tokens")
print(f"   Moyenne: {token_counts.mean():.2f} tokens")
print(f"   Médiane: {np.median(token_counts):.0f} tokens")
print(f"   Écart-type: {token_counts.std():.2f}")

# Percentiles pour tokens

print(f"\n Percentiles (tokens):")
for p in percentiles:
    val = np.percentile(token_counts, p)
    print(f"   {p}e percentile: {int(val)} tokens")

# RECOMMANDATION DE max_length

percentile_95 = int(np.percentile(token_counts, 95))
percentile_99 = int(np.percentile(token_counts, 99))

print(f"\n RECOMMANDATION DE max_length:")
print(f"   - Pour couvrir 95% des textes: max_length = {percentile_95}")
print(f"   - Pour couvrir 99% des textes: max_length = {percentile_99}")
print(f"   - Choix conservatif (plus sûr): max_length = 256 ou 512")


#  AFFICHER 5+ EXEMPLES


print("\n" + "="*80)
print("5. EXEMPLES DE TEXTES (Au moins 5)")
print("="*80)

print(f"\n Affichage de 10 exemples du dataset:\n")

for idx in range(min(10, len(df))):
    text = df[TEXT_COLUMN].iloc[idx]
    label = df[LABEL_COLUMN].iloc[idx]
    num_tokens = token_counts[idx]
    
    # Limiter le texte à 150 caractères pour l'affichage
    text_display = text[:150] + "..." if len(text) > 150 else text
    
    print(f"Exemple {idx+1}:")
    print(f"  Label: {label}")
    print(f"  Longueur: {len(text)} caractères, {num_tokens} tokens")
    print(f"  Texte: {text_display}")
    print()