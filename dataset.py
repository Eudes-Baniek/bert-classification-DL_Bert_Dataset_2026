import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextClassificationDataset(Dataset):

    def __init__(self, premises, hypotheses, labels, tokenizer, max_length=128):
        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):


        return len(self.labels)
    
    def __getitem__(self, idx):
  
        # Récupère un exemple à l'index idx et le prépare pour BERT.
        
        # Le tokenizer BERT retourne 3 choses:
        # - input_ids: Les IDs numériques des tokens
        # - attention_mask: Indique quels tokens sont réels (1) vs padding (0)
        # - token_type_ids: Pour les paires de phrases [CLS] texte [SEP] (ici 0 partout)
   
        
        # Récupérer le texte et le label
        
        premise = str(self.premises[idx])
        hypothesis = str(self.hypotheses[idx])
        label = self.labels[idx]
        
        #  TOKENIZATION AVEC BERT
        # - padding='max_length' : ajouter du padding pour atteindre max_length
        # - truncation=True : couper si le texte est plus long que max_length
        # - return_tensors='pt' : retourner des tensors PyTorch (au lieu de listes)
        
        encoding = self.tokenizer(
            premise,
            hypothesis,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Les tensors retournés ont shape (1, max_length), on squeeze pour (max_length,)
        return {
            'input_ids': encoding['input_ids'].flatten(),      
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
