from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class berForTextClassification(nn.Module):
    
    def __init__(self, model_name='bert-base-multilingual-cased', num_classes=2, dropout=0.1):
        
        super().__init__()
        
        #Chargement pré-entrainé des poids du cerveau du model
        
        self.bert=AutoModel.from_pretrained(model_name)
        
        self.classifier=nn.Linear(self.bert.config.hidden_size, num_classes)
        
        
    def forward(self, input_ids, attention_mask):
        
        outputs=self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        cls_output=outputs.last_hidden_state[:, 0, :]
        
        logits=self.classifier(cls_output)
        
        return logits

    def get_model(model_name, num_classes):
    
        #Fonction utilitaire pour instancier facilement le modèle 
    
        return berForTextClassification(model_name=model_name, num_classes=num_classes)