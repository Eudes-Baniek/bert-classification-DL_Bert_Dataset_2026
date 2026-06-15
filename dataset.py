import torch
from torch.utils.data import Dataset


class TextClassificationDataset(Dataset):
    """
    Dataset PyTorch pour la classification de texte avec BERT.
    """

    def __init__(
        self,
        premises,
        hypotheses,
        labels,
        tokenizer,
        max_length
    ):

        self.premises = premises
        self.hypotheses = hypotheses
        self.labels = labels

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        premise = str(self.premises[idx])

        hypothesis = str(self.hypotheses[idx])

        label = self.labels[idx]

        encoding = self.tokenizer(

            premise,

            hypothesis,

            add_special_tokens=True,

            max_length=self.max_length,

            padding="max_length",

            truncation=True,

            return_attention_mask=True,

            return_tensors="pt"
        )

        return {

            "input_ids":
                encoding["input_ids"].flatten(),

            "attention_mask":
                encoding["attention_mask"].flatten(),

            "labels":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }