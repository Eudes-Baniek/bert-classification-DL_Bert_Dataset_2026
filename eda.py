import pandas as pd
import numpy as np
from transformers import AutoTokenizer

# Chargement du dataset

df = pd.read_csv("data/train.csv")

print("=" * 50)
print("Informations générales")
print("=" * 50)

print(f"Nombre d'exemples : {len(df)}")
print(f"Nombre de classes : {df['label'].nunique()}")

print("\nDistribution des classes :")
print(df["label"].value_counts())

print("\nLangues présentes :")
print(df["language"].value_counts())

# Affichage d'exemples

print("\nQuelques exemples :")

for i in range(5):
    print("-" * 50)
    print("Premise :", df.iloc[i]["premise"])
    print("Hypothesis :", df.iloc[i]["hypothesis"])
    print("Label :", df.iloc[i]["label"])

# Analyse des longueurs

tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)

lengths = []

for _, row in df.iterrows():

    encoding = tokenizer.encode(
        row["premise"],
        row["hypothesis"],
        add_special_tokens=True
    )

    lengths.append(len(encoding))


print("\nAnalyse des longueurs")

print(f"Minimum : {np.min(lengths)}")
print(f"Maximum : {np.max(lengths)}")
print(f"Moyenne : {np.mean(lengths):.2f}")

print(f"90e percentile : {np.percentile(lengths,90):.0f}")
print(f"95e percentile : {np.percentile(lengths,95):.0f}")
print(f"99e percentile : {np.percentile(lengths,99):.0f}")

recommended_max_length = int(np.percentile(lengths,95))

print("\nmax_length recommandé :", recommended_max_length)