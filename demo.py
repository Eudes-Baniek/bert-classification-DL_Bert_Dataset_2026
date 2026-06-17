import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from model import berForTextClassification


# CONFIGURATION ET CONSTANTES

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = './checkpoints/best_model.pt'
TOKENIZER_PATH = './checkpoints/tokenizer'
MODEL_NAME = 'bert-base-multilingual-cased'
NUM_CLASSES = 3  
MAX_LENGTH = 128

# Les 3 vraies classes de la tâche Natural Language Inference (NLI)

CLASS_NAMES = {
    0: 'Implication',
    1: 'Neutre',
    2: 'Contradiction'
}


# CHARGEMENT DES COMPOSANTS

def load_model():
    """Charger le meilleur modèle sauvegardé avec la bonne dimension de sortie."""
    model = berForTextClassification(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES  # Initialise correctement avec 3 classes
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_tokenizer():
    """Charger le tokenizer depuis le dossier local."""
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Chargement unique au démarrage du serveur
model = load_model()
tokenizer = load_tokenizer()

print(" Modèle mBERT (3 classes) et tokenizer chargés avec succès !")


# FONCTION DE PRÉDICTION (PAIRE DE TEXTES : PREMISSE + HYPOTHESE)

def predict_nli(premise, hypothesis):
    """
    Prédit la relation logique entre une prémisse et une hypothèse.
    """
    if not premise.strip() or not hypothesis.strip():
        return "Veuillez remplir les deux champs."
        
    # Tokenization conjointe de la prémisse et de l'hypothèse (séparées par [SEP])
    encoding = tokenizer(
        premise,
        hypothesis,
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Inférence sans calcul de gradients
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probas = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    # Créer le dictionnaire classe -> probabilité pour les 3 classes
    predictions = {CLASS_NAMES[i]: float(probas[i]) for i in range(NUM_CLASSES)}
    
    return predictions


# INTERFACE GRAPHIQUE GRADIO (CONFORME À LA FINALITÉ DU SUJET)


examples = [
    ["Un homme joue de la guitare dans la rue.", "Quelqu'un fait de la musique."],    # Implication
    ["Une femme dort sur son canapé.", "La femme s'apprête à courir un marathon."], # Contradiction
    ["Un chien noir court dans la neige.", "Le chien appartient à un jeune garçon."] # Neutre
]

with gr.Blocks(theme=gr.themes.Soft(), title="BERT NLI Classification") as demo:
    
    # En-tête
    gr.Markdown("""
    #  BERT Text Inference Demo (NLI)
    
    Interface de démonstration pour le **Devoir Pratique n°3**. Ce modèle mBERT a été fine-tuné pour analyser la relation sémantique et logique entre deux phrases.
    """)
    
    # Formulaire de saisie et affichage des résultats
    with gr.Row():
        with gr.Column():
            premise_input = gr.Textbox(
                label=" Prémisse (Contexte de référence)",
                placeholder="Entrez la phrase de contexte...",
                lines=3
            )
            hypothesis_input = gr.Textbox(
                label=" Hypothèse (Proposition à tester)",
                placeholder="Entrez l'hypothèse à vérifier...",
                lines=2
            )
            predict_btn = gr.Button(" Analyser la relation logique", variant="primary", size="lg")
        
        with gr.Column():
            output_label = gr.Label(
                label=" Résultats de l'Inférence",
                num_top_classes=NUM_CLASSES
            )
    
    # Liaison du bouton à la fonction de prédiction
    
    predict_btn.click(
        fn=predict_nli,
        inputs=[premise_input, hypothesis_input],
        outputs=output_label
    )
    
    # Zone des exemples pré-remplis (Prend en entrée les deux composants)
    gr.Examples(
        examples=examples,
        inputs=[premise_input, hypothesis_input],
        outputs=output_label,
        fn=predict_nli,
        label=" Exemples de test"
    )
       
    # Pied de page
    gr.Markdown(f"""
    ---
    **Modèle utilisé** : `{MODEL_NAME}` | **Tâche** : Natural Language Inference (NLI) à {NUM_CLASSES} classes
    """)

# Lancement de l'application

if __name__ == "__main__":
    demo.launch(share=False)