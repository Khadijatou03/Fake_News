import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    CamembertTokenizer, CamembertForSequenceClassification,
    FlaubertTokenizer, FlaubertForSequenceClassification,
    BertTokenizer, BertForSequenceClassification,
    RobertaTokenizer, RobertaForSequenceClassification,
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AdamW, get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Définition des modèles disponibles
AVAILABLE_MODELS = {
    'camembert': {
        'name': 'camembert-base',
        'tokenizer': CamembertTokenizer,
        'model': CamembertForSequenceClassification
    },
    'flaubert': {
        'name': 'flaubert/flaubert_base_cased',
        'tokenizer': FlaubertTokenizer,
        'model': FlaubertForSequenceClassification
    },
    'bert': {
        'name': 'bert-base-multilingual-cased',
        'tokenizer': BertTokenizer,
        'model': BertForSequenceClassification
    },
    'roberta': {
        'name': 'roberta-base',
        'tokenizer': RobertaTokenizer,
        'model': RobertaForSequenceClassification
    },
    'distilbert': {
        'name': 'distilbert-base-multilingual-cased',
        'tokenizer': DistilBertTokenizer,
        'model': DistilBertForSequenceClassification
    }
}

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerFakeNewsDetector:
    def __init__(self, model_name='camembert', num_labels=2):
        """
        Initialise le détecteur de fake news basé sur un modèle transformer.
        
        Args:
            model_name (str): Nom du modèle à utiliser ('camembert', 'flaubert', 'bert', 'roberta', 'distilbert')
            num_labels (int): Nombre de classes (2 pour la détection binaire de fake news)
        """
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Modèle {model_name} non disponible. Choisissez parmi: {', '.join(AVAILABLE_MODELS.keys())}")
        
        self.model_name = model_name
        self.model_info = AVAILABLE_MODELS[model_name]
        self.num_labels = num_labels
        
        # Initialiser le tokenizer
        self.tokenizer = self.model_info['tokenizer'].from_pretrained(self.model_info['name'])
        
        # Initialiser le modèle
        self.model = self.model_info['model'].from_pretrained(
            self.model_info['name'],
            num_labels=num_labels
        )
        
        # Vérifier si CUDA est disponible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"Modèle {model_name} initialisé sur {self.device}")
        
    def train(self, train_texts, train_labels, val_texts=None, val_labels=None, 
              batch_size=8, epochs=3, learning_rate=2e-5):
        """
        Entraîne le modèle sur les données fournies.
        
        Args:
            train_texts (list): Liste des textes d'entraînement
            train_labels (list): Liste des labels d'entraînement (0 pour vrai, 1 pour fake)
            val_texts (list, optional): Liste des textes de validation
            val_labels (list, optional): Liste des labels de validation
            batch_size (int): Taille du batch
            epochs (int): Nombre d'époques
            learning_rate (float): Taux d'apprentissage
        """
        # Si pas de données de validation, en créer à partir des données d'entraînement
        if val_texts is None or val_labels is None:
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                train_texts, train_labels, test_size=0.1, random_state=42
            )
        
        # Créer les datasets
        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = FakeNewsDataset(val_texts, val_labels, self.tokenizer)
        
        # Créer les dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Optimiseur et scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Boucle d'entraînement
        best_accuracy = 0
        
        for epoch in range(epochs):
            logger.info(f"Époque {epoch+1}/{epochs}")
            
            # Entraînement
            self.model.train()
            train_loss = 0
            
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_dataloader)
            logger.info(f"Perte moyenne d'entraînement: {avg_train_loss:.4f}")
            
            # Évaluation
            self.model.eval()
            val_loss = 0
            predictions = []
            true_labels = []
            
            for batch in val_dataloader:
                with torch.no_grad():
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    predictions.extend(preds)
                    true_labels.extend(labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(val_dataloader)
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_labels, predictions, average='binary'
            )
            
            logger.info(f"Perte de validation: {avg_val_loss:.4f}")
            logger.info(f"Précision: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            
            # Sauvegarder le meilleur modèle
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                logger.info(f"Nouvelle meilleure précision: {best_accuracy:.4f}")
        
        logger.info(f"Entraînement terminé avec une précision finale de {best_accuracy:.4f}")
        
    def predict(self, texts):
        """
        Prédit si un texte est une fake news ou non.
        
        Args:
            texts (list): Liste des textes à prédire
            
        Returns:
            list: Liste des prédictions (0 pour vrai, 1 pour fake)
        """
        self.model.eval()
        predictions = []
        
        # Créer un dataset et dataloader
        dummy_labels = [0] * len(texts)  # Labels fictifs
        dataset = FakeNewsDataset(texts, dummy_labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8)
        
        for batch in dataloader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        return predictions
    
    def predict_one(self, text):
        """
        Prédit si un texte est une fake news ou non et retourne les détails.
        
        Args:
            text (str): Texte à prédire
            
        Returns:
            dict: Dictionnaire contenant les détails de la prédiction
        """
        self.model.eval()
        
        # Tokenizer le texte
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            
            prediction = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
        return {
            'is_fake': bool(prediction),
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist()
        }
    
    def save(self, path):
        """
        Sauvegarde le modèle et le tokenizer.
        
        Args:
            path (str): Chemin où sauvegarder le modèle
        """
        # Créer le dossier s'il n'existe pas
        os.makedirs(path, exist_ok=True)
        
        # Sauvegarder le modèle et le tokenizer
        self.model.save_pretrained(os.path.join(path, 'model'))
        self.tokenizer.save_pretrained(os.path.join(path, 'tokenizer'))
        
        # Sauvegarder les métadonnées
        metadata = {
            'model_name': self.model_name,
            'num_labels': self.num_labels
        }
        
        with open(os.path.join(path, 'metadata.joblib'), 'wb') as f:
            joblib.dump(metadata, f)
        
        logger.info(f"Modèle sauvegardé dans {path}")
    
    @classmethod
    def load(cls, path):
        """
        Charge un modèle sauvegardé.
        
        Args:
            path (str): Chemin où le modèle a été sauvegardé
            
        Returns:
            TransformerFakeNewsDetector: Instance du détecteur chargé
        """
        # Charger les métadonnées
        with open(os.path.join(path, 'metadata.joblib'), 'rb') as f:
            metadata = joblib.load(f)
        
        # Créer une instance
        detector = cls(model_name=metadata['model_name'], num_labels=metadata['num_labels'])
        
        # Charger le modèle et le tokenizer
        model_path = os.path.join(path, 'model')
        tokenizer_path = os.path.join(path, 'tokenizer')
        
        detector.model = detector.model_info['model'].from_pretrained(model_path)
        detector.tokenizer = detector.model_info['tokenizer'].from_pretrained(tokenizer_path)
        
        # Déplacer le modèle sur le bon device
        detector.model.to(detector.device)
        
        logger.info(f"Modèle chargé depuis {path}")
        
        return detector

def train_transformer_model(model_name='camembert', epochs=3, batch_size=8, learning_rate=2e-5):
    """
    Entraîne un modèle transformer sur le dataset français et le sauvegarde.
    
    Args:
        model_name (str): Nom du modèle à utiliser
        epochs (int): Nombre d'époques
        batch_size (int): Taille du batch
        learning_rate (float): Taux d'apprentissage
        
    Returns:
        TransformerFakeNewsDetector: Modèle entraîné
    """
    logger.info(f"Entraînement du modèle {model_name}...")
    
    # Charger les données françaises
    try:
        df_fr = pd.read_csv('french dataset/train.csv', sep=';', encoding='utf-8')
        logger.info(f"Dataset français chargé: {len(df_fr)} articles")
    except Exception as e:
        logger.error(f"Erreur lors du chargement du dataset français: {e}")
        raise
    
    # Préparation des données
    texts = df_fr['post'].values
    labels = df_fr['fake'].values
    
    # Split train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, random_state=42
    )
    
    # Initialiser le modèle
    detector = TransformerFakeNewsDetector(model_name=model_name)
    
    # Entraîner le modèle
    detector.train(
        train_texts=train_texts,
        train_labels=train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate
    )
    
    # Sauvegarder le modèle
    model_path = f'transformer_models/{model_name}_fake_news'
    detector.save(model_path)
    
    logger.info(f"Modèle {model_name} entraîné et sauvegardé dans {model_path}")
    
    return detector

if __name__ == "__main__":
    # Entraîner le modèle CamemBERT (le plus adapté pour le français)
    train_transformer_model(model_name='camembert', epochs=3)
