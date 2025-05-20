import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Télécharger les ressources NLTK nécessaires
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FakeNewsDetector:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        # Augmenter le nombre de features et ajouter des n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_stats = None
        self.cluster_fake_ratios = None  # Pour stocker le ratio de fake news par cluster
        
    def preprocess_text(self, texts):
        processed_texts = []
        stop_words = set(stopwords.words('english'))
        for text in texts:
            tokens = word_tokenize(str(text).lower())
            tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
            processed_texts.append(' '.join(tokens))
        return processed_texts
    
    def prepare_data(self, titles, texts):
        combined_texts = [f"{title} {title} {text}" for title, text in zip(titles, texts)]
        processed_texts = self.preprocess_text(combined_texts)
        X = self.vectorizer.fit_transform(processed_texts)
        X_scaled = self.scaler.fit_transform(X.toarray())
        return X_scaled
    
    def fit(self, X, is_fake_array):
        self.kmeans.fit(X)
        
        # Calculer les statistiques des clusters
        cluster_distances = []
        cluster_fake_ratios = []
        
        for i in range(self.n_clusters):
            cluster_mask = self.kmeans.labels_ == i
            cluster_points = X[cluster_mask]
            cluster_fake_status = is_fake_array[cluster_mask]
            
            centroid = self.kmeans.cluster_centers_[i]
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            
            # Statistiques de distance
            cluster_distances.append({
                'mean': np.mean(distances),
                'std': np.std(distances),
                'max': np.max(distances)
            })
            
            # Ratio de fake news dans le cluster
            fake_ratio = np.mean(cluster_fake_status)
            cluster_fake_ratios.append(fake_ratio)
        
        self.cluster_stats = cluster_distances
        self.cluster_fake_ratios = cluster_fake_ratios
    
    def predict_one(self, title, text):
        # Préparation du texte
        combined_text = [f"{title} {title} {text}"]
        processed_text = self.preprocess_text(combined_text)
        X = self.vectorizer.transform(processed_text)
        X_scaled = self.scaler.transform(X.toarray())
        
        # Prédiction du cluster
        cluster = self.kmeans.predict(X_scaled)[0]
        
        # Calcul de la distance au centroïde
        centroid = self.kmeans.cluster_centers_[cluster]
        distance = np.linalg.norm(X_scaled - centroid)
        
        # Comparaison avec les statistiques du cluster
        cluster_stats = self.cluster_stats[cluster]
        z_score = (distance - cluster_stats['mean']) / cluster_stats['std']
        
        # Utiliser le ratio de fake news du cluster et la distance
        cluster_fake_ratio = self.cluster_fake_ratios[cluster]
        
        # Combiner les signaux pour la détection
        # 1. Ratio de fake news dans le cluster
        # 2. Distance anormale (z-score)
        base_score = cluster_fake_ratio
        distance_penalty = 1 / (1 + np.exp(-z_score + 2))  # Sigmoid centrée à 2
        
        # Score final
        fake_score = (base_score + distance_penalty) / 2
        
        # Décision et confiance
        is_fake = fake_score > 0.5
        confidence = abs(fake_score - 0.5) * 2  # Normaliser entre 0 et 1
        
        return {
            'cluster': cluster,
            'is_fake': is_fake,
            'confidence': confidence,
            'z_score': z_score,
            'fake_score': fake_score,
            'cluster_fake_ratio': cluster_fake_ratio
        }
    
    def save(self, filename):
        model_data = {
            'vectorizer': self.vectorizer,
            'scaler': self.scaler,
            'kmeans': self.kmeans,
            'cluster_stats': self.cluster_stats,
            'cluster_fake_ratios': self.cluster_fake_ratios
        }
        joblib.dump(model_data, filename)
    
    @classmethod
    def load(cls, filename):
        model_data = joblib.load(filename)
        detector = cls()
        detector.vectorizer = model_data['vectorizer']
        detector.scaler = model_data['scaler']
        detector.kmeans = model_data['kmeans']
        detector.cluster_stats = model_data['cluster_stats']
        detector.cluster_fake_ratios = model_data['cluster_fake_ratios']
        return detector

def train_and_save_model():
    print("Chargement des données...")
    # Charger les fake news
    df_fake = pd.read_csv('Fake.csv')
    df_fake['is_fake'] = True
    
    # Charger les vraies news
    df_true = pd.read_csv('True.csv')
    df_true['is_fake'] = False
    
    # Combiner les datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)
    print(f"Total d'articles : {len(df)}")
    print(f"Articles fake : {len(df_fake)}")
    print(f"Articles vrais : {len(df_true)}")
    
    print("\nInitialisation du détecteur...")
    # Augmenter le nombre de clusters pour mieux capturer la variété
    detector = FakeNewsDetector(n_clusters=8)
    
    print("Préparation des données...")
    X = detector.prepare_data(df['title'].values, df['text'].values)
    
    print("Entraînement du modèle...")
    detector.fit(X, df['is_fake'].values)
    
    # Évaluer la qualité de la séparation
    predictions = []
    for i in range(len(df)):
        title = df['title'].iloc[i]
        text = df['text'].iloc[i]
        pred = detector.predict_one(title, text)
        predictions.append(pred['is_fake'])
    
    # Calculer la correspondance entre les prédictions et la réalité
    accuracy = sum(pred == true for pred, true in zip(predictions, df['is_fake'])) / len(df)
    print(f"\nPrécision sur l'ensemble du dataset : {accuracy*100:.2f}%")
    
    print("\nSauvegarde du modèle...")
    detector.save('fake_news_model.joblib')
    print("Modèle sauvegardé avec succès!")

if __name__ == "__main__":
    train_and_save_model()
