import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import re
import joblib
import os

# Liste des stop words en anglais et en français
STOP_WORDS = {
    # Anglais
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
    'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and',
    'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by',
    'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
    'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
    # Français
    'je', 'tu', 'il', 'elle', 'nous', 'vous', 'ils', 'elles', 'le', 'la',
    'les', 'un', 'une', 'des', 'ce', 'cette', 'ces', 'mon', 'ton', 'son',
    'ma', 'ta', 'sa', 'mes', 'tes', 'ses', 'notre', 'votre', 'leur', 'nos',
    'vos', 'leurs', 'qui', 'que', 'quoi', 'dont', 'où', 'quand', 'comment',
    'pourquoi', 'quel', 'quelle', 'quels', 'quelles', 'et', 'ou', 'mais',
    'donc', 'car', 'si', 'dans', 'sur', 'sous', 'avec', 'sans', 'pour',
    'par', 'de', 'à', 'vers', 'chez', 'entre', 'pendant', 'après', 'avant',
    'depuis', 'dès', 'durant', 'être', 'avoir', 'faire', 'dire', 'aller',
    'voir', 'venir', 'devoir', 'vouloir', 'pouvoir', 'falloir', 'très',
    'plus', 'moins', 'peu', 'beaucoup', 'trop', 'assez', 'tout', 'tous',
    'toute', 'toutes', 'aucun', 'aucune', 'même', 'aussi', 'alors', 'donc'
}

class FakeNewsDetector:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        # Augmenter le nombre de features et ajouter des n-grams
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=2
        )
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.cluster_stats = None
        self.cluster_fake_ratios = None  # Pour stocker le ratio de fake news par cluster
        
    def preprocess_text(self, texts):
        processed_texts = []
        for text in texts:
            # Convertir en minuscules
            text = str(text).lower()
            # Supprimer la ponctuation et les chiffres
            text = re.sub(r'[^a-z\s]', ' ', text)
            # Diviser en mots
            words = text.split()
            # Supprimer les stop words et les mots courts
            words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
            # Rejoindre les mots
            processed_texts.append(' '.join(words))
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
    try:
        # Charger les données anglaises
        df_fake = pd.read_csv('Fake.csv')
        df_fake['is_fake'] = True
        df_fake['lang'] = 'en'
        
        df_true = pd.read_csv('True.csv')
        df_true['is_fake'] = False
        df_true['lang'] = 'en'
        
        english_data = pd.concat([df_fake, df_true], ignore_index=True)
        print(f"Articles anglais chargés : {len(english_data)}")
    except Exception as e:
        print(f"Erreur lors du chargement des données anglaises : {e}")
        raise Exception("Impossible de charger les données anglaises. Vérifiez que Fake.csv et True.csv sont présents.")
    
    # Initialiser df_fr comme DataFrame vide
    df_fr = pd.DataFrame()
    
    # Essayer de charger les données françaises si le fichier existe
    french_dataset_path = 'french dataset/train.csv'
    if os.path.exists(french_dataset_path):
        try:
            df_fr = pd.read_csv(french_dataset_path, sep=';', encoding='utf-8')
            print("Colonnes du dataset français :", df_fr.columns.tolist())
            
            # Adapter le format des données françaises
            df_fr['is_fake'] = df_fr['fake'].astype(bool)
            df_fr['lang'] = 'fr'
            
            # Renommer les colonnes pour correspondre au format attendu
            df_fr = df_fr.rename(columns={
                'post': 'text',
                'media': 'source'
            })
            
            # Utiliser le début du texte comme titre puisqu'il n'y en a pas
            df_fr['title'] = df_fr['text'].str[:100] + '...'
            
            print(f"Articles français chargés : {len(df_fr)}")
        except Exception as e:
            print(f"Attention : Impossible de charger les données françaises : {e}")
            print("L'application continuera avec les données anglaises uniquement.")
    
    # Combiner les datasets
    df = pd.concat([english_data, df_fr], ignore_index=True)
    
    print(f"\nTotal d'articles : {len(df)}")
    print(f"Articles en anglais : {len(english_data)}")
    print(f"Articles en français : {len(df_fr)}")
    
    if len(df) == 0:
        raise Exception("Aucune donnée n'a pu être chargée")
    
    print("\nInitialisation du détecteur...")
    # Ajuster le nombre de clusters en fonction de la taille du dataset
    n_clusters = min(8, max(3, len(df) // 1000))
    detector = FakeNewsDetector(n_clusters=n_clusters)
    
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
