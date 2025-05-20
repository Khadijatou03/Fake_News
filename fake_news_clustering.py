import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import warnings
warnings.filterwarnings('ignore')

class FakeNewsClusterer:
    def __init__(self, n_clusters=5):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
    def preprocess_text(self, texts):
        # Télécharger les ressources NLTK nécessaires
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        processed_texts = []
        for text in texts:
            # Tokenization et nettoyage
            tokens = word_tokenize(str(text).lower())
            # Suppression des stop words et ponctuation
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
            processed_texts.append(' '.join(tokens))
        return processed_texts

    def prepare_data(self, titles, texts):
        # Combiner titre et texte avec plus de poids pour le titre
        combined_texts = [f"{title} {title} {text}" for title, text in zip(titles, texts)]
        processed_texts = self.preprocess_text(combined_texts)
        X = self.vectorizer.fit_transform(processed_texts)
        X_scaled = self.scaler.fit_transform(X.toarray())
        return X_scaled

    def kmeans_clustering(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X)
        return clusters, kmeans

    def agglomerative_clustering(self, X):
        agg = AgglomerativeClustering(n_clusters=self.n_clusters)
        clusters = agg.fit_predict(X)
        return clusters, agg

    def dbscan_clustering(self, X):
        # Réduction de dimensionnalité avec PCA pour DBSCAN
        pca = PCA(n_components=min(50, X.shape[1]))
        X_pca = pca.fit_transform(X)
        
        # Application de DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X_pca)
        
        return clusters, dbscan

    def evaluate_clustering(self, X, labels):
        silhouette = silhouette_score(X, labels)
        calinski = calinski_harabasz_score(X, labels)
        return silhouette, calinski

    def plot_results(self, results):
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score']
        algorithms = list(results.keys())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Silhouette Score
        scores = [results[algo]['silhouette'] for algo in algorithms]
        ax1.bar(algorithms, scores)
        ax1.set_title('Silhouette Score par algorithme')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45)
        
        # Calinski-Harabasz Score
        scores = [results[algo]['calinski'] for algo in algorithms]
        ax2.bar(algorithms, scores)
        ax2.set_title('Calinski-Harabasz Score par algorithme')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('clustering_comparison.png')
        plt.close()

def main():
    # Charger les données du fichier Fake.csv
    try:
        df = pd.read_csv('Fake.csv')
        titles = df['title'].values
        texts = df['text'].values
        subjects = df['subject'].values
        
        print(f"Nombre total d'articles : {len(df)}")
        print("\nDistribution des sujets :")
        print(pd.Series(subjects).value_counts())
    except Exception as e:
        print(f"Erreur lors du chargement du fichier : {str(e)}")
        return

    # Initialisation du clusterer
    clusterer = FakeNewsClusterer(n_clusters=5)
    
    # Préparation des données en combinant titres et textes
    X = clusterer.prepare_data(titles, texts)
    
    # Application des différents algorithmes
    results = {}
    
    # K-means
    print("Application de K-means...")
    kmeans_labels, _ = clusterer.kmeans_clustering(X)
    silhouette, calinski = clusterer.evaluate_clustering(X, kmeans_labels)
    results['K-means'] = {'silhouette': silhouette, 'calinski': calinski}
    
    # Clustering Agglomératif
    print("Application du Clustering Agglomératif...")
    agg_labels, _ = clusterer.agglomerative_clustering(X)
    silhouette, calinski = clusterer.evaluate_clustering(X, agg_labels)
    results['Agglomerative'] = {'silhouette': silhouette, 'calinski': calinski}
    
    # DBSCAN Clustering
    print("Application du DBSCAN Clustering...")
    dbscan_labels, _ = clusterer.dbscan_clustering(X)
    if len(np.unique(dbscan_labels)) > 1:  # DBSCAN peut retourner un seul cluster
        silhouette, calinski = clusterer.evaluate_clustering(X, dbscan_labels)
        results['DBSCAN'] = {'silhouette': silhouette, 'calinski': calinski}
    
    # Affichage des résultats
    print("\nRésultats de l'évaluation:")
    for algo, scores in results.items():
        print(f"\n{algo}:")
        print(f"Silhouette Score: {scores['silhouette']:.3f}")
        print(f"Calinski-Harabasz Score: {scores['calinski']:.3f}")
    
    # Création du graphique comparatif
    clusterer.plot_results(results)
    print("\nLe graphique comparatif a été sauvegardé dans 'clustering_comparison.png'")

if __name__ == "__main__":
    main()
