# Détecteur de Fake News

Un système de détection de fake news utilisant l'apprentissage non supervisé (clustering) pour identifier les articles susceptibles d'être de fausses nouvelles.

## Fonctionnalités

- Détection en temps réel des fake news
- Interface utilisateur web avec Streamlit
- Apprentissage non supervisé avec K-means
- Analyse basée sur le texte et le titre des articles
- Score de confiance pour chaque prédiction

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/Rimka33/detection-de-fake-news.git
cd detection-de-fake-news
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

3. Téléchargez les datasets et placez-les dans le répertoire du projet :
- `Fake.csv` : Dataset d'articles fake news
- `True.csv` : Dataset d'articles véridiques

## Utilisation

1. Entraînez le modèle :
```bash
python train_model.py
```

2. Lancez l'application web :
```bash
python -m streamlit run app.py
```

3. Ouvrez votre navigateur à l'adresse indiquée (généralement http://localhost:8502)

## Comment ça marche

1. **Prétraitement des données** :
   - Combinaison du titre et du texte
   - Vectorisation TF-IDF avec n-grams
   - Normalisation des features

2. **Clustering** :
   - Utilisation de K-means pour créer des clusters d'articles
   - Analyse des ratios de fake news par cluster
   - Calcul des statistiques de distance pour chaque cluster

3. **Prédiction** :
   - Attribution d'un nouvel article à un cluster
   - Évaluation basée sur le ratio de fake news du cluster
   - Analyse de la distance par rapport au centroïde
   - Calcul d'un score de confiance

## Performance

- Précision sur l'ensemble du dataset : ~77%
- Capacité à détecter des patterns complexes de fake news
- Score de confiance pour évaluer la fiabilité des prédictions

## Structure du projet

```
├── app.py                 # Application Streamlit
├── train_model.py         # Script d'entraînement
├── fake_news_clustering.py # Classes et fonctions principales
├── requirements.txt       # Dépendances Python
├── Fake.csv              # Dataset d'articles fake
├── True.csv              # Dataset d'articles véridiques
└── README.md             # Documentation
```

## Contribution

Les contributions sont les bienvenues ! N'hésitez pas à :
1. Fork le projet
2. Créer une branche pour votre fonctionnalité
3. Commiter vos changements
4. Pousser vers la branche
5. Ouvrir une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
