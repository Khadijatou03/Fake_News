import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os
from langdetect import detect
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Vérifier si le module transformer_model est disponible
try:
    from transformer_model import TransformerFakeNewsDetector
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰",
    layout="wide",
    menu_items={
        'About': "Détecteur de Fake News - Projet de Fouille de Données"
    }
)

# Configuration de matplotlib pour éviter les problèmes DOM
st.set_option('deprecation.showPyplotGlobalUse', False)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

# Fonction pour détecter la langue du texte
def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'fr':
            return "Français", "fr"
        else:
            return "Anglais", "en"  # Par défaut, on considère que c'est de l'anglais
    except:
        return "Anglais", "en"  # En cas d'erreur, on défaut sur l'anglais
        
# Fonction pour charger les datasets et calculer les statistiques
def load_datasets():
    stats = {}
    
    # Charger les données anglaises
    try:
        df_fake_en = pd.read_csv('Fake.csv')
        df_true_en = pd.read_csv('True.csv')
        english_data = pd.concat([df_fake_en, df_true_en])
        english_data['is_fake'] = [True] * len(df_fake_en) + [False] * len(df_true_en)
        
        stats['en'] = {
            'total': len(english_data),
            'fake': len(df_fake_en),
            'true': len(df_true_en),
            'loaded': True
        }
        
        if 'subject' in english_data.columns:
            stats['en']['subjects'] = english_data['subject'].value_counts().to_dict()
    except Exception as e:
        stats['en'] = {'loaded': False, 'error': str(e)}
    
    # Charger les données françaises
    try:
        df_fr = pd.read_csv('french dataset/train.csv', sep=';', encoding='utf-8')
        
        stats['fr'] = {
            'total': len(df_fr),
            'fake': len(df_fr[df_fr['fake'] == 1]),
            'true': len(df_fr[df_fr['fake'] == 0]),
            'loaded': True
        }
        
        # Statistiques supplémentaires pour le dataset français
        if 'media' in df_fr.columns:
            stats['fr']['sources'] = df_fr['media'].value_counts().to_dict()
    except Exception as e:
        stats['fr'] = {'loaded': False, 'error': str(e)}
    
    return stats

# Fonction pour charger ou entraîner le modèle
def get_model():
    try:
        detector = FakeNewsDetector.load('fake_news_model.joblib')
        st.success("Modèle chargé avec succès !")
    except:
        with st.spinner("Première utilisation : entraînement du modèle en cours..."):
            # Vérifier si les fichiers de données existent
            if not os.path.exists('Fake.csv') or not os.path.exists('True.csv'):
                # Charger les données depuis le repository
                fake_url = "https://raw.githubusercontent.com/Rimka33/detection-de-fake-news/main/Fake.csv"
                true_url = "https://raw.githubusercontent.com/Rimka33/detection-de-fake-news/main/True.csv"
                
                try:
                    df_fake = pd.read_csv(fake_url)
                    df_true = pd.read_csv(true_url)
                    
                    # Sauvegarder localement
                    df_fake.to_csv('Fake.csv', index=False)
                    df_true.to_csv('True.csv', index=False)
                except Exception as e:
                    st.error(f"Erreur lors du chargement des données : {str(e)}")
                    st.stop()
            
            # Entraîner le modèle
            train_and_save_model()
            detector = FakeNewsDetector.load('fake_news_model.joblib')
            st.success("Modèle entraîné et chargé avec succès !")
    
    return detector

# Charger les statistiques des datasets
dataset_stats = load_datasets()

# Interface de l'application avec onglets
tab1, tab2, tab3 = st.tabs(["📝 Analyser un article", "📊 Statistiques des datasets", "ℹ️ À propos"])

try:
    # Charger le modèle
    detector = get_model()
    
    # Vérifier si un modèle transformer est disponible
    transformer_detector = None
    if TRANSFORMER_AVAILABLE and os.path.exists('transformer_models/camembert_fake_news/metadata.joblib'):
        try:
            transformer_detector = TransformerFakeNewsDetector.load('transformer_models/camembert_fake_news')
            st.sidebar.success("Modèle transformer CamemBERT chargé !")
        except Exception as e:
            st.sidebar.warning(f"Impossible de charger le modèle transformer: {str(e)}")
    
    if detector is not None:
        # Onglet 1: Analyser un article
        with tab1:
            st.header("📝 Analyser un article")
            
            # Sélection du modèle à utiliser
            model_type = st.radio(
                "Modèle à utiliser",
                ["Clustering (K-means)"] + (["Transformer (CamemBERT)"] if transformer_detector else []),
                horizontal=True
            )
            
            # Formulaire de saisie
            with st.form("news_form"):
                title = st.text_input("Titre de l'article")
                text = st.text_area("Contenu de l'article")
                submitted = st.form_submit_button("Analyser")
            
            if submitted and text:  # Le titre peut être optionnel
                with st.spinner("Analyse en cours..."):
                    # Détecter automatiquement la langue
                    detected_language, lang_code = detect_language(text)
                    st.info(f"Langue détectée : {detected_language}")
                    
                    # Prédiction selon le modèle choisi
                    if "Transformer" in model_type and transformer_detector:
                        result = transformer_detector.predict_one(text if not title else f"{title}. {text}")
                    else:
                        # Utiliser le modèle de clustering par défaut
                        result = detector.predict_one(title if title else text[:50], text)
                    
                    # Affichage des résultats
                    st.header("📊 Résultats de l'analyse")
                    
                    # Affichage du verdict
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if result['is_fake']:
                            st.error("⚠️ Cet article est probablement une FAKE NEWS")
                        else:
                            st.success("✅ Cet article semble fiable")
                    
                    with col2:
                        st.metric("Indice de confiance", f"{result['confidence']*100:.1f}%")
                    
                    # Détails techniques
                    with st.expander("Détails techniques"):
                        st.write(f"Langue détectée : {detected_language}")
                        if "Transformer" not in model_type:
                            st.write(f"Cluster assigné : {result['cluster']}")
                            st.write(f"Z-score : {result['z_score']:.2f}")
                            st.write("Note : Un Z-score > 2.0 indique une anomalie potentielle")
        
        # Onglet 2: Statistiques des datasets
        with tab2:
            st.header("📊 Statistiques des datasets d'entraînement")
            
            # Affichage des statistiques dans deux colonnes
            col1, col2 = st.columns(2)
            
            # Dataset Anglais
            with col1:
                st.subheader("Dataset Anglais")
                if dataset_stats['en']['loaded']:
                    # Métriques de base
                    st.metric("Nombre total d'articles", dataset_stats['en']['total'])
                    st.metric("Articles vrais", dataset_stats['en']['true'])
                    st.metric("Articles faux", dataset_stats['en']['fake'])
                    
                    # Distribution des sujets si disponible
                    if 'subjects' in dataset_stats['en']:
                        st.write("\nDistribution des sujets :")
                        subjects_df = pd.DataFrame(
                            list(dataset_stats['en']['subjects'].items()), 
                            columns=['Sujet', 'Nombre']
                        ).sort_values('Nombre', ascending=False)
                        
                        st.dataframe(subjects_df)
                        
                        # Afficher le top 5 des sujets
                        st.write("Top 5 des sujets les plus fréquents :")
                        st.table(subjects_df.head(5))
                else:
                    st.warning("⚠️ Impossible de charger les données anglaises")
                    if 'error' in dataset_stats['en']:
                        st.error(dataset_stats['en']['error'])
            
            # Dataset Français
            with col2:
                st.subheader("Dataset Français")
                if dataset_stats['fr']['loaded']:
                    # Métriques de base
                    st.metric("Nombre total d'articles", dataset_stats['fr']['total'])
                    st.metric("Articles vrais", dataset_stats['fr']['true'])
                    st.metric("Articles faux", dataset_stats['fr']['fake'])
                    
                    # Affichage des sources si disponible
                    if 'sources' in dataset_stats['fr']:
                        st.write("\nDistribution des sources :")
                        sources_df = pd.DataFrame(
                            list(dataset_stats['fr']['sources'].items()), 
                            columns=['Source', 'Nombre']
                        ).sort_values('Nombre', ascending=False)
                        
                        st.dataframe(sources_df)
                        
                        # Afficher le top 5 des sources
                        st.write("Top 5 des sources les plus fréquentes :")
                        st.table(sources_df.head(5))
                else:
                    st.warning("⚠️ Impossible de charger les données françaises")
                    if 'error' in dataset_stats['fr']:
                        st.error(dataset_stats['fr']['error'])
        
        # Onglet 3: À propos
        with tab3:
            st.header("ℹ️ À propos de cette application")
            st.write("Cette application utilise l'apprentissage automatique pour détecter les fake news.")
            st.write("Le modèle a été entraîné sur des datasets en anglais et en français.")
            st.write("La langue de l'article est détectée automatiquement.")
            
            st.subheader("Modèles disponibles")
            st.markdown("""
            1. **Modèle de clustering (K-means)**
               - Utilise TF-IDF pour extraire les caractéristiques du texte
               - Regroupe les articles en clusters et détecte les anomalies
               - Rapide et efficace pour le déploiement
            
            2. **Modèles Transformer**
               - CamemBERT: Spécialement entraîné pour le français
               - FlauBERT: Alternative française à BERT
               - BERT: Modèle multilingue pré-entraîné
               - RoBERTa: Version optimisée de BERT
               - DistilBERT: Version légère de BERT pour le déploiement
            """)
            
            st.subheader("Datasets utilisés")
            st.markdown("""
            - **Dataset anglais**: Articles de différentes sources, classés en vrais et faux
            - **Dataset français**: Articles adaptés au contexte francophone
            """)
            
            with st.expander("Informations techniques détaillées"):
                st.write("Le modèle de clustering utilise une approche non supervisée avec K-means.")
                st.write("Les caractéristiques sont extraites à l'aide de TF-IDF sur le texte prétraité.")
                st.write("La détection se base sur la distance aux centroïdes et les ratios de fake news par cluster.")
                
                if TRANSFORMER_AVAILABLE:
                    st.write("\n**Modèles Transformer:**")
                    st.write("Ces modèles utilisent l'architecture Transformer pour comprendre le contexte et les nuances du langage.")
                    st.write("Ils sont fine-tunés sur notre dataset de fake news pour s'adapter à cette tâche spécifique.")
                    st.write("Les modèles français comme CamemBERT et FlauBERT sont particulièrement adaptés pour les textes en français.")

    else:
        st.warning("⚠️ Le modèle n'est pas chargé. Veuillez exécuter train_model.py d'abord.")

except Exception as e:
    st.error(f"Erreur : {str(e)}")
    st.info("Si vous rencontrez des problèmes, essayez d'exécuter l'application en local.")
