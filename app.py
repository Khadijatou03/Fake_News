import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os
from langdetect import detect
from analyze_datasets import get_dataset_stats

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰",
    layout="wide"
)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

# Fonction pour charger ou entraîner le modèle
@st.cache_resource
def get_model():
    try:
        # Essayer de charger le modèle existant
        if os.path.exists('fake_news_model.joblib'):
            detector = FakeNewsDetector.load('fake_news_model.joblib')
            st.success("Modèle chargé avec succès !")
            return detector
    except Exception as e:
        st.warning(f"Impossible de charger le modèle existant : {str(e)}")
    
    # Si le modèle n'existe pas ou n'a pas pu être chargé, on l'entraîne
    with st.spinner("Entraînement du modèle en cours..."):
        try:
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
                    st.success("Données téléchargées avec succès !")
                except Exception as e:
                    st.error(f"Erreur lors du téléchargement des données : {str(e)}")
                    st.error("Veuillez vérifier que les fichiers Fake.csv et True.csv sont disponibles dans le repository.")
                    return None
            
            # Entraîner le modèle
            train_and_save_model()
            detector = FakeNewsDetector.load('fake_news_model.joblib')
            st.success("Modèle entraîné et chargé avec succès !")
            return detector
            
        except Exception as e:
            st.error(f"Erreur lors de l'entraînement du modèle : {str(e)}")
            return None

# Interface principale
try:
    # Charger le modèle
    detector = get_model()
    
    if detector is not None:
        # Interface de saisie
        st.header("📝 Analyser un article")
        
        with st.form("news_form"):
            title = st.text_input("Titre de l'article")
            text = st.text_area("Contenu de l'article", height=200)
            submitted = st.form_submit_button("Analyser")
        
        if submitted and text:
            with st.spinner("Analyse en cours..."):
                try:
                    # Détecter la langue
                    lang = detect(text)
                    detected_language = "Français" if lang == 'fr' else "Anglais"
                    st.info(f"Langue détectée : {detected_language}")
                    
                    # Analyser l'article
                    result = detector.predict_one(title if title else text[:50], text)
                    
                    # Afficher les résultats
                    st.header("📊 Résultats de l'analyse")
                    
                    # Verdict
                    if result['is_fake']:
                        st.error("⚠️ Cet article est probablement une FAKE NEWS")
                    else:
                        st.success("✅ Cet article semble fiable")
                    
                    # Métriques
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Indice de confiance", f"{result['confidence']*100:.1f}%")
                    with col2:
                        st.metric("Cluster", result['cluster'])
                    with col3:
                        st.metric("Z-score", f"{result['z_score']:.2f}")
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {str(e)}")
        
        # Statistiques des datasets
        st.markdown("---")
        st.header("📊 Statistiques des datasets")
        
        try:
            stats = get_dataset_stats()
            
            # Statistiques globales
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total d'articles", stats['total_articles'])
            with col2:
                st.metric("Articles fake", stats['total_fake'])
            with col3:
                st.metric("Articles vrais", stats['total_true'])
            
            if stats['total_articles'] > 0:
                fake_percent = (stats['total_fake'] / stats['total_articles']) * 100
                st.progress(fake_percent/100)
                st.caption(f"Répartition : {fake_percent:.1f}% fake news vs {100-fake_percent:.1f}% articles vrais")
            
            # Détails par dataset
            if stats['datasets']:
                st.subheader("Détails par dataset")
                for dataset in stats['datasets']:
                    with st.expander(f"{dataset['name']} - {dataset['articles']} articles"):
                        st.write(f"Articles fake: {dataset['fake']}")
                        st.write(f"Articles vrais: {dataset['true']}")
                        if dataset['articles'] > 0:
                            fake_pct = (dataset['fake'] / dataset['articles']) * 100
                            st.progress(fake_pct/100)
                            st.caption(f"Répartition : {fake_pct:.1f}% fake news vs {100-fake_pct:.1f}% articles vrais")
        except Exception as e:
            st.warning("Les statistiques ne sont pas disponibles pour le moment.")
            st.info("Les statistiques seront disponibles une fois que les datasets seront chargés.")
        
        # À propos
        st.markdown("---")
        st.header("ℹ️ À propos")
        st.write("Cette application utilise l'apprentissage automatique pour détecter les fake news.")
        st.write("Le modèle a été entraîné sur des datasets en anglais et en français.")
        st.write("La langue de l'article est détectée automatiquement.")
    
    else:
        st.error("⚠️ Impossible de charger ou d'entraîner le modèle.")
        st.info("Veuillez vérifier que les fichiers de données (Fake.csv et True.csv) sont disponibles.")

except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}")
    st.info("Si le problème persiste, essayez de redémarrer l'application.")
