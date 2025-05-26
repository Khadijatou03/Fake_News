import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os
from langdetect import detect
from analyze_datasets import get_dataset_stats

# Configuration de la page - version minimale pour déploiement
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰"
)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

# Version simplifiée - pas de fonctions complexes

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

# Version simplifiée pour le déploiement - pas d'onglets ni de statistiques complexes

try:
    # Charger le modèle
    detector = get_model()
    
    if detector is not None:
        st.header("📝 Analyser un article")
        
        # Formulaire de saisie simplifié
        with st.form("news_form"):
            title = st.text_input("Titre de l'article")
            text = st.text_area("Contenu de l'article")
            submitted = st.form_submit_button("Analyser")
        
        if submitted and text:  # Le titre peut être optionnel
            with st.spinner("Analyse en cours..."):
                # Détecter automatiquement la langue
                try:
                    lang = detect(text)
                    if lang == 'fr':
                        detected_language = "Français"
                    else:
                        detected_language = "Anglais"
                except:
                    detected_language = "Anglais"
                
                st.info(f"Langue détectée : {detected_language}")
                
                # Utiliser le modèle de clustering par défaut
                result = detector.predict_one(title if title else text[:50], text)
                
                # Affichage des résultats simplifiés
                st.header("📊 Résultats de l'analyse")
                
                # Affichage du verdict sans colonnes
                if result['is_fake']:
                    st.error("⚠️ Cet article est probablement une FAKE NEWS")
                else:
                    st.success("✅ Cet article semble fiable")
                
                st.metric("Indice de confiance", f"{result['confidence']*100:.1f}%")
                
                # Détails techniques simplifiés
                st.write(f"Langue détectée : {detected_language}")
                st.write(f"Cluster assigné : {result['cluster']}")
                st.write(f"Z-score : {result['z_score']:.2f}")
        
        # Statistiques sur les datasets
        st.markdown("---")
        st.header("📊 Statistiques des datasets")
        
        with st.spinner("Chargement des statistiques..."):
            try:
                stats = get_dataset_stats()
                
                # Affichage des statistiques globales
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total d'articles", stats['total_articles'])
                with col2:
                    st.metric("Articles fake", stats['total_fake'])
                with col3:
                    st.metric("Articles vrais", stats['total_true'])
                
                if stats['total_articles'] > 0:
                    fake_percent = (stats['total_fake'] / stats['total_articles']) * 100
                    true_percent = (stats['total_true'] / stats['total_articles']) * 100
                    
                    st.progress(fake_percent/100)
                    st.caption(f"Répartition : {fake_percent:.1f}% fake news vs {true_percent:.1f}% articles vrais")
                
                # Détails par dataset
                st.subheader("Détails par dataset")
                for dataset in stats['datasets']:
                    with st.expander(f"{dataset['name']} - {dataset['articles']} articles"):
                        st.write(f"Articles fake: {dataset['fake']}")
                        st.write(f"Articles vrais: {dataset['true']}")
                        if dataset['articles'] > 0:
                            fake_pct = (dataset['fake'] / dataset['articles']) * 100
                            true_pct = (dataset['true'] / dataset['articles']) * 100
                            st.progress(fake_pct/100)
                            st.caption(f"Répartition : {fake_pct:.1f}% fake news vs {true_pct:.1f}% articles vrais")
            except Exception as e:
                st.error(f"Erreur lors du chargement des statistiques : {str(e)}")
                st.info("Les statistiques ne sont pas disponibles. Vérifiez que les datasets sont accessibles.")
        
        # Informations sur l'application
        st.markdown("---")
        st.header("ℹ️ À propos")
        st.write("Cette application utilise l'apprentissage automatique pour détecter les fake news.")
        st.write("Le modèle a été entraîné sur des datasets en anglais et en français.")
        st.write("La langue de l'article est détectée automatiquement.")

    else:
        st.warning("⚠️ Le modèle n'est pas chargé. Veuillez exécuter train_model.py d'abord.")

except Exception as e:
    st.error(f"Erreur : {str(e)}")
    st.info("Si vous rencontrez des problèmes, essayez d'exécuter l'application en local.")
