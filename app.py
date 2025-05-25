import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰",
    layout="wide"
)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

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

# Charger ou entraîner le modèle
detector = get_model()

if detector is not None:
    # Interface utilisateur pour la détection
    st.header("📝 Analyser un article")
    
    # Formulaire de saisie
    with st.form("news_form"):
        language = st.selectbox("Langue de l'article", ["Français", "Anglais"])
        title = st.text_input("Titre de l'article")
        text = st.text_area("Contenu de l'article")
        submitted = st.form_submit_button("Analyser")
    
    if submitted and title and text:
        with st.spinner("Analyse en cours..."):
            # Traduction si nécessaire
            if language == "Français":
                translator = GoogleTranslator(source='fr', target='en')
                try:
                    title_en = translator.translate(title)
                    text_en = translator.translate(text)
                    st.info("Article traduit en anglais pour l'analyse")
                except Exception as e:
                    st.error(f"Erreur lors de la traduction : {str(e)}")
                    st.stop()
            else:
                title_en = title
                text_en = text
            
            # Prédiction
            result = detector.predict_one(title_en, text_en)
            
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
                st.write(f"Cluster assigné : {result['cluster']}")
                st.write(f"Z-score : {result['z_score']:.2f}")
                st.write("Note : Un Z-score > 2.0 indique une anomalie potentielle")
    
    # Statistiques du dataset d'entraînement
    st.header("📊 Statistiques du dataset d'entraînement")
    df = pd.read_csv('Fake.csv')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"Nombre total d'articles : {len(df)}")
        st.write("\nDistribution des sujets :")
        st.write(df['subject'].value_counts())
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        df['subject'].value_counts().plot(kind='bar')
        plt.title("Distribution des sujets dans le dataset d'entraînement")
        plt.xticks(rotation=45)
        st.pyplot(fig)

else:
    st.warning("⚠️ Le modèle n'est pas chargé. Veuillez exécuter train_model.py d'abord.")
