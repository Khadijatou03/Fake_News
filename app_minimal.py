import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os

# Configuration de la page - version simplifiée
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰"
)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

try:
    # Charger le modèle
    try:
        detector = FakeNewsDetector.load('fake_news_model.joblib')
        st.success("Modèle chargé avec succès !")
    except:
        st.error("Impossible de charger le modèle. Veuillez exécuter train_model.py d'abord.")
        st.stop()
        
    # Interface utilisateur pour la détection
    st.header("📝 Analyser un article")
    
    # Formulaire de saisie
    with st.form("news_form"):
        title = st.text_input("Titre de l'article")
        text = st.text_area("Contenu de l'article")
        submitted = st.form_submit_button("Analyser")
    
    if submitted and title and text:
        with st.spinner("Analyse en cours..."):
            # Prédiction
            result = detector.predict_one(title, text)
            
            # Affichage des résultats
            st.header("📊 Résultats de l'analyse")
            
            # Affichage du verdict
            if result['is_fake']:
                st.error("⚠️ Cet article est probablement une FAKE NEWS")
            else:
                st.success("✅ Cet article semble fiable")
            
            st.metric("Indice de confiance", f"{result['confidence']*100:.1f}%")
            
            # Détails techniques
            with st.expander("Détails techniques"):
                st.write(f"Cluster assigné : {result['cluster']}")
                st.write(f"Z-score : {result['z_score']:.2f}")
    
    # Informations sur le modèle
    st.header("ℹ️ À propos")
    st.write("Cette application utilise l'apprentissage automatique pour détecter les fake news.")
    st.write("Développé dans le cadre d'un projet de fouille de données.")
    
except Exception as e:
    st.error(f"Erreur : {str(e)}")
    st.info("Si vous rencontrez des problèmes, essayez d'exécuter l'application en local.")
