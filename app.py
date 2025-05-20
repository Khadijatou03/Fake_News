import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰",
    layout="wide"
)

# Titre de l'application
st.title("🔍 Détecteur de Fake News")

# Chargement du modèle
@st.cache_resource
def load_model():
    try:
        return FakeNewsDetector.load('fake_news_model.joblib')
    except:
        st.error("Le modèle n'est pas encore entraîné. Veuillez exécuter train_model.py d'abord.")
        return None

detector = load_model()

if detector is not None:
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
