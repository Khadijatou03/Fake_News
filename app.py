import streamlit as st
import pandas as pd
import numpy as np
from train_model import FakeNewsDetector, train_and_save_model
import joblib
import os
from analyze_datasets import get_dataset_stats

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰"
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

try:
    # Charger le modèle
    detector = get_model()
    
    if detector is not None:
        st.header("📝 Analyser un article")
        
        # Formulaire de saisie
        with st.form("news_form"):
            title = st.text_input("Titre de l'article")
            text = st.text_area("Contenu de l'article")
            submitted = st.form_submit_button("Analyser")
        
        if submitted and text:
            with st.spinner("Analyse en cours..."):
                # Utiliser le modèle de clustering
                result = detector.predict_one(title if title else text[:50], text)
                
                # Affichage des résultats
                st.header("📊 Résultats de l'analyse")
                
                # Affichage du verdict
                if result['is_fake']:
                    st.error("⚠️ FAKE NEWS DÉTECTÉE")
                    st.metric(
                        "Probabilité que ce soit une fake news",
                        f"{result['confidence']*100:.1f}%"
                    )
                else:
                    st.success("✅ INFORMATION VRAIE")
                    st.metric(
                        "Probabilité que ce soit une information vraie",
                        f"{(1-result['confidence'])*100:.1f}%"
                    )
                
                # Détails techniques
                st.write(f"Cluster assigné : {result['cluster']}")
                st.write(f"Z-score : {result['z_score']:.2f}")
                
                # Conseils
                st.info("💡 Conseils pour évaluer la fiabilité d'un article :")
                st.markdown("""
                - **Source** : Vérifiez la source de l'article et sa réputation
                - **Vérification** : Recherchez des informations similaires sur d'autres sites fiables
                - **Sensationalisme** : Méfiez-vous des titres exagérés ou trop sensationnels
                - **Date** : Vérifiez la date de publication
                - **Sources citées** : Examinez les citations et les sources mentionnées
                - **Émotions** : Méfiez-vous des articles qui cherchent à provoquer des émotions fortes
                - **Vérification croisée** : Utilisez des sites de fact-checking reconnus
                """)
        
        # À propos
        st.markdown("---")
        st.header("ℹ️ À propos")
        st.write("""
        Cette application utilise un modèle de machine learning pour détecter les fake news.
        Le modèle a été entraîné sur un ensemble de données d'exemple contenant des vraies et fausses nouvelles.
        
        ⚠️ **Note importante** : Cette application est un outil d'aide à la décision. 
        Pour une analyse complète, utilisez toujours votre jugement critique et vérifiez les sources.
        """)

except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}")
    st.info("Si le problème persiste, essayez de redémarrer l'application.")