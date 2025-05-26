import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Configuration de base
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰",
    layout="wide"
)

# Titre
st.title("Détecteur de Fake News 📰")

# Fonction pour charger ou créer le modèle
@st.cache_resource
def get_model():
    model_path = 'fake_news_model.joblib'
    
    # Si le modèle existe déjà, le charger
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            st.success("Modèle chargé avec succès !")
            return model
        except Exception as e:
            st.warning(f"Erreur lors du chargement du modèle : {str(e)}")
    
    # Sinon, créer et entraîner un nouveau modèle
    with st.spinner("Création d'un nouveau modèle..."):
        try:
            # Créer un modèle simple
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Créer quelques exemples de données d'entraînement
            fake_texts = [
                "BREAKING: Scientists discover unicorns in Antarctica!",
                "Aliens make contact with Earth government",
                "New study proves chocolate cures all diseases",
                "Time travel machine invented in garage",
                "World's first flying car goes on sale next week"
            ]
            
            real_texts = [
                "New study shows benefits of regular exercise",
                "Local school wins regional science competition",
                "City council approves new park development",
                "Weather forecast predicts rain for weekend",
                "New restaurant opens in downtown area"
            ]
            
            # Préparer les données
            texts = fake_texts + real_texts
            labels = [1] * len(fake_texts) + [0] * len(real_texts)
            
            # Entraîner le modèle
            X = vectorizer.fit_transform(texts)
            classifier.fit(X, labels)
            
            # Sauvegarder le modèle
            model = {
                'vectorizer': vectorizer,
                'classifier': classifier
            }
            joblib.dump(model, model_path)
            st.success("Nouveau modèle créé et sauvegardé !")
            return model
            
        except Exception as e:
            st.error(f"Erreur lors de la création du modèle : {str(e)}")
            return None

# Interface principale
try:
    # Charger le modèle
    model = get_model()
    
    if model is not None:
        # Interface de saisie
        st.header("📝 Analyser un article")
        
        with st.form("news_form"):
            text = st.text_area("Contenu de l'article", height=200)
            submitted = st.form_submit_button("Analyser")
        
        if submitted and text:
            with st.spinner("Analyse en cours..."):
                try:
                    # Préparer le texte
                    X = model['vectorizer'].transform([text])
                    
                    # Faire la prédiction
                    prediction = model['classifier'].predict(X)[0]
                    probability = model['classifier'].predict_proba(X)[0]
                    
                    # Afficher les résultats
                    st.header("📊 Résultats de l'analyse")
                    
                    if prediction == 1:
                        st.error("⚠️ Cet article est probablement une FAKE NEWS")
                    else:
                        st.success("✅ Cet article semble fiable")
                    
                    # Afficher la confiance
                    confidence = probability[1] if prediction == 1 else probability[0]
                    st.metric("Indice de confiance", f"{confidence*100:.1f}%")
                    
                    # Afficher des conseils
                    st.info("💡 Conseils pour évaluer la fiabilité d'un article :")
                    st.markdown("""
                    - Vérifiez la source de l'article
                    - Recherchez des informations similaires sur d'autres sites fiables
                    - Méfiez-vous des titres sensationnels ou exagérés
                    - Vérifiez la date de publication
                    - Examinez les citations et les sources citées
                    """)
                    
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse : {str(e)}")
        
        # À propos
        st.markdown("---")
        st.header("ℹ️ À propos")
        st.write("Cette application utilise un modèle simple de machine learning pour détecter les fake news.")
        st.write("Le modèle a été entraîné sur un petit ensemble de données d'exemple.")
        st.write("⚠️ Note : Cette version est une démonstration simplifiée. Pour une analyse plus précise, utilisez votre jugement critique et vérifiez les sources.")
    
    else:
        st.error("⚠️ Impossible de charger ou de créer le modèle.")
        st.info("Veuillez réessayer ou contacter l'administrateur.")

except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}")
    st.info("Si le problème persiste, essayez de redémarrer l'application.")
