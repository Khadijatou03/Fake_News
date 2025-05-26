import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="Détecteur de Fake News",
    page_icon="📰"
)

# Titre de l'application
st.title("Détecteur de Fake News 📰")

# Exemples de textes pour l'entraînement
EXAMPLES = {
    'fake': [
        "BREAKING: Scientists discover that drinking coffee makes you immortal!",
        "ALERT: Government confirms aliens living in your backyard!",
        "SHOCKING: New study proves that the Earth is actually flat!",
        "URGENT: Your phone is secretly recording your thoughts!",
        "EXCLUSIVE: Time travel machine invented in garage!",
        "INCREDIBLE: Man grows wings after eating special fruit!",
        "MIRACLE: Fountain of youth discovered in local park!",
        "SCANDAL: Politician caught riding a unicorn to work!",
        "REVELATION: Moon landing was filmed in Hollywood!",
        "BREAKING: Dragons found living in city sewers!"
    ],
    'true': [
        "New study shows benefits of regular exercise on heart health",
        "Local community opens new public library",
        "Weather forecast predicts rain for the weekend",
        "City council approves new bike lanes",
        "Scientists discover new species of butterfly",
        "Local school wins regional science competition",
        "New public transportation schedule announced",
        "Community garden project receives funding",
        "Local restaurant wins culinary award",
        "New park opens in downtown area"
    ]
}

def create_model():
    """Crée et entraîne un modèle simple avec les exemples"""
    # Préparation des données
    texts = EXAMPLES['fake'] + EXAMPLES['true']
    labels = [1] * len(EXAMPLES['fake']) + [0] * len(EXAMPLES['true'])
    
    # Vectorisation
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    
    # Entraînement du modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, labels)
    
    return vectorizer, model

def get_model():
    """Charge ou crée le modèle"""
    try:
        # Essayer de charger le modèle sauvegardé
        model_data = joblib.load('fake_news_model.joblib')
        vectorizer = model_data['vectorizer']
        model = model_data['model']
        st.success("Modèle chargé avec succès !")
    except:
        # Si le modèle n'existe pas, en créer un nouveau
        with st.spinner("Création d'un nouveau modèle..."):
            vectorizer, model = create_model()
            # Sauvegarder le modèle
            model_data = {
                'vectorizer': vectorizer,
                'model': model
            }
            joblib.dump(model_data, 'fake_news_model.joblib')
            st.success("Nouveau modèle créé et sauvegardé !")
    
    return vectorizer, model

try:
    # Charger ou créer le modèle
    vectorizer, model = get_model()
    
    # Interface utilisateur
    st.header("📝 Analyser un article")
    
    # Formulaire de saisie
    with st.form("news_form"):
        title = st.text_input("Titre de l'article")
        text = st.text_area("Contenu de l'article")
        submitted = st.form_submit_button("Analyser")
    
    if submitted and text:
        with st.spinner("Analyse en cours..."):
            # Préparation du texte
            input_text = f"{title} {text}" if title else text
            X = vectorizer.transform([input_text])
            
            # Prédiction
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = probabilities[1] if prediction == 1 else probabilities[0]
            
            # Affichage des résultats
            st.header("📊 Résultats de l'analyse")
            
            if prediction == 1:
                st.error("⚠️ FAKE NEWS DÉTECTÉE")
                st.metric(
                    "Probabilité que ce soit une fake news",
                    f"{confidence*100:.1f}%"
                )
            else:
                st.success("✅ INFORMATION VRAIE")
                st.metric(
                    "Probabilité que ce soit une information vraie",
                    f"{confidence*100:.1f}%"
                )
            
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
    Cette application utilise un modèle de machine learning simple pour détecter les fake news.
    Le modèle a été entraîné sur un ensemble d'exemples de vraies et fausses nouvelles.
    
    ⚠️ **Note importante** : Cette application est un outil d'aide à la décision. 
    Pour une analyse complète, utilisez toujours votre jugement critique et vérifiez les sources.
    """)

except Exception as e:
    st.error(f"Une erreur est survenue : {str(e)}")
    st.info("Si le problème persiste, essayez de redémarrer l'application.")