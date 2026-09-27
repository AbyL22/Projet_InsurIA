import streamlit as st
import pandas as pd
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="InsurIA - Plateforme d'Assurance Intelligente",
    page_icon="🛡️",
    layout="wide"
)


# Chargement des modèles
@st.cache_resource
def load_assets():
    pricing_model = joblib.load("models/pricing_model.pkl")
    nlp_vectorizer = joblib.load("models/nlp_vectorizer.pkl")
    nlp_model = joblib.load("models/nlp_model.pkl")
    return pricing_model, nlp_vectorizer, nlp_model


try:
    pricing_model, nlp_vectorizer, nlp_model = load_assets()
except Exception as e:
    st.error("Erreur lors du chargement des modèles. Assurez-vous d'avoir exécuté les scripts d'entraînement.")
    st.stop()

st.title("🛡️ InsurIA - Estimation & Traitement des Sinistres")
st.markdown("---")

# Navigation par onglets
tab1, tab2 = st.tabs(["💰 Tarification de Prime", "📑 Analyse de Sinistre (NLP)"])

# O N G L E T 1 : TARIFICATION TABULAIRE
with tab1:
    st.header("Simulation de Prime d'Assurance Automobile")

    col1, col2 = st.columns(2)

    with col1:
        age_conducteur = st.slider("Âge du conducteur", 18, 85, 30)
        anciennete_permis = st.slider("Ancienneté du permis (années)", 0, 50, 5)
        bonus_malus = st.slider("Coefficient Bonus/Malus", 0.50, 3.50, 1.00, step=0.05)

    with col2:
        valeur_vehicule = st.number_input("Valeur estimée du véhicule (€)", min_value=1000, max_value=150000,
                                          value=15000)
        puissance_fiscale = st.number_input("Puissance fiscale (CV)", min_value=3, max_value=30, value=6)
        region = st.selectbox("Région de résidence",
                              ["Île-de-France", "PACA", "Auvergne-Rhône-Alpes", "Nouvelle-Aquitaine", "Autre"])

    if st.button("Calculer la Prime Estimée"):
        input_data = pd.DataFrame([{
            'age_conducteur': age_conducteur,
            'anciennete_permis': anciennete_permis,
            'bonus_malus': bonus_malus,
            'valeur_vehicule': valeur_vehicule,
            'puissance_fiscale': puissance_fiscale,
            'region': region
        }])

        prediction = pricing_model.predict(input_data)[0]
        st.success(f"💶 **Prime annuelle estimée : {prediction:.2f} €**")

# O N G L E T 2 : CLASSIFICATION NLP DES SINISTRES
with tab2:
    st.header("Évaluation de la Gravité d'un Sinistre par NLP")

    description = st.text_area(
        "Description textuelle du sinistre par l'assuré :",
        height=120,
        placeholder="Exemple : Véhicule stationné percuté au niveau du pare-chocs arrière par un tiers en stationnement."
    )

    if st.button("Analyser la Gravité"):
        if description.strip():
            desc_vec = nlp_vectorizer.transform([description])
            gravite = nlp_model.predict(desc_vec)[0]

            if gravite == "Faible":
                st.success(f"🟢 **Gravité prédite : {gravite}**")
            elif gravite == "Moyenne":
                st.warning(f"🟠 **Gravité prédite : {gravite}**")
            else:
                st.error(f"🔴 **Gravité prédite : {gravite}**")
        else:
            st.warning("Veuillez saisir une description du sinistre.")