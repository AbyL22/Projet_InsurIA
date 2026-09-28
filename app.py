import streamlit as st
import pandas as pd
import joblib
import os

# Configuration de la page
st.set_page_config(
    page_title="InsurIA - Plateforme d'Assurance Intelligente (Sénégal)",
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

st.title("🛡️ InsurIA - Estimation & Traitement des Sinistres (Sénégal)")
st.markdown("---")

# Navigation par onglets
tab1, tab2 = st.tabs(["💰 Tarification de Prime", "📑 Analyse de Sinistre (NLP)"])

# ---------------------------------------------------------
# O N G L E T 1 : TARIFICATION TABULAIRE (SÉNÉGAL)
# ---------------------------------------------------------
with tab1:
    st.header("Simulation de Prime d'Assurance Automobile")

    col1, col2 = st.columns(2)

    with col1:
        age_conducteur = st.slider("Âge du conducteur", 18, 85, 30)
        experience_permis = st.slider("Ancienneté du permis (années)", 0, 50, 5)
        puissance_vehicule = st.number_input("Puissance fiscale (CV)", min_value=3, max_value=30, value=6)

    with col2:
        ville = st.selectbox("Ville / Zone de résidence", [
            "Dakar - Plateau / Médina",
            "Dakar - Keur Massar / Pikine",
            "Dakar - VDN / Almadies",
            "Thiès",
            "Mbour / Saly",
            "Saint-Louis",
            "Touba / Mbacké",
            "Rufisque",
            "Ziguinchor",
            "Kaolack"
        ])

        zones_dict = {
            "Dakar - Plateau / Médina": "Élevé",
            "Dakar - Keur Massar / Pikine": "Très Élevé",
            "Dakar - VDN / Almadies": "Élevé",
            "Thiès": "Moyen",
            "Mbour / Saly": "Moyen",
            "Saint-Louis": "Faible",
            "Touba / Mbacké": "Moyen",
            "Rufisque": "Élevé",
            "Ziguinchor": "Faible",
            "Kaolack": "Moyen"
        }
        zone_risque = zones_dict[ville]

        type_vehicule = st.selectbox("Type de véhicule", [
            "Particulier (Berline / SUV)",
            "Taxi Jaune-Noir",
            "Transport / Clandos",
            "Moto / Jakarta",
            "Camion / Utilitaire"
        ])

    if st.button("Calculer la Prime Estimée"):
        input_data = pd.DataFrame([{
            'age_conducteur': age_conducteur,
            'experience_permis': experience_permis,
            'puissance_vehicule': puissance_vehicule,
            'ville': ville,
            'zone_risque': zone_risque,
            'type_vehicule': type_vehicule
        }])

        prediction = pricing_model.predict(input_data)[0]
        st.success(f"💶 **Prime annuelle estimée : {prediction:,.0f} FCFA**".replace(",", " "))

# ---------------------------------------------------------
# O N G L E T 2 : CLASSIFICATION NLP DES SINISTRES
# ---------------------------------------------------------
with tab2:
    st.header("Évaluation de la Gravité d'un Sinistre par NLP")
    st.markdown("Saisissez une description ou cliquez sur un exemple ci-dessous pour tester le modèle :")

    # Initialisation de la variable de session si elle n'existe pas
    if "texte_sinistre" not in st.session_state:
        st.session_state["texte_sinistre"] = ""

    # Boutons d'exemples pré-remplis
    col_ex1, col_ex2, col_ex3 = st.columns(3)

    with col_ex1:
        if st.button("🟢 Exemple 1 (Faible)"):
            st.session_state[
                "texte_sinistre"] = "Léger accrochage sur le pare-chocs arrière par une moto Jakarta. Simple rayure sur la peinture."

    with col_ex2:
        if st.button("🟠 Exemple 2 (Moyenne)"):
            st.session_state[
                "texte_sinistre"] = "Collision latérale avec un taxi au rond-point VDN. Portière enfoncée et rétroviseur cassé, aucun blessé."

    with col_ex3:
        if st.button("🔴 Exemple 3 (Élevée)"):
            st.session_state[
                "texte_sinistre"] = "Choc frontal violent avec un camion sur l'autoroute à péage. Véhicule totalement détruit et intervention des secours."

    # Zone de texte liée à la session state
    description = st.text_area(
        "Description textuelle du sinistre par l'assuré :",
        value=st.session_state["texte_sinistre"],
        height=120,
        placeholder="Exemple : Retro frotté par une moto Jakarta au niveau du rond-point VDN dans les embouteillages."
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