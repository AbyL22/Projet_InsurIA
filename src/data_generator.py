import os
import random
import pandas as pd
import numpy as np

# Création du dossier data s'il n'existe pas
os.makedirs("data", exist_ok=True)

# Set seed pour la reproductibilité
np.random.seed(42)
random.seed(42)

N_SAMPLES = 5000

print(f"Génération de {N_SAMPLES} données tabulaires (Sénégal)...")

# --- 1. GÉNÉRATION DES DONNÉES TABULAIRES (Tarification ML) ---

villes_regions = [
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
]

# Pondération des zones de risque (Dakar = risque plus élevé en raison de la densité)
zones_risque = {
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

types_vehicules = [
    "Particulier (Berline / SUV)",
    "Taxi Jaune-Noir",
    "Transport / Clandos",
    "Moto / Jakarta",
    "Camion / Utilitaire"
]

age_conducteur = np.random.randint(18, 70, size=N_SAMPLES)
# L'expérience du permis dépend de l'âge
experience_permis = np.array([
    random.randint(0, max(0, age - 18)) for age in age_conducteur
])

puissance_vehicule = np.random.choice([4, 6, 7, 9, 11, 14], size=N_SAMPLES, p=[0.2, 0.3, 0.25, 0.15, 0.07, 0.03])
villes_sample = np.random.choice(villes_regions, size=N_SAMPLES)
zone_risque_sample = [zones_risque[v] for v in villes_sample]
type_vehicule_sample = np.random.choice(types_vehicules, size=N_SAMPLES, p=[0.5, 0.2, 0.15, 0.1, 0.05])

# Calcul du coût annuel estimé de la prime en FCFA (Prime de base + facteurs de risque)
prime_base = 120000  # 120 000 FCFA

prime_annuelle = []
for i in range(N_SAMPLES):
    prime = prime_base

    # Ajustement selon la zone
    if zone_risque_sample[i] == "Très Élevé":
        prime += 65000
    elif zone_risque_sample[i] == "Élevé":
        prime += 45000
    elif zone_risque_sample[i] == "Moyen":
        prime += 20000

    # Ajustement selon l'expérience
    if experience_permis[i] < 2:
        prime += 50000
    elif experience_permis[i] < 5:
        prime += 25000

    # Ajustement selon la puissance
    prime += puissance_vehicule[i] * 4500

    # Ajustement selon le type de véhicule
    if type_vehicule_sample[i] == "Taxi Jaune-Noir":
        prime += 35000
    elif type_vehicule_sample[i] == "Moto / Jakarta":
        prime += 15000
    elif type_vehicule_sample[i] == "Camion / Utilitaire":
        prime += 80000

    # Variabilité aléatoire
    prime += random.randint(-10000, 10000)
    prime_annuelle.append(int(prime))

df_tabular = pd.DataFrame({
    "age_conducteur": age_conducteur,
    "experience_permis": experience_permis,
    "puissance_vehicule": puissance_vehicule,
    "ville": villes_sample,
    "zone_risque": zone_risque_sample,
    "type_vehicule": type_vehicule_sample,
    "prime_annuelle_fcfa": prime_annuelle
})

df_tabular.to_csv("data/donnees_tarification.csv", index=False)
print("-> Fichier 'data/donnees_tarification.csv' généré avec succès !")

# --- 2. GÉNÉRATION DES DONNÉES NLP (Gravité des Sinistres au Sénégal) ---

print("Génération du jeu de données NLP pour la classification de sinistres...")

phrases_faible = [
    "Légère rature sur la portière en faisant un créneau au marché Sandaga.",
    "Retro frotté par une moto Jakarta au niveau du rond-point VDN dans les embouteillages.",
    "Petit choc arrière à faible vitesse en stationnant près de la corniche Ouest.",
    "Feu arrière légèrement fissuré sur le parking du supermarché.",
    "Trace de peinture et petite bosse sur le pare-chocs avant suite à une manœuvre à Thiès.",
    "Optique de phare avant fêlé en évitant un nid-de-poule à Colobane.",
    "Rétroviseur droit cassé en passant dans une ruelle étroite à Médina.",
    "Légère égratignure sur le pare-chocs arrière suite à un freinage doux."
]

phrases_moyenne = [
    "Choc latéral avec un taxi jaune-noir au niveau de l'échangeur de Patte d'Oie, portière enfoncée.",
    "Accident sur l'avenue Cheikh Anta Diop : impact à l'arrière avec un clandos, coffre bloqué.",
    "Refus de priorité au carrefour Castors, pare-chocs détruit et radiateur touché mais véhicule roulant.",
    "Acrochage à Rufisque avec une camionnette, aile gauche froissée et pare-brise fissuré.",
    "Pneu éclaté sur la route de Saly provoquant une sortie de route dans le décor, carrosserie abîmée.",
    "Collision modérée vers le rond-point Camberène avec dégâts matériels importants, pas de blessé.",
    "Choc frontal à faible vitesse avec un Ndiaga Ndiaye, capot plié et phares brisés."
]

phrases_forte = [
    "Collision frontale violente sur la Route Nationale RN1 entre Thiès et Dakar entre une berline et un camion.",
    "Accident grave sur l'Autoroute à Péage : plusieurs tonneaux à haute vitesse, voiture totalement détruite.",
    "Choc très violent vers Keur Massar, véhicule déclaré épave et conducteur évacué en urgence à l'hôpital Principal.",
    "Perte de contrôle totale de nuit vers Kaolack, plusieurs blessés graves et dégâts matériels totaux.",
    "Accident grave impliquant un car rapide et deux véhicules à Keur Mbaye Fall, secours et pompiers sur place.",
    "Collision à grande vitesse sur la route de Saint-Louis, voiture broyée et désincarcération nécessaire."
]

texts = []
labels = []

# Génération par tirage aléatoire avec variations
for _ in range(1500):
    texts.append(random.choice(phrases_faible))
    labels.append("Faible")

for _ in range(1500):
    texts.append(random.choice(phrases_moyenne))
    labels.append("Moyenne")

for _ in range(1500):
    texts.append(random.choice(phrases_forte))
    labels.append("Forte")

df_nlp = pd.DataFrame({
    "description_sinistre": texts,
    "gravite": labels
})

# Mélange des données
df_nlp = df_nlp.sample(frac=1, random_state=42).reset_index(drop=True)
df_nlp.to_csv("data/donnees_sinistres_nlp.csv", index=False)
print("-> Fichier 'data/donnees_sinistres_nlp.csv' généré avec succès !")