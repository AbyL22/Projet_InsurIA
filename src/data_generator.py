import pandas as pd
import numpy as np
import random
import os


def generate_insurance_data(n_samples=1000, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # Données tabulaires (Tarification / Actuariat)
    age = np.random.randint(18, 70, size=n_samples)
    experience_permis = np.clip(age - 18 - np.random.randint(0, 5, size=n_samples), 0, None)
    puissance_vehicule = np.random.choice([4, 6, 8, 10, 12], size=n_samples)
    valeur_vehicule = np.random.randint(3000, 35000, size=n_samples)
    zone_risque = np.random.choice(['Urbaine', 'Suburbaine', 'Rurale'], size=n_samples, p=[0.5, 0.3, 0.2])

    # Calcul théorique de la prime pure
    prime_base = 200
    prime = (
            prime_base
            + (100 - age) * 2
            + puissance_vehicule * 15
            + (valeur_vehicule * 0.02)
            + np.where(zone_risque == 'Urbaine', 150, np.where(zone_risque == 'Suburbaine', 75, 0))
            + np.random.normal(0, 50, size=n_samples)
    )

    # Données textuelles : Déclarations de sinistres (NLP)
    textes_leger = [
        "Un léger impact sur le pare-chocs avant lors d'un créneau.",
        "Rétroviseur droit fissuré dans un parking de supermarché.",
        "Petite rayure sur la portière conducteur.",
        "Impact de gravillon sur le pare-brise sans fissure majeure."
    ]

    textes_moyen = [
        "Accident à un croisement, aile droite et phare enfoncés.",
        "Choc à basse vitesse à un feu rouge, capot tordu.",
        "Glissade sous la pluie, choc contre la bordure avec train avant abîmé.",
        "Refus de priorité, porte latérale fortement bosselée."
    ]

    textes_grave = [
        "Collision frontale violente sur l'autoroute, airbags déclenchés.",
        "Tonneau suite à une perte de contrôle, véhicule totalement détruit.",
        "Choc frontal avec un camion, moteur déplacé et châssis déformé.",
        "Incendie complet du véhicule suite à une fuite de carburant."
    ]

    gravites = np.random.choice(['Faible', 'Moyenne', 'Élevée'], size=n_samples, p=[0.5, 0.35, 0.15])

    descriptions = []
    for g in gravites:
        if g == 'Faible':
            descriptions.append(random.choice(textes_leger))
        elif g == 'Moyenne':
            descriptions.append(random.choice(textes_moyen))
        else:
            descriptions.append(random.choice(textes_grave))

    df = pd.DataFrame({
        'age_conducteur': age,
        'experience_permis': experience_permis,
        'puissance_vehicule': puissance_vehicule,
        'valeur_vehicule': valeur_vehicule,
        'zone_risque': zone_risque,
        'prime_estimee': np.round(prime, 2),
        'description_sinistre': descriptions,
        'gravite_sinistre': gravites
    })

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_insurance_data(1000)
    df.to_csv("data/insurance_claims_dataset.csv", index=False)
    print("✅ Dataset généré avec succès dans 'data/insurance_claims_dataset.csv' !")