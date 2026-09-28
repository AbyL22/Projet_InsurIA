import pandas as pd
import numpy as np
import os


def generate_data():
    np.random.seed(42)
    n_samples = 5000

    print("Génération de 5000 données tabulaires (Sénégal)...")

    # 1. Caractéristiques des conducteurs et véhicules
    age_conducteur = np.random.randint(18, 75, n_samples)
    experience_permis = np.clip(age_conducteur - 18 - np.random.randint(0, 5, n_samples), 0, 50)
    puissance_vehicule = np.random.randint(3, 20, n_samples)

    villes_list = [
        "Dakar - Plateau / Médina", "Dakar - Keur Massar / Pikine",
        "Dakar - VDN / Almadies", "Thiès", "Mbour / Saly",
        "Saint-Louis", "Touba / Mbacké", "Rufisque", "Ziguinchor", "Kaolack"
    ]

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

    types_vehicules_list = [
        "Particulier (Berline / SUV)", "Taxi Jaune-Noir",
        "Transport / Clandos", "Moto / Jakarta", "Camion / Utilitaire"
    ]

    villes = np.random.choice(villes_list, n_samples)
    zones_risque = [zones_dict[v] for v in villes]
    types_vehicules = np.random.choice(types_vehicules_list, n_samples, p=[0.5, 0.15, 0.1, 0.15, 0.1])

    # 2. Calcul réaliste de la Prime en FCFA
    prime_base = 40000  # Tarif de base : 40 000 FCFA

    # Facteurs multiplicateurs ajustés
    coeff_zone = {"Faible": 0.85, "Moyen": 1.0, "Élevé": 1.2, "Très Élevé": 1.35}
    coeff_vehicule = {
        "Moto / Jakarta": 0.6,
        "Particulier (Berline / SUV)": 1.0,
        "Taxi Jaune-Noir": 1.3,
        "Transport / Clandos": 1.4,
        "Camion / Utilitaire": 1.6
    }

    primes = []
    for i in range(n_samples):
        # Réduction avec l'âge et l'expérience
        malus_jeune = 1.3 if age_conducteur[i] < 25 else (1.15 if age_conducteur[i] < 30 else 1.0)
        bonus_exp = max(0.7, 1.0 - (experience_permis[i] * 0.01))

        # Impact de la puissance
        effet_puissance = 1.0 + (puissance_vehicule[i] * 0.03)

        # Calcul final de la prime
        p = prime_base * coeff_zone[zones_risque[i]] * coeff_vehicule[
            types_vehicules[i]] * malus_jeune * bonus_exp * effet_puissance

        # Ajout d'une petite variation aléatoire (+/- 5%)
        p *= np.random.uniform(0.95, 1.05)
        primes.append(round(p, -2))  # Arrondi à la centaine près

    df_tabular = pd.DataFrame({
        'age_conducteur': age_conducteur,
        'experience_permis': experience_permis,
        'puissance_vehicule': puissance_vehicule,
        'ville': villes,
        'zone_risque': zones_risque,
        'type_vehicule': types_vehicules,
        'prime_annuelle_fcfa': primes
    })

    os.makedirs("data", exist_ok=True)
    df_tabular.to_csv("data/donnees_tarification.csv", index=False)
    print("-> Fichier 'data/donnees_tarification.csv' généré avec succès !")


if __name__ == "__main__":
    generate_data()