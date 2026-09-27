import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_pricing_model():
    data_path = "data/insurance_claims_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Le fichier de données n'existe pas. Exécutez d'abord data_generator.py")

    df = pd.read_csv(data_path)

    X = df[['age_conducteur', 'experience_permis', 'puissance_vehicule', 'valeur_vehicule', 'zone_risque']]
    y = df['prime_estimee']

    num_features = ['age_conducteur', 'experience_permis', 'puissance_vehicule', 'valeur_vehicule']
    cat_features = ['zone_risque']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ]
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"✅ Entraînement du modèle terminé !")
    print(f"📊 RMSE : {rmse:.2f} €")
    print(f"📊 R² Score : {r2:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model_pipeline, "models/pricing_model.pkl")
    print("💾 Modèle sauvegardé dans 'models/pricing_model.pkl'")


if __name__ == "__main__":
    train_pricing_model()