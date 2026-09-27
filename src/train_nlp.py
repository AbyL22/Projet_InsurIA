import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def train_nlp_model():
    # 1. Chargement du dataset
    data_path = "data/insurance_claims_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError("Le fichier de données est introuvable. Exécutez data_generator.py d'abord.")

    df = pd.read_csv(data_path)

    X = df['description_sinistre']
    y = df['gravite_sinistre']

    # 2. Vectorisation du texte (TF-IDF) & Séparation des données
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    # 3. Entraînement du classifieur
    clf = LogisticRegression(random_state=42)
    clf.fit(X_train, y_train)

    # 4. Évaluation
    predictions = clf.predict(X_test)
    print("✅ Modèle NLP entraîné avec succès !")
    print("\n📊 Rapport de classification :")
    print(classification_report(y_test, predictions))

    # 5. Sauvegarde des artefacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/nlp_vectorizer.pkl")
    joblib.dump(clf, "models/nlp_model.pkl")
    print("💾 Modèles NLP sauvegardés dans le dossier 'models/' !")


if __name__ == "__main__":
    train_nlp_model()