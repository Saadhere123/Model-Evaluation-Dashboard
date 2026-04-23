import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_models():
    # Current file path
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Dataset path
    dataset_path = os.path.join(base_dir, "dataset_500_samples.csv")

    # Models folder create
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save model path
    model_path = os.path.join(models_dir, "saved_models.pkl")

    # Read dataset
    df = pd.read_csv(dataset_path)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer()

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": LinearSVC(),
        "Naive Bayes": MultinomialNB()
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, pred), 2),
            "Precision": round(precision_score(y_test, pred), 2),
            "Recall": round(recall_score(y_test, pred), 2),
            "F1-score": round(f1_score(y_test, pred), 2)
        })

        trained_models[name] = model

    # Save models
    joblib.dump((vectorizer, trained_models), model_path)

    return pd.DataFrame(results)