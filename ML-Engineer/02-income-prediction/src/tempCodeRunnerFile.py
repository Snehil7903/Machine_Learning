from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from preprocess import create_preprocessor
from evaluate import evaluate_model


DATA_PATH = Path("E:\\restart\\MachineLearning\\ML-Engineer\\02-income-prediction\\data\\adult.csv")
MODEL_DIR = Path("models")


def load_data():
    """
    Load the Adult Income dataset.
    """

    df = pd.read_csv(DATA_PATH)

    # Replace missing values represented by '?'
    df.replace("?", np.nan, inplace=True)

    return df


def train():
    """
    Train a Logistic Regression model.
    """

    df = load_data()

    # Features and Target
    X = df.drop("income", axis=1)
    y = df["income"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Preprocessing
    preprocessor = create_preprocessor(X_train)

    # Model Pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    # Train
    model.fit(X_train, y_train)

    # Training Accuracy
    train_pred = model.predict(X_train)

    train_accuracy = accuracy_score(y_train, train_pred)

    print("=" * 60)
    print("LOGISTIC REGRESSION")
    print("=" * 60)

    print(f"\nTraining Accuracy : {train_accuracy:.4f}\n")

    # Test Evaluation
    evaluate_model(
        model,
        X_test,
        y_test,
        model_name="logistic"
    )

    # Save Model
    MODEL_DIR.mkdir(exist_ok=True)

    joblib.dump(
        model,
        MODEL_DIR / "logistic_pipeline.pkl"
    )

    print("\nModel saved successfully!")
    print("Location : models/logistic_pipeline.pkl")


def main():
    train()


if __name__ == "__main__":
    main()