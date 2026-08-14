from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/logistic_pipeline.pkl")


def load_model():
    return joblib.load(MODEL_PATH)


def main():

    model = load_model()

    sample = pd.DataFrame(
        {
            "age": [39],
            "workclass": ["State-gov"],
            "fnlwgt": [77516],
            "education": ["Bachelors"],
            "education-num": [13],
            "marital-status": ["Never-married"],
            "occupation": ["Adm-clerical"],
            "relationship": ["Not-in-family"],
            "race": ["White"],
            "sex": ["Male"],
            "capital-gain": [2174],
            "capital-loss": [0],
            "hours-per-week": [40],
            "native-country": ["United-States"],
        }
    )

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample).max()

    print("=" * 60)
    print("Prediction")
    print("=" * 60)

    print(f"Predicted Income : {prediction}")
    print(f"Confidence : {probability:.2%}")


if __name__ == "__main__":
    main()