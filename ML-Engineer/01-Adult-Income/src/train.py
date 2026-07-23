import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from preprocess import preprocessor

df = pd.read_csv("E:\\restart\\MachineLearning\\ML-Engineer\\01-Adult-Income\\data\\adult.csv")

df.replace("?", np.nan, inplace=True)

X = df.drop("income", axis=1)

y = df["income"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

joblib.dump(
    model,
    "E:\\restart\\MachineLearning\\ML-Engineer\\01-Adult-Income\\models\\adult_income_pipeline.pkl"
)







predictions = model.predict(X_test)

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

print("Accuracy")

print(accuracy_score(y_test, predictions))

print(confusion_matrix(
    y_test,
    predictions
))

print(classification_report(
    y_test,
    predictions
))