import joblib

model = joblib.load(
    "E:\\restart\\MachineLearning\\ML-Engineer\\01-Adult-Income\\models\\adult_income_pipeline.pkl"
)

# prediction = model.predict(new_data)