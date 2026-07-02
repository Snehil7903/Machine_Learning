from sklearn.ensemble import RandomForestClassifier
import pandas as pd

data = {
    "Hours":[1,2,3,4,5,6],
    "Pass":[0,0,0,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=5
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(model.feature_importances_)

