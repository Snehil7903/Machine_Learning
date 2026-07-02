import pandas as pd

data = {
    "Hours":[1,2,3,4,5,6,7,8],
    "Pass":[0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train,y_train)

predictions = model.predict(X_test)

print(predictions)

probab = model.predict_proba(X_test)

print(probab)

from sklearn.metrics import accuracy_score

acc =accuracy_score(y_test, predictions)

print(acc)