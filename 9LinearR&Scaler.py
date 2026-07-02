import pandas as pd
data = {
    "Experience":[1,2,3,5,8],
    "Education":[12,12,14,16,16],
    "Age":[22,23,25,28,32],
    "Salary":[30000,35000,45000,70000,100000]
}

df = pd.DataFrame(data)
X = df[["Experience","Education","Age"]]
y = df["Salary"]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.4,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)

print(model.score(X_test, y_test))

print(model.coef_)
print(model.intercept_)

