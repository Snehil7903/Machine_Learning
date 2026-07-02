from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

data = {
    "Hours":[
        1,2,3,4,5,6,7,8,
        2,3,4,5,6,7,8,9,
        1,2,3,4,5,6,7,8
    ],
    "Pass":[
        0,0,0,1,1,1,1,1,
        0,0,1,1,1,1,1,1,
        0,0,0,1,1,1,1,1
    ]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.25,
    random_state=42
)

model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)

model.fit(X_train,y_train)

pred = model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))

print(model.feature_importances_)