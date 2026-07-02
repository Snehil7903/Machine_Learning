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

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    max_depth=5
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(predictions)

from sklearn.metrics import accuracy_score

print(
    accuracy_score(y_test, predictions)
)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plot_tree(
    model,
    feature_names=["Hours"],
    class_names=["Fail","Pass"],
    filled=True
)

plt.show()