from sklearn.model_selection import cross_val_score

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

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


model = KNeighborsClassifier(n_neighbors=3)

scores = cross_val_score(
    model,
    X,
    y,
    cv=5
)

print(scores)
print(scores.mean())