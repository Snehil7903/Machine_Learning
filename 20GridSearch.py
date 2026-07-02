from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

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


param_grid = {
    "n_neighbors":[1,3,5,7,9]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5
)

grid.fit(X,y)

print(grid.best_params_)

print(grid.best_score_)


