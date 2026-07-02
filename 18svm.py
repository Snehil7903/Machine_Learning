# Support Vectors are the training samples closest to the decision boundary.

# RBF (Most Common)
# kernel="rbf"

import pandas as pd 
from sklearn.svm import SVC

model = SVC(
    kernel = "linear"
)

model.fit(X_train,y_train)

pred = model.predict(X_test)