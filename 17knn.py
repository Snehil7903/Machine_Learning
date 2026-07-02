from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

data = {
    "Hours":[1,2,3,4,5,6,7,8],
    "Pass":[0,0,0,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
    test_size=0.25,
    random_state=42
)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)

from sklearn.metrics import accuracy_score

print(
    accuracy_score(y_test,pred)
)


# Would KNN training be slow?

# Answer:

# No.

# KNN basically stores the data.

# Training is very fast.

# But prediction?

# Very expensive.

# Because for every new point:

# Compare with
# 1,000,000 samples

# This is one of KNN's biggest weaknesses.

# Why KNN Matters in Modern AI

# You might be surprised, but KNN concepts appear everywhere:

# Recommendation Systems
# Find similar users

# Image Search
# Find similar images
# Vector Databases
# Find nearest embeddings
# RAG Systems
# Find nearest documents

# The same nearest-neighbor idea powers many modern AI systems.


# One More Important Concept

# Suppose:

# K = 4

# Neighbors:

# Red
# Red
# Blue
# Blue

# Vote:

# 2 vs 2

# Tie.

# What now?

# Different libraries handle ties differently.

# This is one reason we often prefer:

# Odd K values

# such as:

# K = 3
# K = 5
# K = 7

# to avoid ties.

# Big Interview Insight

# Suppose an interviewer asks:

# How do you choose K?

# Good answer:

# K is a hyperparameter. Small K may overfit, while large K may underfit. We typically use cross-validation to find the best value of K.

# Memorize that.