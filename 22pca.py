# That's exactly what PCA does.

# 100 Features
#       ↓
# PCA
#       ↓
# 10 New Features

# Those new features are called Principal Components.

# Generally:

# Fewer features → Faster training
# Less memory usage
# Reduced noise
# Lower risk of overfitting (though not always)
# Easier visualization

# This is called Dimensionality Reduction.

# PCA says:

# "Instead of measuring movement in the X direction and Y direction separately, why don't I rotate my axes?"

# After rotation:

# -----------------------> PC1

# Most of the variation lies along PC1.

# The second direction (PC2) contains very little information.

# So we can often discard PC2 and lose almost nothing.

# This is the heart of PCA.


# Raw Data
#       ↓
# Handle Missing Values
#       ↓
# Encode Categorical Variables
#       ↓
# Feature Scaling
#       ↓
# PCA
#       ↓
# Train Model


#Taking help of the iris data...
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
X = iris.data
print(X.shape)

#Standard Scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(X_pca.shape)
print(pca.explained_variance_ratio_)

import matplotlib.pyplot as plt

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=iris.target
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# from sklearn.decomposition import PCA
# pca = PCA(n_components=0.95)
# X_new = pca.fit_transform(X_scaled)
# print(X_new.shape)


# The Biggest Disadvantage of PCA

# This is one of Google's favorite interview questions.

# Suppose PCA creates:

# PC1

# Question:

# What does

# PC1

# actually mean?

# You cannot answer.

# Maybe it is

# 0.4 × Height
# +
# 0.7 × Weight
# -
# 0.2 × Age

# It's difficult to interpret.

# This is called:

# Loss of Interpretability.

# That's the biggest disadvantage of PCA.