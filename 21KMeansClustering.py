from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=2,
    random_state=42
)

model.fit(X)

print(model.labels_)

print(model.cluster_centers_)