import numpy as np

scores = np.array([
    65, 80, 90, 45, 75,
    88, 92, 55, 70, 99
])

np.mean(scores)
np.median(scores)
np.var(scores)
np.std(scores)
np.max(scores)
np.min(scores)
scores1=scores[scores>=80]

np.sort(scores)
np.unique(scores)

np.sum(scores>70)