from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# Which algorithms are highly sensitive to feature scaling?

# Answer:

# KNN ✅
# Logistic Regression ✅
# SVM ✅
# Neural Networks ✅

# Which algorithms are generally less sensitive?

# Decision Trees ✅
# Random Forests ✅
# XGBoost ✅

# Interviewers ask this surprisingly often.