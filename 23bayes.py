# Probability(email is Spam | word = FREE)
# This is called Conditional Probability.

# P(A|B) = P(A INTERSECTION B)/P(B)
# P(A|B) = P(B|A)P(A)/P(B)

# Naive Bayes assumes:
# Every feature is independent of every other feature.

# Surprisingly...
# Even though the assumption is wrong...
# Naive Bayes performs amazingly well for:
# Spam Detection ✅
# Sentiment Analysis ✅
# Text Classification ✅
# News Classification ✅
# One of the coolest algorithms in ML.

# Types of Naive Bayes

# Gaussian Naive Bayes
# Continuous numerical data

# Multinomial Naive Bayes
# Word counts

# Bernoulli Naive Bayes
# 0 / 1
# True / False


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = GaussianNB()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print(pred)

print(
    accuracy_score(y_test, pred)
)

# no scaling
# | Algorithm            | Scaling Needed? |
# | -------------------- | --------------- |
# | KNN                  | ✅ Yes           |
# | SVM                  | ✅ Yes           |
# | Logistic Regression  | ✅ Yes           |
# | Neural Networks      | ✅ Yes           |
# | Decision Tree        | ❌ No            |
# | Random Forest        | ❌ No            |
# | XGBoost              | ❌ No            |
# | Gaussian Naive Bayes | ❌ Generally No  |
