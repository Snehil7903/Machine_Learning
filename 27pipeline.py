# Why Pipelines Are Amazing

# Suppose you forget to scale the test data.

# Oops.

# Your model performs terribly.

# Pipelines prevent mistakes like this.

# Benefits:

# ✅ No data leakage

# ✅ Cleaner code

# ✅ Easier deployment

# ✅ Easier hyperparameter tuning

# ✅ Industry standard

# Instead of dozens of preprocessing lines, they write something like:

# pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("model", LinearRegression())
# ])

# pipeline.fit(X_train, y_train)

# That's it.
# Everything happens automatically.

# pipeline.predict(X_test)


# Hyperparameter Tuning
# Suppose tomorrow we use GridSearchCV.
# Without Pipeline...
# Nightmare.
# Because now GridSearch has to remember:
# Encoding
# Scaling
# Model
# With Pipeline
# GridSearch treats everything as ONE object.
# This is exactly how companies use it.


# Deployment
# Imagine you trained a model today.
# Three months later...
# Your company gets a new employee's data.
# Without Pipeline
# You must remember:
# Fill Missing Values
# ↓
# Encode
# ↓
# Scale
# ↓
# Predict
# Miss one step...
# Prediction becomes wrong.

# With Pipeline
# pipeline.predict(new_employee)
# Done.
# Everything happens automatically.

# Pipelines automate preprocessing and model training into a single workflow. They reduce repetitive code, prevent preprocessing mistakes and data leakage, ensure consistent transformations during training and inference, integrate seamlessly with cross-validation and hyperparameter tuning, and make deployment much easier.



from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

num_features = ["Age", "Education"]

cat_features = ["Education","City"]

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regression",LinearRegression())
])

model.fit(X_train, y_train)

prediction = model.predict(X_test)

# No manual preprocessing.
# The pipeline handles everything.

# If your ML model crashed every time it saw a new city...
# it would be useless.
# So engineers almost always write
# handle_unknown="ignore"

# Suppose there are
# 200 Countries
# and every month
# 5 new countries appear.
# One-Hot Encoding starts becoming less attractive.
# Engineers may then use techniques like:
# Target Encoding
# Frequency Encoding
# Embeddings (Deep Learning)