import pandas as pd

#Load Data
df = pd.read_csv("salary.csv")

#Inspect Data
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

#HandleMissingValues
from sklearn.impute import SimpleImputer

num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

#Encode Categorical Features
from sklearn.preprocessing import OneHotEncoder

#Scale Numerical Features
from sklearn.preprocessing import StandardScaler

#Combination
from sklearn.compose import ColumnTransformer
# This lets us say:
# Apply StandardScaler to numerical columns.
# Apply OneHotEncoder to categorical columns.
# All in one preprocessing pipeline.