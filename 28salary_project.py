import pandas as pd

# Load dataset
df = pd.read_csv("salary.csv")

# First 5 rows
print(df.head())

# Shape
print("\nShape:")
print(df.shape)

# Information
print("\nInfo:")
print(df.info())

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicates:")
print(df.duplicated().sum())

# Statistics
print("\nStatistics:")
print(df.describe())

# Column names
print("\nColumns:")
print(df.columns)