# Problem Statement
#         ↓
# Understand Dataset
#         ↓
# EDA (Exploratory Data Analysis)
#         ↓
# Handle Missing Values
#         ↓
# Feature Engineering
#         ↓
# Encoding
#         ↓
# Scaling
#         ↓
# Train/Test Split
#         ↓
# Train Multiple Models
#         ↓
# Cross Validation
#         ↓
# Grid Search
#         ↓
# Model Comparison
#         ↓
# Final Evaluation
#         ↓
# Save the Model
#         ↓
# Deploy the Model



# WorkFlow

# Business Problem
#         ↓
# Collect Data
#         ↓
# EDA
#         ↓
# Cleaning
#         ↓
# Feature Engineering
#         ↓
# Split
#         ↓
# Train
#         ↓
# Evaluate
#         ↓
# Tune
#         ↓
# Deploy



import pandas as pd

#load data
df = pd.read_csv("salary.csv")

print(df.head())
# What columns exist?
# Are there missing values?
# Are there categorical columns?
# Does everything look reasonable?

print(df.info())
# This tells us
# Number of rows
# Number of columns
# Data types
# Missing values

print(df.describe())
# This gives statistics.
# For numerical columns:
# Mean
# Median
# Standard Deviation
# Min
# Max

print(df.isnull().sum())
# It tells us missing values.

# THIS is called EDA
# Exploratory Data Analysis.


# | Column     | Missing | Good Strategy              |
# | ---------- | ------- | -------------------------- |
# | Age        | 2       | Fill with Median ✅         |
# | Experience | 2       | Fill with Median or Mean ✅ |
# | Education  | 5       | Fill with Mode ✅           |
# | City       | 1       | Fill with Mode ✅           |

df["Experience"].fillna(df["Experience"].median())

# When should you use Mean vs Median?
# Answer:
# Mean → Data is approximately symmetric and has no major outliers.
# Median → Data contains outliers or is skewed.


# Outlier Found
#       ↓
# Investigate
#       ↓
# Is it an error?
#       ↓
# Yes → Correct/Delete

# No
#       ↓
# Keep 


