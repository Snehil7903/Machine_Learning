import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name":["A","B","C","D","E"],
    "Marks":[80,np.nan,75,90,np.nan],
    "Department":["IT","IT","HR","HR","IT"]
})

print(df.isnull())
print(df.isnull().sum())
df.fillna(df["Marks"].mean(), inplace=True)
print(df)

print(df.groupby("Department")["Marks"].mean())

print(df.sort_values("Marks", ascending=False))

df.info()

df.describe()
