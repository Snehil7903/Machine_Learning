data = {
    "Name":["A","B","C","D"],
    "Age":[20,21,19,22],
    "Marks":[85,92,78,88]
}

import pandas as pd

df = pd.DataFrame(data)

df.iloc[0:2] 

df.shape

df["Marks"]

df["Marks"].max()

df["Marks"].mean()

df[df["Marks"]>80]

df["Passed"] = ["yes","yes","yes","yes"]

print(df.describe())

print(df)