import pandas as pd

df = pd.read_csv("data/student-mat.csv", sep=";")

print("Shape of data (rows, columns):", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nAny missing values?")
print(df.isnull().sum())
print("\nBasic stats:")
print(df.describe())
