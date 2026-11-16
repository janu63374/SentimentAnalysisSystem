import pandas as pd

df = pd.read_csv("dataset/processed/train.csv")

print("First 5 cleaned records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())