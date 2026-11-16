
import pandas as pd

df = pd.read_csv("dataset/train.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nSentiment Distribution:")
print(df["sentiment"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum())
