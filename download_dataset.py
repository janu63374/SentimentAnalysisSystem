
from datasets import load_dataset
import pandas as pd
import os

# Create dataset folder
os.makedirs("dataset", exist_ok=True)

# Download dataset
print("Downloading TweetEval dataset...")

data = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# Sentiment labels
labels = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

# Save datasets as CSV files
for split in ["train", "validation", "test"]:
    df = data[split].to_pandas()
    df["sentiment"] = df["label"].map(labels)
    df = df[["text", "sentiment"]]

    df.to_csv(f"dataset/{split}.csv", index=False)

    print(f"{split}.csv downloaded: {len(df)} records")

print("All datasets downloaded successfully!")
