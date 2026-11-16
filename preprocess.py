
import pandas as pd
import re
import os
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Keep words that are important for sentiment
stop_words = stop_words - {"not", "no", "nor", "never"}

# Text cleaning function
def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove user mentions
    text = re.sub(r"@\w+", "", text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r"#", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z\s']", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


# Create output folder
os.makedirs("dataset/processed", exist_ok=True)

# Process all three datasets
for split in ["train", "validation", "test"]:

    print(f"Processing {split} dataset...")

    df = pd.read_csv(f"dataset/{split}.csv")

    # Remove missing values
    df = df.dropna(subset=["text", "sentiment"])

    # Clean text
    df["text"] = df["text"].apply(clean_text)

    # Save processed dataset
    df.to_csv(
        f"dataset/processed/{split}.csv",
        index=False
    )

    print(f"{split} preprocessing completed!")

print("\nAll datasets preprocessed successfully!")
