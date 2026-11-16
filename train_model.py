
import pandas as pd
import os
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load processed datasets
print("Loading datasets...")

train_df = pd.read_csv("dataset/processed/train.csv")
val_df = pd.read_csv("dataset/processed/validation.csv")

X_train = train_df["text"]
y_train = train_df["sentiment"]

X_val = val_df["text"]
y_val = val_df["sentiment"]

# Convert text into numerical features using TF-IDF
print("\nConverting text into numerical features...")

tfidf = TfidfVectorizer(
    max_features=50000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

# Create Logistic Regression model
print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train the model
model.fit(X_train_tfidf, y_train)

print("Model training completed!")

# Evaluate using validation dataset
y_pred = model.predict(X_val_tfidf)

print("\nValidation Accuracy:")
print(accuracy_score(y_val, y_pred))

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Save trained model and TF-IDF vectorizer
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModel and vectorizer saved successfully!")
