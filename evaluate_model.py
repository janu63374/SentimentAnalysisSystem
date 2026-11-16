
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load test dataset
print("Loading test dataset...")

df = pd.read_csv("dataset/processed/test.csv")

# Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# Remove empty text records
df = df[df["text"].str.strip() != ""]

X_test = df["text"].astype(str)
y_test = df["sentiment"]
# Load saved model and vectorizer
print("\nLoading trained model...")

model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Convert test text into numerical features
X_test_tfidf = tfidf.transform(X_test)

# Predict sentiments
print("\nEvaluating model...")

y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nTest Accuracy:", round(accuracy * 100, 2), "%")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
labels = ["negative", "neutral", "positive"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels
)

plt.title("Sentiment Analysis Confusion Matrix")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.tight_layout()
plt.savefig("confusion_matrix.png")

plt.show()

print("\nConfusion matrix saved successfully!")
