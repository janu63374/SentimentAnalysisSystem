
from flask import Flask, render_template, request
import joblib
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)

# Load trained model and TF-IDF vectorizer
model = joblib.load("models/sentiment_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")

# Load stopwords
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# Preserve important negative words
stop_words = stop_words - {"not", "no", "nor", "never"}


# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)


@app.route("/", methods=["GET", "POST"])
def home():

    sentiment = None
    confidence = None
    original_text = ""
    error = None

    if request.method == "POST":

        original_text = request.form.get("text", "").strip()

        if not original_text:
            error = "Please enter some text to analyze."

        else:
            cleaned_text = clean_text(original_text)

            if not cleaned_text:
                error = "Please enter meaningful text."

            else:
                # Convert input into TF-IDF features
                text_features = tfidf.transform([cleaned_text])

                # Predict sentiment
                sentiment = model.predict(text_features)[0]

                # Calculate model confidence
                probabilities = model.predict_proba(text_features)[0]
                confidence = round(max(probabilities) * 100, 2)

    return render_template(
        "index.html",
        sentiment=sentiment,
        confidence=confidence,
        original_text=original_text,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)
