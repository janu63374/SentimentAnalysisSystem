# Sentiment Analysis System
A Machine Learning-based web application that classifies text into **Positive, Negative, or Neutral** sentiments using Natural Language Processing (NLP).

## Features
- Text preprocessing and cleaning.
- TF-IDF feature extraction.
- Sentiment classification using Logistic Regression.
- Prediction with confidence score.
- Model evaluation and confusion matrix.
- Interactive web interface using Flask.

## Technologies Used
- **Language:** Python
- **Machine Learning:** Scikit-learn
- **NLP:** NLTK
- **Data Processing:** Pandas, NumPy
- **Web Framework:** Flask
- **Frontend:** HTML, CSS
- **Visualization:** Matplotlib, Seaborn

## Project Workflow
1. **Data Collection:** Collected the TweetEval sentiment dataset.
2. **Preprocessing:** Cleaned text and removed unnecessary words and symbols.
3. **Feature Extraction:** Converted text into numerical features using TF-IDF.
4. **Model Training:** Trained a Logistic Regression classifier.
5. **Evaluation:** Evaluated using accuracy, precision, recall, F1-score, and confusion matrix.
6. **Implementation:** Integrated the model into a Flask web application.

## Model Performance
- **Test Accuracy:** 58.31%
- **Evaluation Metrics:** Precision, Recall, F1-score, and Confusion Matrix.

## Installation and Execution
**1. Clone the repository**
git clone https://github.com/janu63374/SentimentAnalysisSystem.git

**2. Navigate to the project folder**
cd SentimentAnalysisSystem

**3. Create and activate virtual environment**
python -m venv venv
venv\Scripts\activate

**4. Install dependencies**
pip install -r requirements.txt

**5. Run the application**
python app.py

Open `http://127.0.0.1:5000` in your browser.

## Project Structure
SentimentAnalysisSystem/
├── dataset/
├── models/
├── static/
├── templates/
├── app.py
├── preprocess.py
├── train_model.py
├── evaluate_model.py
├── download_dataset.py
├── requirements.txt
└── README.md


## Author
**Janani Selvaraj**  
M.Sc. Computer Science
GitHub: [janu63374](https://github.com/janu63374)
