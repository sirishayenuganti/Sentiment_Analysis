import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import csv
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.properties import StringProperty
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize VADER
vader_analyzer = SentimentIntensityAnalyzer()

# Function to load the dataset with robust CSV parsing
def load_amazon_reviews(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please provide a valid file path.")
    try:
        return pd.read_csv(file_path, quoting=csv.QUOTE_ALL, escapechar='\\', sep=',')
    except pd.errors.ParserError as e:
        print(f"ParserError: {e}")
        print("Attempting to load with error handling (skipping bad lines)...")
        try:
            return pd.read_csv(file_path, quoting=csv.QUOTE_ALL, escapechar='\\', sep=',', on_bad_lines='warn')
        except Exception as e2:
            raise Exception(f"Failed to load CSV even with error handling: {e2}")
    except Exception as e:
        raise Exception(f"Unexpected error loading CSV: {e}")

# Modified text preprocessing function
def preprocess_text(text, keep_negation=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Preserve some punctuation for sentiment (e.g., !)
    text = re.sub(r'[^\w\s!]', '', text)
    tokens = word_tokenize(text)
    # Customize stopwords to keep negation words
    stop_words = set(stopwords.words('english')) - {'not', 'no', 'never'}
    if keep_negation:
        tokens = [token for token in tokens if token not in stop_words]
    else:
        tokens = [token for token in tokens if token not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Load and prepare the dataset
file_path = r"C:\Users\HP\OneDrive\Desktop\website\amazon_reviews.csv"
df = load_amazon_reviews(file_path)

# Check sentiment distribution
print("Sentiment Distribution:")
print(df['overall'].value_counts(normalize=True))

# Define sentiment labels
def get_sentiment(rating):
    if pd.isna(rating):
        return None
    rating = float(rating)
    if rating >= 4:
        return 'positive'
    elif rating == 3:
        return 'neutral'
    else:
        return 'negative'

# Apply sentiment labels
df['sentiment'] = df['overall'].apply(get_sentiment)
df = df.dropna(subset=['reviewText', 'sentiment'])
df['cleaned_text'] = df['reviewText'].apply(lambda x: preprocess_text(x, keep_negation=True))
df = df[df['cleaned_text'] != '']

# Prepare data for training
X = df['cleaned_text']
y = df['sentiment']
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))  # Extended to trigrams
X_tfidf = vectorizer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_model.fit(X_train, y_train)

# Train Random Forest for comparison
rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate models
print("\nLogistic Regression Evaluation:")
lr_y_pred = lr_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, lr_y_pred):.2f}")
print(classification_report(y_test, lr_y_pred))

print("\nRandom Forest Evaluation:")
rf_y_pred = rf_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, rf_y_pred):.2f}")
print(classification_report(y_test, rf_y_pred))

# Function to predict sentiment with hybrid approach
def predict_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return "Invalid input: Text is empty"
    # VADER for short inputs
    vader_scores = vader_analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']
    if len(text.split()) <= 5:  # Use VADER for short texts
        if vader_compound >= 0.05:
            return 'positive'
        elif vader_compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    # Preprocess and use ML model for longer texts
    cleaned_text = preprocess_text(text, keep_negation=True)
    if not cleaned_text:
        return "Invalid input: No valid words"
    text_tfidf = vectorizer.transform([cleaned_text])
    # Use Random Forest for better nuance
    prediction = rf_model.predict(text_tfidf)[0]
    # Rule-based adjustment for intensifiers
    if 'very' in text.lower() and 'bad' in text.lower():
        prediction = 'negative'
    if 'not bad' in text.lower() or 'ok' in text.lower():
        prediction = 'neutral'
    return prediction

# Kivy App
class SentimentAppLayout(BoxLayout):
    prediction_text = StringProperty('')

    def predict_sentiment(self, review):
        prediction = predict_sentiment(review)
        self.prediction_text = f"Sentiment: {prediction.capitalize()}"

class SentimentApp(App):
    def build(self):
        return SentimentAppLayout()

if __name__ == '__main__':
    SentimentApp().run()