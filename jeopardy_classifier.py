
import pandas as pd
import numpy as np
import nltk

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Jeopardy data
data = pd.read_json("jeopardy.json")

# Remove missing questions and values
data = data.dropna(subset=["question", "value"])

# Clean question values
data["value"] = data["value"].astype(str)
data["value"] = data["value"].str.replace("$", "", regex=False)
data["value"] = data["value"].str.replace(",", "", regex=False)
data["value"] = pd.to_numeric(data["value"], errors="coerce")

# Remove invalid values
data = data.dropna(subset=["value"])
data = data[data["value"] > 0]

# Create high-value and low-value categories
# High-value questions are worth $1000 or more
data["label"] = np.where(
    data["value"] >= 1000,
    "High-value",
    "Low-value"
)

# Set up text cleaning tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Clean and process questions
def clean_text(text):
    text = text.lower()

    words = word_tokenize(text)

    words = [word for word in words if word.isalpha()]

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

# Apply text cleaning
data["clean_question"] = data["question"].apply(clean_text)

# Remove empty questions
data = data[data["clean_question"] != ""]

# Separate questions and labels
X = data["clean_question"]
y = data["label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Convert text into TF-IDF features
vectorizer = TfidfVectorizer(use_idf=True)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Naive Bayes classifier
classifier = MultinomialNB(alpha=0.5, fit_prior=False)
classifier.fit(X_train, y_train)

# Make predictions
predictions = classifier.predict(X_test)

# Evaluate classifier
accuracy = accuracy_score(y_test, predictions)

# Display results
print("Jeopardy Question Value Classifier")
print("----------------------------------")
print("Total questions used:", len(data))
print("Training questions:", X_train.shape[0])
print("Testing questions:", X_test.shape[0])
print("Accuracy: {:.2f}%".format(accuracy * 100))

print("\nClassification Report:")
print(classification_report(y_test, predictions))
