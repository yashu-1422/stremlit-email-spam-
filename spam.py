import pandas as pd
import numpy as np
import re
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
data = pd.read_csv('spam.csv')  # Replace with your dataset path

# Text cleaning function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    return text

# Drop rows with missing data
data = data.dropna(subset=['Message'])

# Apply preprocessing
data['processed_text'] = data['Message'].apply(preprocess_text)

# Vectorize text using TF-IDF
tfidf = TfidfVectorizer(max_features=5000)  # Limit to 5000 features
X = tfidf.fit_transform(data['processed_text']).toarray()

# Define target variable
y = data['Category']  # Replace 'Category' with your target column name

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LogisticRegression()
lr.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

ada = AdaBoostClassifier()
ada.fit(X_train, y_train)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Save the Logistic Regression model and vectorizer
pickle.dump(lr, open('spam_model.pkl', 'wb'))
pickle.dump(tfidf, open('tfidf_vectorizer.pkl', 'wb'))

# Streamlit app
st.title("Spam Email Detector")

# User input
email_input = st.text_area("Enter the email content:")

if st.button("Classify"):
    if email_input.strip():
        # Load the model and vectorizer
        model = pickle.load(open('spam_model.pkl', 'rb'))
        vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

        # Preprocess and predict
        processed_email = vectorizer.transform([preprocess_text(email_input)])
        prediction = model.predict(processed_email)
        result = "Spam" if prediction[0] == "spam" else "Not Spam"
        st.success(f"Prediction: {result}")
    else:
        st.error("Please enter email content to classify.")
