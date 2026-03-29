import pandas as pd
import numpy as np
import string
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords
nltk.download('stopwords', quiet=True)

def preprocess_text(text):
    """
    Cleans the input text: lowercase, remove punctuation, remove stopwords, and stemming.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 3. Tokenization and Stopword removal
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    
    # 4. Stemming
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]
    
    return " ".join(words)

def run_pipeline(data_path='emails.csv'):
    print("--- Starting Spam Classification Pipeline ---")
    
    # 1. Load Data
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please ensure the dataset exists.")
        return
    
    df = pd.read_csv(data_path)
    print(f"Dataset loaded. Shape: {df.shape}")
    
    # 2. Data Preprocessing
    # Remove null values
    df.dropna(subset=['text', 'label'], inplace=True)
    
    # Clean text
    print("Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # 3. Train-Test Split (80% training, 20% testing)
    X = df['cleaned_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Save split datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print("Training and testing datasets saved as train.csv and test.csv.")
    
    # 4. Feature Engineering (TF-IDF Vectorization)
    tfidf = TfidfVectorizer(max_features=3000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # 5. Model Training (Logistic Regression)
    print("Training Logistic Regression model...")
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)
    
    # 6. Evaluation
    y_pred = model.predict(X_test_tfidf)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    
    # 7. Model Saving
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    print("\nModel and vectorizer saved as model.pkl and vectorizer.pkl.")

def predict_spam(email_text):
    """
    Predicts whether a new email is SPAM or NOT SPAM.
    """
    # Load model and vectorizer
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('vectorizer.pkl', 'rb') as f:
            tfidf = pickle.load(f)
    except FileNotFoundError:
        return "Error: Model files not found. Please run the training pipeline first."
    
    # Preprocess
    cleaned = preprocess_text(email_text)
    
    # Vectorize
    vectorized = tfidf.transform([cleaned])
    
    # Predict
    prediction = model.predict(vectorized)[0]
    
    return "SPAM" if prediction == 1 else "NOT SPAM"

if __name__ == "__main__":
    # Run the full pipeline
    run_pipeline()
    
    # Example Prediction
    print("\n--- Testing Prediction Function ---")
    test_email = "Congratulations! You've won a free cruise to the Bahamas. Click here to claim your prize."
    result = predict_spam(test_email)
    print(f"Email: {test_email}")
    print(f"Prediction: {result}")
    
    test_email_2 = "Hi John, can we schedule the meeting for tomorrow at 2 PM?"
    result_2 = predict_spam(test_email_2)
    print(f"Email: {test_email_2}")
    print(f"Prediction: {result_2}")
