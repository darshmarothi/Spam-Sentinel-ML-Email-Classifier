import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv(
    "/Users/darshmarothi/Documents/Uni/text mining/spam-sentinel_-ml-email-classifier/dataset/combine.csv",
    encoding="latin-1",
    on_bad_lines="skip"
)

print("Columns:", df.columns)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['text'] = df['text'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'],
    df['spam'],   # ✅ FIXED HERE
    test_size=0.2,
    random_state=42,
    stratify=df['spam']
)

vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

def predict_email(email):
    email = clean_text(email)
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]
    return "SPAM" if pred == 1 else "NOT SPAM"

print(predict_email("Win a free iPhone now"))