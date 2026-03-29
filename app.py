from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import string

app = Flask(__name__)
CORS(app)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    email = data.get("email", "")

    email = clean_text(email)
    vec = vectorizer.transform([email])
    pred = model.predict(vec)[0]

    return jsonify({
        "result": "SPAM" if pred == 1 else "NOT SPAM"
    })

if __name__ == "__main__":
    app.run(debug=True)