# 🛡️ Spam Sentinel - ML Email Classifier

An AI-powered email spam detection system that classifies emails as **Spam** or **Not Spam** using Machine Learning.

---

## 🚀 Features

- 📧 Real-time email classification
- 🧠 Machine Learning model (TF-IDF + Logistic Regression)
- ⚡ Fast API using Flask
- 🎨 Modern UI inspired by AI Studio
- 📊 Train/Test dataset handling
- 🔍 Live prediction system

---

## 🧠 Tech Stack

- Python
- scikit-learn
- Flask
- HTML, CSS, JavaScript
- Vite (AI Studio frontend setup)

---

## ⚙️ How It Works

1. Dataset is preprocessed and cleaned  
2. TF-IDF converts text into numerical features  
3. Logistic Regression model is trained  
4. Flask API serves predictions  
5. Frontend sends input → receives classification  

---

## 🧪 Sample Inputs

| Email | Output |
|------|--------|
| "Win a free iPhone now!!!" | SPAM |
| "Meeting tomorrow at 10 AM" | NOT SPAM |

---

## 🖼️ Working Demo
<img width="1010" height="718" alt="Screenshot 2026-03-29 at 2 41 15 PM" src="https://github.com/user-attachments/assets/41e94d5e-8ec6-43e3-9c87-95867360ec88" />
<img width="1470" height="956" alt="Screenshot 2026-03-29 at 6 33 39 PM" src="https://github.com/user-attachments/assets/f51dcb47-732d-4920-bf6f-9f25bd36f1bb" />
<img width="1470" height="956" alt="Screenshot 2026-03-29 at 6 33 50 PM" src="https://github.com/user-attachments/assets/bf6d62eb-adf0-47f8-af92-e02a8553a572" />


---

## 📊 Model Performance

<img width="1640" height="714" alt="image" src="https://github.com/user-attachments/assets/b8183294-327e-4c88-a6dc-4ecbeda8506b" />


---

## 📈 Model Details

- Algorithm: Logistic Regression  
- Feature Extraction: TF-IDF  
- Train/Test Split: 80/20  
- Metrics:
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  

---

## ▶️ How to Run

### 1. Install dependencies
pip install -r requirements.txt


### 2. Train the model

python train.py


### 3. Run backend

python app.py


### 4. Open frontend

open index.html

---
## 🧠 Future Improvements

- Deep Learning models (LSTM / BERT)  
- Deploy online (Render / Vercel)  
- Add confidence score  
- Improve UI/UX  
- Gmail API integration  
