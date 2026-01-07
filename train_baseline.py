import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import json
import re
import string

# --- CONFIGURATION ---
DATA_PATH = 'data/train.csv'  
MODEL_DIR = 'models'

def clean_text(text):
    text = str(text).lower()  
    text = re.sub(r'https?://\S+|www\.\S+', '', text) 
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text) 
    text = re.sub(r'\n', '', text) 
    text = re.sub(r'\w*\d\w*', '', text) 
    return text.strip()

def main():
    print("Starting Baseline Training Pipeline...")

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"Error: File '{DATA_PATH}' not found.")
        print("Please download 'train.csv' from Kaggle and place it in the 'data/' folder.")
        return

    print("Loading data...")
    try:
        data = pd.read_csv(DATA_PATH)
        data = data.dropna(subset=['comment_text'])
        print(f"Data loaded! Shape: {data.shape}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    print("Cleaning text data...")
    data['comment_text'] = data['comment_text'].apply(clean_text)
    # 2. Preprocessing
    print("Preprocessing data...")
    # Drop rows with missing text
    data = data.dropna(subset=['comment_text'])
    
    X = data['comment_text']
    y = data['toxic']  

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Vectorization (TF-IDF)
    print("Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Model Training
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    # 5. Evaluation
    print("Evaluating model...")

    y_pred = model.predict(X_test_vec)
    y_prob = model.predict_proba(X_test_vec)[:, 1]

    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)

    cm = confusion_matrix(y_test, y_pred)

    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    metrics = {
    "model": "Logistic Regression (TF-IDF)",
    "f1": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "confusion_matrix": cm.tolist()
}

    with open(os.path.join(MODEL_DIR, "metrics_baseline.json"), "w") as f:
        json.dump(metrics, f, indent=4)


    # 6. Save Model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Saving artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, 'model_logreg.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'vectorizer.pkl'))
    print(f"Baseline model saved to '{MODEL_DIR}/' successfully!")

if __name__ == "__main__":
    main()