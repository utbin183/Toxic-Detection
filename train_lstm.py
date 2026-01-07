import pandas as pd
import numpy as np
import os
import pickle
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GlobalMaxPool1D, Bidirectional, SpatialDropout1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
import json

# --- CONFIGURATION ---
DATA_PATH = 'data/train.csv'
MODEL_DIR = 'models'
MAX_WORDS = 20000       
MAX_LEN = 150         
EMBEDDING_DIM = 128     

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def main():
    print(">>> Starting Deep Learning (LSTM) Pipeline...")

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: '{DATA_PATH}' not found.")
        return

    # 1. Load Data
    print("Loading data...")
    data = pd.read_csv(DATA_PATH)
    data = data.dropna(subset=['comment_text'])
    
    print("Cleaning data...")
    data['comment_text'] = data['comment_text'].apply(clean_text)
    
    X = data['comment_text'].astype(str)
    y = data['toxic'].values

    # 2. Tokenization
    print("Tokenizing text...")
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X)
    sequences = tokenizer.texts_to_sequences(X)
    padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

    print("Computing class weights...")
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    dict_weights = dict(enumerate(class_weights))
    print(f"Class Weights: {dict_weights}")

    # 3. Build Model 
    print("Building LSTM model...")
    model = Sequential([
        Embedding(input_dim=MAX_WORDS, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
        SpatialDropout1D(0.2), 
        Bidirectional(LSTM(64, return_sequences=True)), 
        GlobalMaxPool1D(),
        Dense(64, activation='relu'),
        Dropout(0.4), 
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # 4. Train Model
    print("Training model...")
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    
    history = model.fit(
        X_train, y_train,
        epochs=7,
        batch_size=64,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        class_weight=dict_weights
    )

    # 5. Evaluate
    print("\nEvaluating...")
    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob > 0.5).astype(int)

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
        "model": "LSTM",
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm.tolist()
    }

    with open(os.path.join(MODEL_DIR, "metrics_lstm.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # 6. Save
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print("Saving artifacts...")
    model.save(os.path.join(MODEL_DIR, 'model_lstm.h5'))
    with open(os.path.join(MODEL_DIR, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"LSTM model saved successfully!")

if __name__ == "__main__":
    main()