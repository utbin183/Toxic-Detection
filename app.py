import streamlit as st
import joblib
import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import seaborn as sns 
import matplotlib.pyplot as plt
import re 
import string
import pandas as pd
import time
import random

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Toxic Comment Detection", page_icon="🛡️", layout="wide")
MAX_LEN = 100 

def clean_text(text):
    # Basic cleanup: lowercase, remove links, html tags, punctuation, etc.
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

def simulate_fetching(platform, num_samples, query=""):
    data = []
    
    # 1. NO KEYWORD -> JUST GRAB RANDOM STUFF
    if not query.strip():
        random_comments = [
            "This video is amazing!", "I hate you so much.", "Please subscribe to my channel.",
            "You are an idiot.", "Great content, keep it up!", "Spam spam spam",
            "Kill yourself", "Love this song <3", "What a loser", "Hello from the US",
            "Why is everyone so toxic here?", "Beautiful place!", "Shut up", "Can you make a tutorial?"
        ]
        source_pool = random_comments
    
    # 2. KEYWORD DETECTED -> GENERATE CONTEXTUAL COMMENTS
    else:
        q = query.strip()
        templates = [
            # POSITIVE / CLEAN 
            f"I really love {q}, it is the best!",
            f"Why are people talking about {q}?",
            f"Information about {q} is very helpful.",
            f"{q} changed my life, thank you.",
            f"Thinking about {q} makes me happy.",
            f"Honestly, {q} is a masterpiece.",
            f"I've been following {q} for years, never disappointed.",
            f"Can someone explain more about {q}? It looks interesting.",
            f"Wow, {q} is actually pretty underrated.",
            f"This is exactly what I needed to know about {q}.",
            f"Great content regarding {q}, keep it up!",
            f"I totally agree with your take on {q}.",
            f"{q} is amazing, ignore the haters.",
            f"Looking forward to seeing more updates on {q}.",
            f"Respect for talking about {q} openly.",

            # TOXIC / HATE / INSULT 
            f"{q} is garbage and useless.",
            f"Whoever likes {q} is an idiot.",
            f"Stop posting about {q}, nobody cares!",
            f"I will kill anyone who supports {q}.",
            f"{q} is a scam, do not trust it.",
            f"{q} makes me sick, delete this.",
            f"You are so stupid for believing in {q}.",
            f"Imagine actually spending money on {q}. Lmao.",
            f"{q} is absolute trash, don't waste your time.",
            f"People who support {q} deserve to die.",
            f"Shut up about {q}, you loser.",
            f"This is the worst take on {q} I have ever seen.",
            f"Only brainwashed sheep like {q}.",
            f"{q}? More like pure cancer.",
            f"I hope {q} fails miserably.",
            f"You look ugly when you talk about {q}.",
            f"Get a life instead of obsessing over {q}.",
            f"WTF is this {q} nonsense? Disgusting."
        ]
        source_pool = templates

    
    with st.spinner(f"Searching {platform} for: '{query if query else 'Random'}'..."):
        progress_bar = st.progress(0)
        for i in range(num_samples):
            time.sleep(0.05) 
            
            # Pick a random comment from our pool
            comment = random.choice(source_pool)
            
            data.append({
                "platform": platform, 
                "query": query if query else "Random",
                "comment": comment
            })
            progress_bar.progress((i + 1) / num_samples)
            
    return pd.DataFrame(data)

# Real fetching function (Requires API Key)
def fetch_youtube_comments(api_key, video_id, max_results=20):
    # If you aren't using this, it just returns an empty DF to prevent crashes.
    # Uncomment the import if you have the library installed.
    try:
        from googleapiclient.discovery import build
        youtube = build('youtube', 'v3', developerKey=api_key)
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=max_results,
            textFormat='plainText'
        ).execute()

        comments = []
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append({"platform": "YouTube", "query": video_id, "comment": comment})
        return pd.DataFrame(comments)
    except ImportError:
        st.warning("Library 'google-api-python-client' is missing.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"YouTube API Error: {str(e)}")
        return pd.DataFrame()

# --- LOAD MODELS ---
@st.cache_resource
def load_baseline_model():
    # Attempting to load the Logistic Regression model
    try:
        model = joblib.load('models/model_logreg.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        return model, vectorizer
    except Exception as e:
        return None, None

@st.cache_resource
def load_lstm_model():
    # Attempting to load the Deep Learning model
    try:
        model = load_model('models/model_lstm.h5')
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        return None, None

def load_metrics(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

# --- MAIN UI ---
st.title("🛡️ Toxic Comment Detection System")

# Creating Tabs
tab1, tab2 = st.tabs(["🔍 Single Detect", "📥 Batch Analysis"])

# --- TAB 1: DETECTION ---
with tab1:
    st.markdown("### Comment Classification Demo (Binary)")
    
    col_main, col_sidebar = st.columns([3, 1])
    
    with col_sidebar:
        st.header("⚙️ Model Config")
        model_choice = st.radio("Select Model:", ["Baseline (Logistic Regression)", "Deep Learning (LSTM)"])
        st.info("💡 **Baseline**: Fast, explainable.\n\n🚀 **Deep Learning**: Better context awareness.")

    with col_main:
        input_text = st.text_area("Enter comment to check:", height=100, placeholder="Ex: You are stupid...")

        if st.button("Detect", type="primary"):
            if not input_text.strip():
                st.warning("⚠️ Please enter some text!")
            else:
                clean_input = clean_text(input_text)
                prediction = 0
                confidence = 0.0
                
                # PREDICTION LOGIC
                if model_choice == "Baseline (Logistic Regression)":
                    model_base, vectorizer = load_baseline_model()
                    if model_base:
                        vec_input = vectorizer.transform([clean_input])
                        prediction = model_base.predict(vec_input)[0]
                        confidence = model_base.predict_proba(vec_input)[0][1]
                    else:
                        st.error("❌ Baseline model not found/trained!")
                        st.stop()
                        
                elif model_choice == "Deep Learning (LSTM)":
                    model_lstm, tokenizer = load_lstm_model()
                    if model_lstm:
                        seq = tokenizer.texts_to_sequences([clean_input])
                        padded = pad_sequences(seq, maxlen=MAX_LEN)
                        confidence = float(model_lstm.predict(padded, verbose=0)[0][0])
                        prediction = 1 if confidence > 0.5 else 0
                    else:
                        st.error("❌ LSTM model not found/trained!")
                        st.stop()

                # DISPLAY RESULTS
                st.divider()
                c1, c2 = st.columns([1, 2])
                with c1:
                    if prediction == 1:
                        st.error(f"🚨 **TOXIC**")
                    else:
                        st.success(f"✅ **CLEAN**")
                with c2:
                    st.metric("Confidence Score", f"{confidence*100:.2f}%")
                    st.progress(confidence)

                # SHOW METRICS
                with st.expander("📊 View Metrics & Confusion Matrix"):
                    metric_path = "models/metrics_baseline.json" if "Baseline" in model_choice else "models/metrics_lstm.json"
                    metrics = load_metrics(metric_path)
                    
                    if metrics:
                        st.write(f"**F1-Score:** {metrics['f1']:.4f} | **ROC-AUC:** {metrics['roc_auc']:.4f}")
                        cm = np.array(metrics['confusion_matrix'])
                        fig, ax = plt.subplots(figsize=(4, 3))
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Clean', 'Toxic'], yticklabels=['Clean', 'Toxic'])
                        st.pyplot(fig)
                    else:
                        st.warning("Metrics file not found.")

# --- TAB 2: DATA COLLECTION & BATCH ANALYSIS ---
with tab2:
    st.markdown("### 📥 Data Collection & Analysis")
    st.write("Social Listening Simulator: Crawl comments by keyword and scan for toxicity.")

    # 1. Setup Collection
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        platform = st.selectbox("Platform:", ["Facebook", "TikTok", "YouTube"])
    with col2:
        search_query = st.text_input("🔍 Keyword / Hashtag (Empty = Random):", placeholder="Ex: Ronaldo, Bitcoin, Scandal...")
    with col3:
        num_samples = st.slider("Sample Count:", 10, 100, 20)

    # Initialize Session State
    if 'crawled_data' not in st.session_state:
        st.session_state.crawled_data = pd.DataFrame()

    # 2. Run Button
    if st.button("🚀 Start Crawling"):
        # Call simulation function
        df_result = simulate_fetching(platform, num_samples, search_query)
        st.session_state.crawled_data = df_result
        st.success(f"✅ Found {len(df_result)} related comments!")

    # 3. Display & Analyze
    if not st.session_state.crawled_data.empty:
        st.divider()
        st.subheader("1️⃣ Raw Data")
        st.dataframe(
            st.session_state.crawled_data,
            use_container_width=True, 
            column_config={
                "comment": st.column_config.TextColumn(
                    "Comment Content", 
                    width="large" 
                ),
                "label": st.column_config.NumberColumn(
                    "Label",
                    format="%d"
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                )
            }
        )

        st.divider()
        st.subheader("2️⃣ AI Toxicity Analysis")
        
        batch_model = st.selectbox("Select Model:", ["Baseline (LogReg)", "Deep Learning (LSTM)"], key="batch_mode")
        
        if st.button("⚡ Run AI Analysis"):
            data_to_analyze = st.session_state.crawled_data.copy()
            
            # --- MODEL INFERENCE ---
            if "Baseline" in batch_model:
                model, vectorizer = load_baseline_model()
                if model:
                    with st.spinner("AI is reading comments..."):
                        data_to_analyze['clean_text'] = data_to_analyze['comment'].apply(clean_text)
                        vecs = vectorizer.transform(data_to_analyze['clean_text'])
                        data_to_analyze['label'] = model.predict(vecs)
                        data_to_analyze['confidence'] = model.predict_proba(vecs)[:, 1]
                else:
                    st.error("Error: Baseline model missing.")

            elif "LSTM" in batch_model:
                model, tokenizer = load_lstm_model()
                if model:
                    with st.spinner("AI is reading comments..."):
                        data_to_analyze['clean_text'] = data_to_analyze['comment'].apply(clean_text)
                        seq = tokenizer.texts_to_sequences(data_to_analyze['clean_text'])
                        padded = pad_sequences(seq, maxlen=MAX_LEN)
                        preds = model.predict(padded, verbose=0)
                        data_to_analyze['confidence'] = preds
                        data_to_analyze['label'] = (preds > 0.5).astype(int)
                else:
                    st.error("Error: LSTM model missing.")

            # --- REPORT ---
            if 'label' in data_to_analyze.columns:
                num_toxic = data_to_analyze[data_to_analyze['label'] == 1].shape[0]
                num_clean = data_to_analyze[data_to_analyze['label'] == 0].shape[0]

                c1, c2 = st.columns(2)
                c1.metric("🔴 Toxic Comments", num_toxic)
                c2.metric("🟢 Clean Comments", num_clean)

                st.write("📋 **Detailed Results:**")
                def highlight_toxic(row):
                    return ['background-color: #ffcccc' if row.label == 1 else '' for _ in row]

                st.dataframe(
                    data_to_analyze[['comment', 'label', 'confidence']]
                    .style.apply(highlight_toxic, axis=1)
                    .format({'confidence': "{:.2%}"})
                )
                
                # Chart
                fig, ax = plt.subplots(figsize=(5, 2))
                ax.barh(['Toxic', 'Clean'], [num_toxic, num_clean], color=['red', 'green'])
                st.pyplot(fig)