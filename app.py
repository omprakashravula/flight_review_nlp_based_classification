import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
import re
import os
import warnings

# Hide TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Hide Scikit-Learn InconsistentVersionWarning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------

st.set_page_config(
    page_title="Twitter Airline Sentiment Analysis",
    page_icon="✈️",
    layout="centered"
)

# ---------------------------------------------------
# Load Files
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_model.keras")


@st.cache_resource
def load_scaler():
    with open("scaler.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_encoder():
    with open("label_encoder.pkl", "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_embeddings():
    with open("embedding_lookup.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
scaler = load_scaler()
encoder = load_encoder()
embedding_lookup = load_embeddings()

# ---------------------------------------------------
# Text Cleaning
# ---------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", " ", text)

    text = re.sub(r"@\w+", " @ ", text)

    text = re.sub(r"[^a-z@ ]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text.split()

# ---------------------------------------------------
# Sentence Embedding
# ---------------------------------------------------

def sentence_vector(sentence):

    words = clean_text(sentence)

    vectors = []

    for word in words:

        if word in embedding_lookup:
            vectors.append(embedding_lookup[word])

    if len(vectors) == 0:
        return np.zeros(200)

    return np.mean(vectors, axis=0)

# ---------------------------------------------------
# UI

# 1. Setup Sidebar Input Configuration
with st.sidebar:
    st.header("✈️ Control Panel")
    st.write("Submit target airline text metrics below.")
    
    tweet = st.text_area(
        "Tweet Text Content:", 
        height=200, 
        placeholder="Type airline feedback here..."
    )
    predict_button = st.button("Evaluate Sentiment", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.caption("v1.0.0 | Powered by TensorFlow & Word2Vec")

# 2. Setup Main Panel Display
st.title("📊 Airline Sentiment Analytics Desk")
st.write("This application monitors customer feedback channels using machine learning.")

if predict_button:
    if tweet.strip() == "":
        st.error("Error: Please look at the sidebar panel and supply input text.")
    else:
        vector = sentence_vector(tweet)
        vector = scaler.transform(vector.reshape(1, -1))
        prediction = model.predict(vector, verbose=0)
        pred_index = np.argmax(prediction)
        sentiment = encoder.inverse_transform([pred_index])[0]
        confidence = prediction[0][pred_index]

        # Display clean workspace results
        st.subheader("Detected Intent & Emotion")
        
        # Large banner callout
        if sentiment == "positive":
            st.markdown(f"### 🎉 Highly Positive Reaction Detected")
        elif sentiment == "neutral":
            st.markdown(f"### 🤝 Objective/Neutral Statement Detected")
        else:
            st.markdown(f"### ⚠️ Negative Review Action Required")

        # Layout structured statistics
        stat1, stat2 = st.columns(2)
        stat1.metric(label="Predicted Class", value=sentiment.capitalize())
        stat2.metric(label="Certainty Engine Score", value=f"{confidence*100:.2f}%")

        st.markdown("---")
        st.subheader("Probability Distributions")
        
        # Clean inline data alignment
        for label, prob in zip(encoder.classes_, prediction[0]):
            col_lbl, col_bar = st.columns([1, 4])
            col_lbl.write(f"**{label.capitalize()}**")
            col_bar.progress(float(prob), text=f"{prob*100:.1f}%")

else:
    # Beautiful landing placeholder info
    st.info("⬅️ Provide text inputs inside the sidebar utility menu to initiate processing layers.")