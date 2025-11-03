import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from textstat import flesch_reading_ease
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ---------------------------
# Load trained model
# ---------------------------
model = joblib.load("models/quality_model.pkl")

# ---------------------------
# Helper: extract content from URL
# ---------------------------
def extract_content(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string if soup.title else "No Title"
        body = " ".join([p.get_text() for p in soup.find_all("p")])
        return {"title": title, "body": body}
    except Exception as e:
        return None

# ---------------------------
# Helper: extract features
# ---------------------------
def extract_features(text):
    text = text.lower().strip()
    word_count = len(text.split())
    sentence_count = text.count(".") + text.count("!") + text.count("?")
    readability = flesch_reading_ease(text)

    vectorizer = TfidfVectorizer(max_features=5, stop_words="english")
    vectorizer.fit([text])
    keywords = vectorizer.get_feature_names_out()

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "flesch_reading_ease": readability,
        "top_keywords": list(keywords)
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="SEO Content Quality & Duplicate Detector", layout="centered")

st.title(" SEO Content Quality & Duplicate Detector")
st.write("Analyze webpage SEO content quality and detect duplicates using ML model.")

url = st.text_input("Enter a webpage URL:")

if st.button("Analyze"):
    with st.spinner("Fetching and analyzing content..."):
        content = extract_content(url)
        if not content:
            st.error("Failed to fetch or parse content. Please check the URL.")
        else:
            features = extract_features(content["body"])
            X = np.array([[features["word_count"], features["sentence_count"], features["flesch_reading_ease"]]])
            prediction = model.predict(X)[0]

            st.subheader(f" Quality Prediction: **{prediction}**")
            st.write(f"**Title:** {content['title']}")
            st.write(f"**Word Count:** {features['word_count']}")
            st.write(f"**Readability Score:** {features['flesch_reading_ease']:.2f}")
            st.write(f"**Top Keywords:** {', '.join(features['top_keywords'])}")


