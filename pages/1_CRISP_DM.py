import streamlit as st

st.title("CRISP-DM Framework")

st.markdown("""
## 1. Business Understanding
Improve academic literature discovery and structuring.

## 2. Data Understanding
Live data from the German National Library (SRU API).

## 3. Data Preparation
XML parsing → cleaning → feature extraction.

## 4. Modeling
TF-IDF + cosine similarity + clustering.

## 5. Evaluation
Relevance scoring + cluster coherence.

## 6. Deployment
Interactive Streamlit web application.
""")
