import streamlit as st
import pandas as pd

from utils.dnb_api import fetch_dnb_documents
from utils.preprocessing import build_vectorizer
from utils.model import compute_similarity
from utils.explainability import explain

st.title("Live DNB Literature Recommender")

query = st.text_input("Search the German National Library")

top_k = st.slider("Number of recommendations", 3, 10, 5)

if query:

    docs = fetch_dnb_documents(query)

    if not docs:
        st.warning("No data found.")
        st.stop()

    df = pd.DataFrame(docs)

    vectorizer, matrix = build_vectorizer(df["text"])

    results = compute_similarity(query, vectorizer, matrix, df)

    st.subheader("Recommendations")

    for _, row in results.head(top_k).iterrows():

        st.markdown(f"### {row['title']}")
        st.write(row["text"])

        st.info(explain(query, row["text"]))

        st.divider()
