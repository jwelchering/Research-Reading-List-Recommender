import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from sklearn.metrics.pairwise import cosine_similarity

from utils.dnb_api import fetch_dnb_documents
from utils.preprocessing import build_vectorizer


st.title("📄 PDF-based Literature Recommender")

# --- Upload ---
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = []

    for page in reader.pages:
        try:
            content = page.extract_text()
            if content:
                text.append(content)
        except:
            continue

    return " ".join(text)


if uploaded_file:
    st.info("Extracting text from PDF...")

    pdf_text = extract_text_from_pdf(uploaded_file)

    if not pdf_text.strip():
        st.error("No text could be extracted from PDF.")
        st.stop()

    st.success("Text extracted!")

    # --- Query DNB ---
    st.info("Fetching DNB documents...")
    docs = fetch_dnb_documents("philosophy")  # optional: dynamic later

    df = pd.DataFrame(docs)

    if df.empty or "text" not in df.columns:
        st.error("No usable DNB data found.")
        st.stop()

    # --- Combine corpus ---
    corpus = df["text"].tolist() + [pdf_text]

    # --- Vectorization ---
    vectorizer, matrix = build_vectorizer(corpus)

    # Last vector = PDF
    pdf_vector = matrix[-1]
    doc_vectors = matrix[:-1]

    # --- Similarity ---
    similarities = cosine_similarity(pdf_vector, doc_vectors)[0]

    df["similarity"] = similarities

    # --- Top results ---
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    results = df.sort_values(by="similarity", ascending=False).head(top_n)

    st.subheader("📚 Recommended Literature")

    for _, row in results.iterrows():
        st.markdown(f"**{row['title']}**")
        st.write(f"Similarity: {row['similarity']:.3f}")
        st.write(row["abstract"])
        st.markdown("---")
