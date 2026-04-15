import streamlit as st
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from utils.dnb_api import fetch_dnb_documents
from utils.preprocessing import build_vectorizer

st.title("Topic Clustering (DNB Data)")

query = st.text_input("Cluster search term")

if query:

    df = pd.DataFrame(fetch_dnb_documents(query))

    vectorizer, matrix = build_vectorizer(df["text"])

    k = st.slider("Number of clusters", 2, 6, 3)

    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(matrix)

    df["cluster"] = labels

    pca = PCA(n_components=2)
    coords = pca.fit_transform(matrix.toarray())

    df["x"] = coords[:, 0]
    df["y"] = coords[:, 1]

    st.scatter_chart(df, x="x", y="y", color="cluster")

    st.dataframe(df[["title", "cluster"]])
