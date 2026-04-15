import streamlit as st
import pandas as pd
from collections import Counter

from utils.dnb_api import fetch_dnb_documents

st.title("Research Analytics Dashboard")

query = st.text_input("DNB analysis term")

if query:

    df = pd.DataFrame(fetch_dnb_documents(query))

    st.metric("Documents retrieved", len(df))

    words = " ".join(df["text"]).lower().split()
    common = Counter(words).most_common(10)

    st.subheader("Top Keywords")
    st.write(common)

    df["length"] = df["text"].apply(len)

    st.subheader("Document Length Distribution")
    st.bar_chart(df["length"])
