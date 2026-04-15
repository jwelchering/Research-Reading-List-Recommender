from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(query, vectorizer, tfidf_matrix, df):

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    df = df.copy()
    df["score"] = scores

    return df.sort_values("score", ascending=False)
