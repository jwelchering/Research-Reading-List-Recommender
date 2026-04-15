from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(texts):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix
