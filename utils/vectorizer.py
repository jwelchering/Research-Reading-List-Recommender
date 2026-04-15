from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(texts):
    """
    Creates TF-IDF matrix from text list.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)

    return vectorizer, matrix
