def explain(query, document):

    q_terms = set(query.lower().split())
    d_terms = set(document.lower().split())

    overlap = q_terms.intersection(d_terms)

    if not overlap:
        return "Relevance is based on semantic vector similarity (TF-IDF + Cosine Similarity)."

    return f"Shared key terms: {', '.join(list(overlap)[:8])}"
