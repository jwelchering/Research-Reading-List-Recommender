import requests


BASE_URL = "https://api.dnb.de/opac.htm"
DEFAULT_PARAMS = {
    "method": "search",
    "format": "json",
    "size": 20
}


def fetch_dnb_documents(query: str, size: int = 20):
    """
    Fetch documents from DNB API and return a normalized list of dicts.

    Each document will ALWAYS contain:
        - title
        - abstract
        - text  (combined field for NLP)

    This guarantees compatibility with vectorizers / clustering pipelines.
    """

    params = DEFAULT_PARAMS.copy()
    params["query"] = query
    params["size"] = size

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"[DNB API ERROR] {e}")
        return []

    # Defensive parsing (API structure may vary)
    results = data.get("records", [])

    documents = []

    for item in results:
        try:
            # --- Extract fields safely ---
            title = ""
            abstract = ""

            # Title extraction (varies depending on API structure)
            if "title" in item:
                title = item.get("title", "")
            elif "titles" in item and isinstance(item["titles"], list):
                title = item["titles"][0]

            # Abstract extraction
            if "abstract" in item:
                abstract = item.get("abstract", "")
            elif "descriptions" in item:
                desc = item.get("descriptions", [])
                if isinstance(desc, list) and len(desc) > 0:
                    abstract = desc[0]

            # Normalize to string
            title = str(title).strip()
            abstract = str(abstract).strip()

            # --- Build TEXT FIELD (critical for NLP) ---
            text = f"{title} {abstract}".strip()

            # Skip empty entries
            if not text:
                continue

            documents.append({
                "title": title,
                "abstract": abstract,
                "text": text
            })

        except Exception as e:
            # Skip malformed records but continue processing
            print(f"[PARSE ERROR] {e}")
            continue

    return documents
