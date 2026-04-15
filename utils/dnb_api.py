import requests

BASE_URL = "https://api.dnb.de/opac.htm"

DEFAULT_PARAMS = {
    "method": "search",
    "format": "json",
    "size": 20
}


def fetch_dnb_documents(query: str, size: int = 20):
    params = DEFAULT_PARAMS.copy()
    params["query"] = query
    params["size"] = size

    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()

        if "json" not in response.headers.get("Content-Type", ""):
            return []

        data = response.json()

    except Exception as e:
        print(f"[DNB API ERROR] {e}")
        return []

    results = data.get("records", [])
    documents = []

    for item in results:
        try:
            title = ""
            abstract = ""

            if isinstance(item.get("title"), str):
                title = item["title"]
            elif isinstance(item.get("titles"), list) and item["titles"]:
                title = item["titles"][0]

            if isinstance(item.get("abstract"), str):
                abstract = item["abstract"]
            elif isinstance(item.get("descriptions"), list) and item["descriptions"]:
                abstract = item["descriptions"][0]

            text = f"{title} {abstract}".strip()

            if not text:
                continue

            documents.append({
                "title": title,
                "abstract": abstract,
                "text": text
            })

        except Exception as e:
            print(f"[PARSE ERROR] {e}")
            continue

    return documents
