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
        r = requests.get(BASE_URL, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[DNB ERROR] {e}")
        return []

    results = data.get("records", [])
    docs = []

    for item in results:
        title = ""
        abstract = ""

        try:
            title = item.get("title", "") or ""
            if isinstance(item.get("titles"), list):
                title = item["titles"][0] if item["titles"] else ""

            if isinstance(item.get("abstract"), str):
                abstract = item["abstract"]

            elif isinstance(item.get("descriptions"), list):
                abstract = item["descriptions"][0] if item["descriptions"] else ""

            text = f"{title} {abstract}".strip()

            if not text:
                continue

            docs.append({
                "title": title,
                "abstract": abstract,
                "text": text
            })

        except Exception as e:
            print(f"[PARSE ERROR] {e}")
            continue

    return docs
