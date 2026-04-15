import requests
import xml.etree.ElementTree as ET
import streamlit as st

DNB_URL = "https://services.dnb.de/sru/dnb"


@st.cache_data(ttl=3600)
def fetch_dnb_documents(query, max_records=25):

    params = {
        "version": "1.1",
        "operation": "searchRetrieve",
        "query": f'any "{query}"',
        "maximumRecords": max_records
    }

    response = requests.get(DNB_URL, params=params, timeout=10)

    if response.status_code != 200:
        return []

    root = ET.fromstring(response.content)

    ns = {"srw": "http://www.loc.gov/zing/srw/"}

    results = []

    for record in root.findall(".//srw:record", ns):

        data = record.find(".//srw:recordData", ns)
        if data is None:
            continue

        xml_str = ET.tostring(data, encoding="unicode", method="xml")

        title = extract(xml_str, "dc:title")
        creator = extract(xml_str, "dc:creator")
        date = extract(xml_str, "dc:date")

    if title == "unknown":
        continue

text = f"{title} {creator} {date}".strip()

if len(text) < 5:
    continue

        results.append({
            "title": title,
            "text": text
        })

    return results


def extract(xml, tag):
    start = xml.find(tag)
    if start == -1:
        return "unknown"
    return xml[start:start+120].replace("<", "").replace(">", "")
