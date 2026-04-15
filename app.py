
import streamlit as st
import numpy as np
import pandas as pd
import requests
import xml.etree.ElementTree as ET
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import plotly.express as px
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="CRISP-DM Research Assistant", layout="wide", page_icon="📚")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.kpi-box {background:#f0f4ff;border-radius:8px;padding:12px 16px;text-align:center;margin-bottom:8px;}
.kpi-val {font-size:2rem;font-weight:700;color:#1a3c8f;}
.kpi-lbl {font-size:.85rem;color:#555;}
.phase-badge {display:inline-block;background:#1a3c8f;color:white;border-radius:5px;
              padding:3px 10px;font-size:.8rem;margin-bottom:8px;}
</style>""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in [("documents",[]),("titles",[]),("sources",[]),("years",[]),
             ("biz_goal",""),("biz_question",""),("feedback",{})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── DNB SRU API ───────────────────────────────────────────────────────────────
DNB_SRU = "https://services.dnb.de/sru/dnb"
NS = {"srw":"http://www.loc.gov/zing/srw/",
      "dc":"http://purl.org/dc/elements/1.1/",
      "oai_dc":"http://www.openarchives.org/OAI/2.0/oai_dc/"}

def query_dnb(term, max_records=10):
    params = {"operation":"searchRetrieve","version":"1.1",
              "query":f'tit="{term}" or abs="{term}"',
              "maximumRecords":max_records,"recordSchema":"oai_dc"}
    try:
        r = requests.get(DNB_SRU, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        st.error(f"DNB request failed: {e}")
        return []
    root = ET.fromstring(r.content)
    results = []
    for rec in root.findall(".//srw:record/srw:recordData/oai_dc:dc", NS):
        title   = rec.findtext("dc:title",       default="(no title)", namespaces=NS)
        desc    = rec.findtext("dc:description", default="",           namespaces=NS)
        year    = rec.findtext("dc:date",        default="",           namespaces=NS)
        subject = rec.findtext("dc:subject",     default="",           namespaces=NS)
        text    = " ".join(filter(None,[title, subject, desc]))
        if text.strip():
            results.append({"title":title,"text":text,"year":year})
    return results

def preprocess(text, lang="english"):
    text = re.sub(r"[^\w\s]"," ", text.lower())
    text = re.sub(r"\s+"," ", text).strip()
    try:
        sw = set(stopwords.words(lang))
    except:
        sw = set()
    stemmer = SnowballStemmer(lang if lang in ("german","english") else "english")
    return " ".join(stemmer.stem(w) for w in text.split() if w not in sw and len(w)>2)

def kpi(col, val, label):
    col.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)

def phase(label):
    st.markdown(f'<div class="phase-badge">CRISP-DM Phase: {label}</div>', unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📚 CRISP-DM Research Intelligence System")
st.caption("Research Reading List Recommender · TF-IDF · Cosine Similarity · Clustering · DNB SRU API")

# ── CRISP-DM Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1️⃣ Business Understanding",
    "2️⃣ Data Collection",
    "3️⃣ Data Preparation",
    "4️⃣ Modeling",
    "5️⃣ Evaluation",
    "6️⃣ Deployment & Insights",
])

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — BUSINESS UNDERSTANDING
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    phase("1 — Business Understanding")
    st.header("Define Your Research Goal")
    st.markdown("""
In CRISP-DM, the first phase establishes **what problem we are solving** and **what success looks like**.
For a research recommender, this means defining the research domain, the central question, and
the criteria for a useful recommendation.
    """)

    st.session_state.biz_goal = st.text_area(
        "📌 Research domain / subject area",
        value=st.session_state.biz_goal,
        placeholder="e.g. AI Ethics in Healthcare, Digital Transformation, Theology and Technology")

    st.session_state.biz_question = st.text_area(
        "❓ Central research question",
        value=st.session_state.biz_question,
        placeholder="e.g. How do bias and fairness considerations shape AI deployment in public institutions?")

    st.markdown("### Success Criteria")
    st.markdown("""
| Criterion | Target |
|---|---|
| Recommendation relevance | Cosine similarity ≥ 0.20 for top results |
| Cluster coherence | Distinct top-keywords per cluster, no dominant overlap |
| Data coverage | At least 8 documents from ≥ 2 sources |
| Transparency | Model logic visible and explainable |
    """)

    if st.session_state.biz_goal and st.session_state.biz_question:
        st.success(f"✅ Goal set: **{st.session_state.biz_goal[:80]}**")
        st.info(f"Research question: *{st.session_state.biz_question[:120]}*")
    else:
        st.warning("Fill in both fields above before proceeding to Data Collection.")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — DATA COLLECTION
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    phase("2 — Data Understanding & Collection")
    st.header("Collect Research Documents")

    col_a, col_b, col_c = st.columns(3)

    # Manual
    with col_a:
        st.subheader("✍️ Manual Entry")
        with st.form("manual_form"):
            t = st.text_input("Title")
            d = st.text_area("Text (abstract / note / excerpt)", height=130)
            y = st.text_input("Year (optional)")
            if st.form_submit_button("Add Document"):
                if d.strip():
                    st.session_state.documents.append(d.strip())
                    st.session_state.titles.append(t.strip() or f"Doc {len(st.session_state.documents)}")
                    st.session_state.sources.append("Manual")
                    st.session_state.years.append(y.strip())
                    st.success("Added!")
                else:
                    st.warning("Please enter some text.")

    # DNB
    with col_b:
        st.subheader("🏛️ Deutsche Nationalbibliothek")
        dnb_q = st.text_input("Search term", placeholder="e.g. AI ethics, digitalization",
                               key="dnb_search")
        dnb_n = st.slider("Max. results", 3, 20, 8, key="dnb_n")
        if st.button("🔍 Search DNB Catalogue"):
            with st.spinner("Querying DNB SRU API…"):
                hits = query_dnb(dnb_q, dnb_n)
            if hits:
                added = sum(1 for h in hits
                            if h["text"] not in st.session_state.documents
                            and (st.session_state.documents.append(h["text"]) or True)
                            and (st.session_state.titles.append(h["title"][:60]) or True)
                            and (st.session_state.sources.append("DNB") or True)
                            and (st.session_state.years.append(h["year"]) or True))
                st.success(f"{added} new entries added from DNB.")
            else:
                st.warning("No results. Try a different term.")
        st.caption("Source: Deutsche Nationalbibliothek SRU interface · Metadata: CC0 1.0")

    # CSV
    with col_c:
        st.subheader("📂 CSV Upload")
        csv_file = st.file_uploader("CSV with columns: title, text", type=["csv"])
        if csv_file:
            df_up = pd.read_csv(csv_file)
            if "text" in df_up.columns:
                for _, row in df_up.iterrows():
                    t2 = str(row.get("title","")).strip() or f"Doc {len(st.session_state.documents)+1}"
                    d2 = str(row["text"]).strip()
                    y2 = str(row.get("year","")).strip()
                    if d2 and d2 not in st.session_state.documents:
                        st.session_state.documents.append(d2)
                        st.session_state.titles.append(t2[:60])
                        st.session_state.sources.append("CSV")
                        st.session_state.years.append(y2)
                st.success(f"{len(df_up)} rows loaded.")

    st.markdown("---")

    # Example data
    col_ex, col_cl = st.columns([3,1])
    with col_ex:
        if st.button("🧪 Load Example Dataset (8 documents)"):
            examples = [
                ("AI Ethics Overview","Artificial intelligence raises ethical concerns about bias, fairness, and accountability. Researchers argue that AI systems must be transparent and explainable to ensure public trust.","2022"),
                ("Bias in Machine Learning","Bias in machine learning models reflects historical inequalities. Debiasing techniques and fairness-aware algorithms are active research topics in AI ethics.","2023"),
                ("Digital Transformation","Digital transformation refers to integrating digital technology into all areas of a business. It changes how companies operate and deliver value to customers.","2021"),
                ("SaaS Business Models","Software as a Service provides subscription-based access to software hosted in the cloud. Key metrics include churn rate, MRR, and customer lifetime value.","2023"),
                ("Theology and AI","Some theologians explore the spiritual implications of artificial intelligence. Questions arise about consciousness, personhood, and the ethical treatment of AI systems.","2022"),
                ("NLP and Text Mining","Natural Language Processing enables machines to understand human language. Applications include sentiment analysis, text classification, and information retrieval.","2023"),
                ("Data Governance","Data governance establishes policies for data quality, privacy, and security. It is critical for organisations managing large volumes of sensitive information.","2021"),
                ("Clustering Algorithms","K-means and hierarchical clustering are common unsupervised learning methods. They group data points by similarity and are widely used in text analysis.","2022"),
            ]
            for title, text, year in examples:
                if text not in st.session_state.documents:
                    st.session_state.documents.append(text)
                    st.session_state.titles.append(title)
                    st.session_state.sources.append("Example")
                    st.session_state.years.append(year)
            st.success("Example dataset loaded!")
    with col_cl:
        if st.button("🗑️ Clear All"):
            for k in ["documents","titles","sources","years"]:
                st.session_state[k] = []
            st.success("Cleared.")

    # Collection overview
    docs = st.session_state.documents
    st.markdown(f"### Current Collection: {len(docs)} document(s)")
    if docs:
        df_overview = pd.DataFrame({"Title":st.session_state.titles,
                                     "Source":st.session_state.sources,
                                     "Year":st.session_state.years,
                                     "Words":[len(d.split()) for d in docs]})
        st.dataframe(df_overview, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — DATA PREPARATION / EDA
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    phase("3 — Data Preparation & EDA")
    st.header("Clean, Preprocess & Explore")

    docs    = st.session_state.documents
    titles  = st.session_state.titles
    sources = st.session_state.sources
    years   = st.session_state.years

    if len(docs) < 2:
        st.info("Add at least 2 documents in Phase 2 to continue.")
        st.stop()

    lang_sel = st.selectbox("Preprocessing language", ["english","german"], index=0)
    processed = [preprocess(d, lang_sel) for d in docs]

    # Preprocessing preview
    with st.expander("🔎 Preprocessing Preview (original vs. cleaned)"):
        idx_p = st.selectbox("Document:", range(len(titles)), format_func=lambda i: titles[i], key="prep_prev")
        ca, cb = st.columns(2)
        ca.markdown("**Original**");    ca.write(docs[idx_p])
        cb.markdown("**Preprocessed**"); cb.write(processed[idx_p])

    # Vectorise
    vectorizer   = TfidfVectorizer(stop_words="english" if lang_sel=="english" else None, max_features=500)
    tfidf_matrix = vectorizer.fit_transform(processed)
    terms        = vectorizer.get_feature_names_out()

    st.markdown("### Summary Statistics")
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, len(docs), "Documents")
    kpi(c2, len(terms), "Vocabulary Size")
    kpi(c3, f"{1-(tfidf_matrix.nnz/(tfidf_matrix.shape[0]*tfidf_matrix.shape[1])):.1%}", "Matrix Sparsity")
    avg_len = int(np.mean([len(d.split()) for d in docs]))
    kpi(c4, avg_len, "Avg. Word Count")

    st.markdown("")
    col1, col2 = st.columns(2)
    with col1:
        df_meta = pd.DataFrame({"Word Count":[len(d.split()) for d in docs],"Source":sources})
        fig_wc  = px.histogram(df_meta, x="Word Count", color="Source", nbins=15,
                                title="Document Length Distribution")
        st.plotly_chart(fig_wc, use_container_width=True)
    with col2:
        src_df = pd.DataFrame({"Source":sources}).value_counts().reset_index()
        src_df.columns = ["Source","Count"]
        fig_src = px.pie(src_df, names="Source", values="Count", title="Documents by Data Source")
        st.plotly_chart(fig_src, use_container_width=True)

    tfidf_dense = pd.DataFrame(tfidf_matrix.toarray(), columns=terms)
    top_t = tfidf_dense.sum().sort_values(ascending=False).head(20).reset_index()
    top_t.columns = ["Term","TF-IDF Weight"]
    fig_terms = px.bar(top_t, x="TF-IDF Weight", y="Term", orientation="h",
                        title="Top 20 Terms by Total TF-IDF Weight")
    fig_terms.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_terms, use_container_width=True)

    # Store for later tabs
    st.session_state["vectorizer"]   = vectorizer
    st.session_state["tfidf_matrix"] = tfidf_matrix
    st.session_state["terms"]        = terms
    st.session_state["processed"]    = processed
    st.session_state["lang_sel"]     = lang_sel

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — MODELING
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    phase("4 — Modeling")
    st.header("Apply ML Models")

    if "tfidf_matrix" not in st.session_state:
        st.info("Complete Phase 3 first to build the TF-IDF model.")
        st.stop()

    tfidf_matrix = st.session_state["tfidf_matrix"]
    vectorizer   = st.session_state["vectorizer"]
    terms        = st.session_state["terms"]
    processed    = st.session_state["processed"]
    lang_sel     = st.session_state.get("lang_sel","english")
    docs         = st.session_state.documents
    titles       = st.session_state.titles
    sources      = st.session_state.sources
    years        = st.session_state.years

    model_tab_a, model_tab_b = st.tabs(["🔍 Similarity & Recommendations", "🗂️ Topic Clustering"])

    with model_tab_a:
        st.subheader("Cosine Similarity – Recommendation Engine")
        st.markdown("Select a document or enter a free-text query. The model ranks all other documents by cosine similarity to find the most relevant matches.")

        mode = st.radio("Query source:", ["Select from collection","Enter new query text"], horizontal=True)
        if mode == "Select from collection":
            sel     = st.selectbox("Document:", range(len(titles)), format_func=lambda i: titles[i])
            q_vec   = tfidf_matrix[sel]
            q_label = titles[sel]
        else:
            q_text = st.text_area("Enter your query note or research question:")
            if not q_text.strip():
                st.info("Please enter a query text.")
                st.stop()
            q_vec   = vectorizer.transform([preprocess(q_text, lang_sel)])
            q_label = "Custom Query"
            sel     = -1

        top_n = st.slider("Recommendations to show:", 1, min(10, len(docs)-(1 if sel>=0 else 0)), 3)
        sims  = cosine_similarity(q_vec, tfidf_matrix).flatten()
        if sel >= 0: sims[sel] = -1
        ranked = np.argsort(sims)[::-1][:top_n]

        sim_df = pd.DataFrame({"Title":[titles[i] for i in ranked],
                                "Similarity":[round(sims[i],4) for i in ranked],
                                "Source":[sources[i] for i in ranked],
                                "Year":[years[i] for i in ranked]})
        st.dataframe(sim_df, use_container_width=True)

        fig_sim = px.bar(sim_df, x="Similarity", y="Title", orientation="h",
                          color="Similarity", color_continuous_scale="Blues",
                          title=f"Cosine Similarity Rankings for: {q_label}", range_x=[0,1])
        fig_sim.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_sim, use_container_width=True)

        st.download_button("⬇️ Export Recommendations as CSV",
                            sim_df.to_csv(index=False).encode("utf-8"),
                            "recommendations.csv","text/csv")

        # Store recommendations for evaluation
        st.session_state["last_recommendations"] = sim_df
        st.session_state["last_q_label"]         = q_label

    with model_tab_b:
        st.subheader("Topic Clustering – Unsupervised Learning")

        cluster_method = st.radio("Algorithm:", ["K-Means","DBSCAN"], horizontal=True)

        if cluster_method == "K-Means":
            n_clust = st.slider("Number of clusters (k):", 2, min(8,len(docs)), min(3,len(docs)))
            inertias = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(tfidf_matrix).inertia_
                        for k in range(2, min(9,len(docs)+1))]
            fig_elbow = px.line(x=list(range(2,min(9,len(docs)+1))), y=inertias, markers=True,
                                 title="Elbow Curve – Optimal k Selection")
            fig_elbow.update_xaxes(title_text="k"); fig_elbow.update_yaxes(title_text="Inertia")
            st.plotly_chart(fig_elbow, use_container_width=True)

            km        = KMeans(n_clusters=n_clust, random_state=42, n_init=10)
            labels_cl = km.fit_predict(tfidf_matrix)
            order_c   = km.cluster_centers_.argsort()[:,::-1]
            kw_fn     = lambda c: ", ".join([terms[i] for i in order_c[c,:8]])
        else:
            eps_val   = st.slider("eps:", 0.1, 2.0, 0.8, 0.05)
            min_s     = st.slider("min_samples:", 1, 5, 2)
            db        = DBSCAN(eps=eps_val, min_samples=min_s, metric="cosine")
            labels_cl = db.fit_predict(normalize(tfidf_matrix))
            n_clust   = len(set(labels_cl)) - (1 if -1 in labels_cl else 0)
            order_c   = None
            kw_fn     = lambda c: ""
            st.info(f"DBSCAN found {n_clust} cluster(s) ({list(labels_cl).count(-1)} noise doc(s)).")

        cluster_df = pd.DataFrame({"Title":titles,"Cluster":labels_cl,"Source":sources})
        for c in sorted(set(labels_cl)):
            name    = f"Cluster {c+1}" if c>=0 else "🔇 Noise"
            members = cluster_df[cluster_df["Cluster"]==c]
            kw      = kw_fn(c) if c>=0 else ""
            with st.expander(f"{name} — {len(members)} doc(s)" + (f" | *{kw}*" if kw else "")):
                for _, row in members.iterrows():
                    idx_m = titles.index(row["Title"])
                    st.markdown(f"**{row['Title']}** *(Source: {row['Source']})*")
                    st.caption(docs[idx_m][:250]+"…")

        if tfidf_matrix.shape[1] >= 2:
            svd    = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(tfidf_matrix)
            sdf    = pd.DataFrame({"x":coords[:,0],"y":coords[:,1],"Title":titles,
                                    "Cluster":[str(l) for l in labels_cl],"Source":sources})
            fig_sc = px.scatter(sdf, x="x", y="y", color="Cluster", text="Title",
                                 hover_data=["Source"],
                                 title="Document Clusters (2D SVD Projection)")
            fig_sc.update_traces(textposition="top center", marker=dict(size=12))
            fig_sc.update_layout(height=480)
            st.plotly_chart(fig_sc, use_container_width=True)

        export_cl = pd.DataFrame({"Title":titles,"Cluster":labels_cl,"Source":sources,"Year":years})
        st.download_button("⬇️ Export Cluster Assignments as CSV",
                            export_cl.to_csv(index=False).encode("utf-8"),
                            "clusters.csv","text/csv")
        st.session_state["last_clusters"] = export_cl

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5 — EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    phase("5 — Evaluation")
    st.header("Assess Model Quality")

    if "tfidf_matrix" not in st.session_state:
        st.info("Complete Phases 3 and 4 first.")
        st.stop()

    tfidf_matrix = st.session_state["tfidf_matrix"]
    titles       = st.session_state.titles
    sources      = st.session_state.sources

    # Similarity heatmap
    st.subheader("Pairwise Similarity Heatmap")
    sim_mat = cosine_similarity(tfidf_matrix)
    heat_df = pd.DataFrame(sim_mat, index=titles, columns=titles)
    fig_h   = px.imshow(heat_df, text_auto=".2f", color_continuous_scale="Blues",
                         title="Cosine Similarity Matrix (all documents)",
                         labels=dict(color="Similarity"))
    fig_h.update_layout(height=max(400, len(titles)*40))
    st.plotly_chart(fig_h, use_container_width=True)
    st.caption("Values close to **1.0** = highly similar | **0.0** = unrelated")

    # KPI summary
    st.subheader("Model KPIs")
    np.fill_diagonal(sim_mat, 0)
    c1,c2,c3,c4 = st.columns(4)
    kpi(c1, f"{sim_mat.max():.3f}", "Max Similarity")
    kpi(c2, f"{sim_mat.mean():.3f}", "Avg. Similarity")
    kpi(c3, len(titles), "Documents Evaluated")
    kpi(c4, st.session_state.get("vectorizer") and len(st.session_state["terms"]) or "–", "Vocab Size")
    st.markdown("")

    # User feedback on recommendations
    st.subheader("User Feedback on Recommendations")
    st.markdown("Rate the last set of recommendations to simulate feedback-driven improvement.")
    if "last_recommendations" in st.session_state:
        rec_df = st.session_state["last_recommendations"]
        q_lbl  = st.session_state.get("last_q_label","–")
        st.markdown(f"Query: *{q_lbl}*")
        for _, row in rec_df.iterrows():
            col_t, col_f = st.columns([4,1])
            col_t.write(f"**{row['Title']}** (similarity: {row['Similarity']})")
            feedback = col_f.radio("", ["👍 Relevant","👎 Not relevant"],
                                    key=f"fb_{row['Title']}", horizontal=True, label_visibility="collapsed")
            st.session_state.feedback[row["Title"]] = feedback
        if st.session_state.feedback:
            fb_df = pd.DataFrame({"Title":list(st.session_state.feedback.keys()),
                                   "Feedback":list(st.session_state.feedback.values())})
            pos = fb_df["Feedback"].str.contains("👍").sum()
            neg = len(fb_df) - pos
            st.markdown(f"**Precision (user-rated):** {pos}/{len(fb_df)} marked relevant — `{pos/len(fb_df):.0%}`")
    else:
        st.info("Run recommendations in Phase 4 first.")

    # Criteria check
    st.subheader("Success Criteria Review")
    docs = st.session_state.documents
    avg_sim_val = sim_mat.mean()
    criteria = [
        ("Avg. similarity ≥ 0.05", avg_sim_val >= 0.05, f"{avg_sim_val:.3f}"),
        ("At least 8 documents",   len(docs) >= 8,      f"{len(docs)} docs"),
        ("Multiple data sources",  len(set(st.session_state.sources)) >= 2,
         f"{len(set(st.session_state.sources))} source(s)"),
        ("Vocabulary size ≥ 50",   len(st.session_state.get('terms',[])) >= 50,
         f"{len(st.session_state.get('terms',[]))} terms"),
    ]
    for label, passed, value in criteria:
        icon = "✅" if passed else "❌"
        st.markdown(f"{icon} **{label}** — {value}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 6 — DEPLOYMENT & INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    phase("6 — Deployment & Insights")
    st.header("Insights, Export & Transparency")

    docs    = st.session_state.documents
    titles  = st.session_state.titles
    sources = st.session_state.sources
    years   = st.session_state.years

    # Auto-generated report
    st.subheader("📝 Auto-Generated Research Summary")
    goal = st.session_state.biz_goal or "(not defined)"
    question = st.session_state.biz_question or "(not defined)"
    n_src = len(set(sources)) if sources else 0
    n_yrs = len(set(years)-{""}) if years else 0

    st.markdown(f"""
> **Research Domain:** {goal}
>
> **Research Question:** {question}

This system analysed **{len(docs)} document(s)** collected from **{n_src} data source(s)**,
spanning **{n_yrs} publication year(s)**. Documents were preprocessed using stopword removal
and stemming, then vectorised with TF-IDF (max. 500 features). Pairwise cosine similarity
was computed to power the recommendation engine. Topic clustering was applied using K-Means
or DBSCAN to identify recurring themes across the reading list.
    """)

    # Full export
    st.subheader("⬇️ Export Full Dataset")
    if docs:
        full_df = pd.DataFrame({"Title":titles,"Source":sources,"Year":years,
                                  "Word Count":[len(d.split()) for d in docs],
                                  "Text":docs})
        st.download_button("Download full collection as CSV",
                            full_df.to_csv(index=False).encode("utf-8"),
                            "research_collection.csv","text/csv")

    # Transparency
    st.subheader("⚖️ Transparency & Responsible AI")
    st.markdown("""
**TF-IDF** weights terms by their frequency in a document relative to the full collection — common
words get low weights, rare meaningful terms get high weights.

**Cosine Similarity** measures the angle between two document vectors: 1.0 = identical vocabulary
distribution, 0.0 = no shared terms.

**K-Means** partitions documents into *k* clusters by minimising within-cluster variance.
Results depend on the chosen *k* and random initialisation (fixed: `random_state=42`).

**Limitations**
- TF-IDF captures word frequency, not semantic meaning — synonyms are not resolved.
- Very short documents produce sparse, less discriminative vectors.
- Clustering quality depends heavily on *k* and document quantity.

**Data Privacy**
> 🔒 All text remains in local browser session state. DNB queries are sent to the public
> Deutsche Nationalbibliothek SRU API (CC0 1.0 metadata licence).
    """)
