
# site_linker.py
import os, pickle
from typing import List, Tuple, Dict, Optional

import streamlit as st
import requests
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SITES = [
    "https://technomeow.com",
    "https://technobark.com",
    "https://dogvills.com",
    "https://petparentadvisor.org",
    "https://seniorpups.com",
]

SITEMAP_CANDIDATES = ["/sitemap_index.xml", "/sitemap.xml", "/sitemap_index.xml.gz", "/sitemap.xml.gz"]

ONTOLOGY = {
    "mobility": {
        "synonyms": ["mobility", "joint health", "hip dysplasia", "arthritis", "stiffness", "lameness", "pain", "recover", "rehab", "physical therapy", "cartilage", "inflammation", "range of motion"],
        "solutions": ["ramp", "steps", "stairs", "harness", "lift harness", "traction", "non-slip", "booties", "low-impact exercise", "hydrotherapy", "massage", "acupuncture", "orthopedic"],
        "supplements": ["glucosamine", "chondroitin", "msm", "omega-3", "fish oil", "turmeric", "curcumin", "green lipped mussel", "hyaluronic acid", "collagen"],
    },
    "senior": {"synonyms": ["senior", "aging", "older", "geriatric", "longevity", "age-related"]},
    "species": {"dog": ["dog", "dogs", "canine"], "cat": ["cat", "cats", "feline"]},
}

CACHE_DIR = "data"
ARTICLES_PKL = os.path.join(CACHE_DIR, "articles.pkl")
TFIDF_PKL = os.path.join(CACHE_DIR, "tfidf.pkl")
REQUEST_TIMEOUT = 20
MAX_URLS_PER_SITEMAP = 5000

os.makedirs(CACHE_DIR, exist_ok=True)

def user_agent() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; SiteLinker/1.0)"}

def fetch(url: str):
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=user_agent())
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def discover_sitemaps() -> List[str]:
    smaps = []
    for base in SITES:
        for path in SITEMAP_CANDIDATES:
            sm_url = base.rstrip("/") + path
            content = fetch(sm_url)
            if not content:
                continue
            head = content[:2000].decode("utf-8", errors="ignore").lower()
            if "<urlset" in head or "<sitemapindex" in head:
                smaps.append(sm_url)
                break
    return smaps

def parse_sitemap_xml(xml_bytes: bytes) -> Tuple[List[str], List[str]]:
    urls, children = [], []
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
        def ns_tag(name: str) -> str:
            if "}" in root.tag:
                ns = root.tag.split("}")[0].strip("{")
                return f"{{{ns}}}{name}"
            return name
        if root.tag.endswith("sitemapindex"):
            for sm in root.findall(ns_tag("sitemap")):
                loc_el = sm.find(ns_tag("loc"))
                if loc_el is not None and loc_el.text:
                    children.append(loc_el.text.strip())
        elif root.tag.endswith("urlset"):
            for u in root.findall(ns_tag("url")):
                loc_el = u.find(ns_tag("loc"))
                if loc_el is not None and loc_el.text:
                    urls.append(loc_el.text.strip())
    except Exception:
        pass
    urls = [u for u in urls if not any(x in u for x in ["/image_sitemap", "/video_sitemap"])]
    return urls, children

def collect_urls_from_sitemap(start_url: str) -> List[str]:
    seen = set(); to_visit = [start_url]; all_urls = []
    while to_visit:
        sm_url = to_visit.pop()
        if sm_url in seen: continue
        seen.add(sm_url)
        xml = fetch(sm_url)
        if not xml: continue
        urls, children = parse_sitemap_xml(xml)
        all_urls.extend(urls[:MAX_URLS_PER_SITEMAP])
        to_visit.extend(children)
        if len(all_urls) >= MAX_URLS_PER_SITEMAP: break
    return list(dict.fromkeys(all_urls))

def smart_extract_text(html_bytes: bytes):
    html = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html5lib")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    main_node = soup.find("article") or soup.find("main")
    if not main_node:
        candidates = soup.find_all(["div", "section"], class_=True)
        best = None; best_len = 0
        for c in candidates:
            cls = " ".join(c.get("class", [])).lower()
            if any(k in cls for k in ["content", "entry-content", "post-content", "article-content", "single-post"]):
                txt = c.get_text(" ", strip=True)
                if len(txt) > best_len:
                    best = c; best_len = len(txt)
        main_node = best if best is not None else soup.body
    text = main_node.get_text(" ", strip=True) if main_node else soup.get_text(" ", strip=True)
    preview = text[:600] + ("…" if len(text) > 600 else "")
    return title or "(untitled)", text, preview

def extract_main_content(url: str):
    html = fetch(url)
    if not html: return None, None, None
    try: return smart_extract_text(html)
    except Exception: return None, None, None

def text_for_index(title: str, text: str) -> str:
    return ((title or "") + "\n\n" + (text or "")[:2000]).strip()

def load_index():
    if not (os.path.exists(ARTICLES_PKL) and os.path.exists(TFIDF_PKL)):
        return [], None, None
    with open(ARTICLES_PKL, "rb") as f:
        rows = pickle.load(f)
    with open(TFIDF_PKL, "rb") as f:
        pack = pickle.load(f)
    return rows, pack["X"], pack["vectorizer"]

def save_index(rows, X, vectorizer):
    with open(ARTICLES_PKL, "wb") as f:
        pickle.dump(rows, f)
    with open(TFIDF_PKL, "wb") as f:
        pickle.dump({"X": X, "vectorizer": vectorizer}, f)

def ontology_overlap_score(text: str):
    t = (text or "").lower(); detail = []
    def count_any(terms, label):
        c = sum(t.count(term.lower()) for term in terms)
        if c > 0: detail.append(f"{label}:{c}")
        return c
    score = 0.0
    c1 = count_any(ONTOLOGY["mobility"]["synonyms"], "mobility")
    c2 = count_any(ONTOLOGY["mobility"]["solutions"], "aids")
    c3 = count_any(ONTOLOGY["mobility"]["supplements"], "supps")
    score += 0.5 * min(c1, 5) + 0.3 * min(c2, 5) + 0.2 * min(c3, 5)
    c4 = sum(t.count(s) for s in ONTOLOGY["senior"]["synonyms"])
    score += 0.3 * min(c4, 5)
    return score, ", ".join(detail[:4])

def species_penalty(a_text: str, b_text: str) -> float:
    get = lambda txt, keys: sum((txt or "").lower().count(k) for k in keys)
    dA = get(a_text, ONTOLOGY["species"]["dog"]); cA = get(a_text, ONTOLOGY["species"]["cat"])
    dB = get(b_text, ONTOLOGY["species"]["dog"]); cB = get(b_text, ONTOLOGY["species"]["cat"])
    if dA > cA and cB > dB: return 0.07
    if cA > dA and dB > cB: return 0.07
    return 0.0

def suggest_anchor(text: str) -> str:
    anchors = []; t = (text or "").lower()
    if "ramp" in t: anchors.append("dog ramps for seniors")
    if "glucosamine" in t or "supplement" in t: anchors.append("joint supplements for mobility")
    if "exercise" in t or "low-impact" in t: anchors.append("low-impact exercise for older dogs")
    if "hip dysplasia" in t: anchors.append("hip dysplasia support")
    if not anchors: anchors.append("help for senior dog mobility")
    return ", ".join(anchors[:2])

def rank_related_tfidf(target_title, target_text, rows, X, vectorizer, same_domain_only=None, min_words=200):
    if not rows or X is None or vectorizer is None: return []
    q = vectorizer.transform([text_for_index(target_title, target_text)])
    sims = cosine_similarity(q, X)[0]
    out = []
    for i, r in enumerate(rows):
        if len((r["text"] or "").split()) < min_words: continue
        if same_domain_only and r["domain"] != same_domain_only: continue
        if r["text"] == target_text: continue
        ont_cand, ont_detail = ontology_overlap_score(r["text"])
        ont_tgt, _ = ontology_overlap_score(target_text)
        penalty = species_penalty(target_text, r["text"])
        final = 0.65*float(sims[i]) + 0.25*(min(ont_cand,5.0)/5.0) + 0.10*(min(ont_tgt,5.0)/5.0) - penalty
        out.append({"title": r["title"], "url": r["url"], "domain": r["domain"], "preview": r["preview"], "sim": float(sims[i]), "why": f"tfidf:{round(float(sims[i]),3)} | {ont_detail}", "anchor": suggest_anchor(r["text"]), "score": final})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

def build_index(sitemap_urls: List[str]):
    all_urls = []
    st.info("Collecting URLs from sitemaps…")
    for sm in sitemap_urls:
        st.write(f"Scanning: {sm}")
        urls = collect_urls_from_sitemap(sm)
        st.write(f"→ Found {len(urls)} URLs")
        all_urls.extend(urls)
    all_urls = list(dict.fromkeys(all_urls)); rows = []
    if not all_urls:
        st.warning("No URLs discovered. Check sitemap endpoints."); return
    progress = st.progress(0.0); total = len(all_urls)
    for i, url in enumerate(all_urls, 1):
        title, text, preview = extract_main_content(url)
        if not text or len(text) < 200: progress.progress(i/total); continue
        rows.append({"url": url, "title": title or "(untitled)", "text": text, "preview": preview, "domain": urlparse(url).netloc})
        progress.progress(i/total)
    if not rows:
        st.warning("No articles were indexed (pages too short or blocked)."); return
    docs = [text_for_index(r["title"], r["text"]) for r in rows]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_df=0.85, min_df=2, stop_words="english")
    X = vectorizer.fit_transform(docs)
    save_index(rows, X, vectorizer)
    st.success(f"Indexed {len(rows)} articles.")

st.set_page_config(page_title="Site Linker (Multi-Site Internal Linking)", layout="wide")
st.title("🔗 Site Linker — Cross-Site Related Article Finder")
st.caption("TF‑IDF similarity + pet ontology. No heavy dependencies.")

with st.expander("⚙️ Index controls"):
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        if st.button("🔄 Build / Refresh Index", type="primary"):
            sitemaps = discover_sitemaps()
            if not sitemaps:
                st.error("Couldn’t find any sitemaps. Check that the sites expose /sitemap_index.xml or /sitemap.xml")
            else:
                build_index(sitemaps)
                st.experimental_rerun()
    with col2:
        if st.button("🗑 Clear Cache Files"):
            for p in [ARTICLES_PKL, TFIDF_PKL]:
                if os.path.exists(p): os.remove(p)
            st.success("Cache cleared. Click refresh to rebuild.")
    with col3:
        st.write("**Tip:** Refresh after new posts or weekly. Searches stay fast between refreshes.")

rows, X, vectorizer = load_index()
st.write(f"**Indexed articles:** {len(rows)}")

tab1, tab2 = st.tabs(["🔎 Find by URL", "📝 Find by pasted text"])

with tab1:
    url = st.text_input("Article URL to analyze", placeholder="https://dogvills.com/why-dogs-lose-mobility-as-they-age/")
    prefer_same_site = st.checkbox("Prefer same-site suggestions first", value=True)
    k = st.slider("How many suggestions?", 3, 20, 8)
    if st.button("Find related (by URL)", disabled=(len(rows)==0)):
        try:
            t_title, t_text, _ = extract_main_content(url)
            if not t_text:
                st.error("Couldn’t extract text from that URL.")
            else:
                domain = urlparse(url).netloc
                domain_filter = domain if prefer_same_site else None
                results = rank_related_tfidf(t_title, t_text, rows, X, vectorizer, same_domain_only=domain_filter)
                if prefer_same_site:
                    same = [r for r in results if r["domain"] == domain]
                    cross = [r for r in results if r["domain"] != domain]
                    results = same[:k] + cross[:k]
                df = pd.DataFrame([{"Score": round(r["score"],3), "Title": r["title"], "URL": r["url"], "Suggested Anchor": r["anchor"], "Why": r["why"]} for r in results[:k]])
                st.dataframe(df, use_container_width=True)
                st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"), file_name="related_links.csv", mime="text/csv")
        except Exception as e:
            st.error(str(e))

with tab2:
    t_title = st.text_input("(Optional) Title for your draft/article")
    t_text = st.text_area("Paste article text (intro is fine)", height=220)
    prefer_domain = st.text_input("(Optional) Prefer matches from this domain", placeholder="dogvills.com")
    k2 = st.slider("How many suggestions? ", 3, 20, 8, key="k2")
    if st.button("Find related (by text)", disabled=(len(rows)==0 or not t_text.strip())):
        results = rank_related_tfidf(t_title, t_text, rows, X, vectorizer, same_domain_only=(prefer_domain or None))
        df = pd.DataFrame([{"Score": round(r["score"],3), "Title": r["title"], "URL": r["url"], "Suggested Anchor": r["anchor"], "Why": r["why"]} for r in results[:k2]])
        st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Export CSV", df.to_csv(index=False).encode("utf-8"), file_name="related_links.csv", mime="text/csv")

st.markdown("---")
st.caption("Scoring = 0.65×TF‑IDF + 0.25×ontology(candidate) + 0.10×ontology(target) − species mismatch penalty. You can tune ONTOLOGY in this file.")
