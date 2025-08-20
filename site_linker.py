# site_linker.py
# ------------------------------------------------------------
# Site Linker — Ultra‑light v0.3.3
# - Pure‑Python TF‑IDF (no sklearn/numpy/pandas)
# - Per‑site checkboxes, speed controls, exclude filters
# - FAST sitemap discovery: robots.txt + concurrent HEAD + cache
# - **Build button is at the very bottom of the page**
# - Fix: uses st.rerun() (no experimental rerun crash)
# ------------------------------------------------------------

import os, pickle, math, re, time, json
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# ---------------------- USER EDITABLE: SITES ----------------------
ALL_SITES = [
    "https://technomeow.com",
    "https://technobark.com",
    "https://dogvills.com",
    "https://petparentadvisor.org",
    "https://seniorpups.com",
]

SITEMAP_CANDIDATES_BASE = ["/sitemap_index.xml", "/sitemap.xml"]
SITEMAP_CANDIDATES_GZ   = ["/sitemap_index.xml.gz", "/sitemap.xml.gz"]

# Common non‑article paths to ignore
DEFAULT_EXCLUDE_SUBSTRINGS = [
    "/tag/", "/category/", "/page/", "/feed/", "/reviews/",
    "/author/", "/about/", "/privacy/", "/terms/", "/contact/",
    "/affiliate/", "/advertise/"
]

# ---------------------- ONTOLOGY -------------------
ONTOLOGY = {
    "mobility": {
        "synonyms": [
            "mobility", "joint health", "hip dysplasia", "arthritis",
            "stiffness", "lameness", "pain", "recover", "rehab",
            "physical therapy", "cartilage", "inflammation", "range of motion"
        ],
        "solutions": [
            "ramp", "steps", "stairs", "harness", "lift harness",
            "traction", "non-slip", "booties", "low-impact exercise",
            "hydrotherapy", "massage", "acupuncture", "orthopedic"
        ],
        "supplements": [
            "glucosamine", "chondroitin", "msm", "omega-3", "fish oil",
            "turmeric", "curcumin", "green lipped mussel", "hyaluronic acid", "collagen"
        ]
    },
    "senior": {"synonyms": ["senior","aging","older","geriatric","longevity","age-related"]},
    "species": {"dog": ["dog","dogs","canine"], "cat": ["cat","cats","feline"]}
}

# ---------------------- CONFIG ------------------------
CACHE_DIR = "data"
ARTICLES_PKL = os.path.join(CACHE_DIR, "articles.pkl")
LITE_PKL     = os.path.join(CACHE_DIR, "lite_index.pkl")
SITEMAP_CACHE_JSON = os.path.join(CACHE_DIR, "sitemaps_cache.json")

REQ_TIMEOUT = 8      # request timeout
HEAD_TIMEOUT = 5     # HEAD timeout
MAX_URLS_PER_SITEMAP = 5000  # hard safety cap

os.makedirs(CACHE_DIR, exist_ok=True)

STOPWORDS = set("""
a about above after again against all am an and any are as at be because been
before being below between both but by can did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not of
off on once only or other our ours ourselves out over own same she should so
some such than that the their theirs them themselves then there these they this
those through to too under until up very was we were what when where which while
who whom why with you your yours yourself yourselves
""".split())

# ---------------------- HTTP helpers ----------------------------------
def user_agent() -> Dict[str, str]:
    return {"User-Agent": "Mozilla/5.0 (compatible; SiteLinker/ultra/1.0)"}

def http_head(url: str, timeout: int = HEAD_TIMEOUT) -> Optional[requests.Response]:
    try:
        r = requests.head(url, headers=user_agent(), allow_redirects=True, timeout=timeout)
        return r
    except Exception:
        return None

def http_get(url: str, timeout: int = REQ_TIMEOUT, first_bytes: Optional[int] = None) -> Optional[bytes]:
    try:
        headers = user_agent().copy()
        if first_bytes:
            headers["Range"] = f"bytes=0-{first_bytes-1}"
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 or (r.status_code == 206 and first_bytes):
            return r.content
    except Exception:
        pass
    return None

# ---------------------- Sitemap discovery (fast) -----------------------
def discover_from_robots(site: str, timeout: int) -> List[str]:
    urls = []
    robots = http_get(site.rstrip("/") + "/robots.txt", timeout=timeout, first_bytes=4096)
    if not robots:
        return urls
    try:
        text = robots.decode("utf-8", errors="ignore")
        for line in text.splitlines():
            if line.lower().startswith("sitemap:"):
                sm = line.split(":", 1)[1].strip()
                if sm:
                    urls.append(sm)
    except Exception:
        pass
    return urls

def candidate_sitemaps_for_site(site: str, include_gz: bool) -> List[str]:
    cands = [site.rstrip("/") + p for p in SITEMAP_CANDIDATES_BASE]
    if include_gz:
        cands += [site.rstrip("/") + p for p in SITEMAP_CANDIDATES_GZ]
    return cands

def is_xmlish(headers: Dict[str, str], url: str) -> bool:
    ct = (headers.get("Content-Type") or "").lower()
    if "xml" in ct or "text/xml" in ct or "application/xml" in ct:
        return True
    if url.endswith(".xml") or url.endswith(".xml.gz"):
        return True
    if "gzip" in ct and url.endswith(".gz"):
        return True
    return False

def discover_sitemaps_fast(
    sites: List[str],
    use_cache: bool = True,
    check_robots: bool = True,
    include_gz: bool = False,
    max_workers: int = 12,
    head_timeout: int = HEAD_TIMEOUT
) -> List[str]:
    # Load cache
    cached: Dict[str, List[str]] = {}
    if use_cache and os.path.exists(SITEMAP_CACHE_JSON):
        try:
            with open(SITEMAP_CACHE_JSON, "r", encoding="utf-8") as f:
                cached = json.load(f)
        except Exception:
            cached = {}

    results: Dict[str, List[str]] = {s: list(cached.get(s, [])) for s in sites}

    # robots.txt hints
    if check_robots:
        for s in sites:
            if not results[s]:  # only if no cache
                hints = discover_from_robots(s, timeout=4)
                if hints:
                    results[s].extend(hints)

    # Build candidate list for remaining
    to_probe: List[Tuple[str, str]] = []  # (site, candidate_url)
    for s in sites:
        if results[s]:
            continue
        for c in candidate_sitemaps_for_site(s, include_gz=include_gz):
            to_probe.append((s, c))

    # Concurrent HEAD probes (fast)
    def head_one(item):
        s, url = item
        r = http_head(url, timeout=head_timeout)
        ok = (r is not None) and (r.status_code == 200) and is_xmlish(r.headers, url)
        return (s, url, ok)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(head_one, it) for it in to_probe]
        for fut in as_completed(futures):
            s, url, ok = fut.result()
            if ok:
                results[s].append(url)

    # Fallback to small GET for anything still empty
    still_empty = [s for s in sites if not results[s]]
    for s in still_empty:
        for c in candidate_sitemaps_for_site(s, include_gz=include_gz):
            blob = http_get(c, timeout=REQ_TIMEOUT, first_bytes=2048)
            if not blob:
                continue
            head = blob[:2048].decode("utf-8", errors="ignore").lower()
            if "<urlset" in head or "<sitemapindex" in head:
                results[s].append(c); break

    # Flatten and cache
    all_smaps = []
    for s in sites:
        uniq = []
        for u in results.get(s, []):
            if u not in uniq:
                uniq.append(u)
        results[s] = uniq
        all_smaps.extend(uniq)

    # Save cache
    try:
        with open(SITEMAP_CACHE_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    except Exception:
        pass

    return all_smaps

# ---------------------- Parsing / extraction -----------------------
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

def collect_urls_from_sitemaps(sitemap_urls: List[str]) -> List[str]:
    seen = set()
    to_visit = list(sitemap_urls)
    all_urls: List[str] = []

    while to_visit:
        sm_url = to_visit.pop()
        if sm_url in seen: 
            continue
        seen.add(sm_url)

        xml = http_get(sm_url, timeout=REQ_TIMEOUT, first_bytes=None)  # usually small
        if not xml:
            continue
        urls, children = parse_sitemap_xml(xml)
        all_urls.extend(urls[:MAX_URLS_PER_SITEMAP])
        to_visit.extend(children)
        if len(all_urls) >= MAX_URLS_PER_SITEMAP:
            break
    return list(dict.fromkeys(all_urls))

def url_allowed(u: str, exclude_patterns: List[str]) -> bool:
    path = urlparse(u).path.lower()
    for p in exclude_patterns:
        p = (p or "").strip().lower()
        if p and p in path:
            return False
    return True

def smart_extract_text(html_bytes: bytes):
    html = html_bytes.decode("utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html5lib")

    # Title & H1
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)

    # Prefer <article>, then <main>, else heuristics
    main_node = soup.find("article") or soup.find("main")
    if not main_node:
        candidates = soup.find_all(["div", "section"], class_=True)
        best = None
        best_len = 0
        for c in candidates:
            cls = " ".join(c.get("class", [])).lower()
            if any(k in cls for k in ["content","entry-content","post-content","article-content","single-post"]):
                txt = c.get_text(" ", strip=True)
                if len(txt) > best_len:
                    best = c; best_len = len(txt)
        main_node = best if best is not None else soup.body

    text = main_node.get_text(" ", strip=True) if main_node else soup.get_text(" ", strip=True)
    preview = text[:600] + ("…" if len(text) > 600 else "")
    return title or "(untitled)", text, preview

def extract_main_content(url: str):
    blob = http_get(url, timeout=REQ_TIMEOUT, first_bytes=None)
    if not blob:
        return None, None, None
    try:
        return smart_extract_text(blob)
    except Exception:
        return None, None, None

# ---------------------- LITE TF‑IDF -------------------------------
_token_re = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> List[str]:
    if not text:
        return []
    return [w for w in _token_re.findall(text.lower()) if w not in STOPWORDS and len(w) > 2]

def tokens_with_bigrams(tokens: List[str]) -> List[str]:
    if len(tokens) < 2:
        return tokens
    bigrams = [tokens[i] + "_" + tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams

def text_for_index(title: str, text: str) -> str:
    return ((title or "") + "\\n\\n" + (text or "")[:2000]).strip()

def build_lite_index(rows: List[Dict]):
    docs_tokens: List[List[str]] = []
    df: Dict[str, int] = {}
    N = len(rows)

    for r in rows:
        toks = tokens_with_bigrams(tokenize(text_for_index(r["title"], r["text"])))
        docs_tokens.append(toks)
        seen_terms = set(toks)
        for t in seen_terms:
            df[t] = df.get(t, 0) + 1

    # IDF
    idf: Dict[str, float] = {}
    for t, c in df.items():
        idf[t] = math.log((1 + N) / (1 + c)) + 1.0  # smoothed IDF

    # Per-doc vectors
    doc_vectors: List[Dict[str, float]] = []
    doc_norms: List[float] = []

    for toks in docs_tokens:
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        for t, f in tf.items():
            w = (1.0 + math.log(f)) * idf.get(t, 0.0)
            if w > 0:
                vec[t] = w
        norm = math.sqrt(sum(w*w for w in vec.values())) or 1.0
        doc_vectors.append(vec)
        doc_norms.append(norm)

    pack = {"idf": idf, "doc_vectors": doc_vectors, "doc_norms": doc_norms, "N": N}
    with open(LITE_PKL, "wb") as f:
        pickle.dump(pack, f)

def load_index():
    if not (os.path.exists(ARTICLES_PKL) and os.path.exists(LITE_PKL)):
        return [], None
    with open(ARTICLES_PKL, "rb") as f:
        rows = pickle.load(f)
    with open(LITE_PKL, "rb") as f:
        pack = pickle.load(f)
    return rows, pack

def save_rows(rows: List[Dict]):
    with open(ARTICLES_PKL, "wb") as f:
        pickle.dump(rows, f)

def cosine_sim_sparse(qvec: Dict[str, float], qnorm: float, dvec: Dict[str, float], dnorm: float) -> float:
    dot = 0.0
    for t, qw in qvec.items():
        dw = dvec.get(t)
        if dw is not None:
            dot += qw * dw
    return dot / (qnorm * dnorm) if qnorm and dnorm else 0.0

def make_query_vector(title: str, text: str, idf: Dict[str, float]) -> Tuple[Dict[str, float], float]:
    toks = tokens_with_bigrams(tokenize(text_for_index(title, text)))
    tf: Dict[str, int] = {}
    for t in toks:
        tf[t] = tf.get(t, 0) + 1
    qvec: Dict[str, float] = {}
    for t, f in tf.items():
        w = (1.0 + math.log(f)) * idf.get(t, 0.0)
        if w > 0:
            qvec[t] = w
    qnorm = math.sqrt(sum(w*w for w in qvec.values())) or 1.0
    return qvec, qnorm

# ---------------------- RANKING ------------------------
def ontology_overlap_score(text: str) -> Tuple[float, str]:
    t = (text or "").lower()
    score = 0.0
    detail = []

    def count_any(terms, label):
        c = sum(t.count(term.lower()) for term in terms)
        if c > 0:
            detail.append(f"{label}:{c}")
        return c

    if "mobility" in ONTOLOGY:
        c1 = count_any(ONTOLOGY["mobility"]["synonyms"], "mobility")
        c2 = count_any(ONTOLOGY["mobility"]["solutions"], "aids")
        c3 = count_any(ONTOLOGY["mobility"]["supplements"], "supps")
        score += 0.5 * min(c1, 5) + 0.3 * min(c2, 5) + 0.2 * min(c3, 5)

    if "senior" in ONTOLOGY:
        c4 = count_any(ONTOLOGY["senior"]["synonyms"], "senior")
        score += 0.3 * min(c4, 5)

    return score, ", ".join(detail[:4])

def species_penalty(a_text: str, b_text: str) -> float:
    get = lambda txt, keys: sum((txt or "").lower().count(k) for k in keys)
    dA = get(a_text, ONTOLOGY["species"]["dog"])
    cA = get(a_text, ONTOLOGY["species"]["cat"])
    dB = get(b_text, ONTOLOGY["species"]["dog"])
    cB = get(b_text, ONTOLOGY["species"]["cat"])
    if dA > cA and cB > dB: return 0.07
    if cA > dA and dB > cB: return 0.07
    return 0.0

def suggest_anchor(text: str) -> str:
    anchors = []
    t = (text or "").lower()
    if "ramp" in t: anchors.append("dog ramps for seniors")
    if "glucosamine" in t or "supplement" in t: anchors.append("joint supplements for mobility")
    if "exercise" in t or "low-impact" in t: anchors.append("low-impact exercise for older dogs")
    if "hip dysplasia" in t: anchors.append("hip dysplasia support")
    if not anchors: anchors.append("help for senior dog mobility")
    return ", ".join(anchors[:2])

def rank_related(
    target_title: str,
    target_text: str,
    rows: List[Dict],
    pack: Dict,
    same_domain_only: Optional[str] = None,
    min_words: int = 200
) -> List[Dict]:
    if not rows or not pack:
        return []
    idf = pack["idf"]
    qvec, qnorm = make_query_vector(target_title, target_text, idf)

    out = []
    for i, r in enumerate(rows):
        if len((r["text"] or "").split()) < min_words:
            continue
        if same_domain_only and r["domain"] != same_domain_only:
            continue
        if r["text"] == target_text:
            continue

        dvec = pack["doc_vectors"][i]
        dnorm = pack["doc_norms"][i]
        sim = cosine_sim_sparse(qvec, qnorm, dvec, dnorm)

        ont_cand, ont_detail = ontology_overlap_score(r["text"])
        ont_tgt, _ = ontology_overlap_score(target_text)
        penalty = species_penalty(target_text, r["text"])

        final = (
            0.70 * float(sim) +
            0.20 * (min(ont_cand, 5.0) / 5.0) +
            0.10 * (min(ont_tgt, 5.0) / 5.0) -
            penalty
        )

        out.append({
            "title": r["title"],
            "url": r["url"],
            "domain": r["domain"],
            "preview": r["preview"],
            "sim": round(float(sim), 4),
            "why": f"tfidf:{round(float(sim),3)} | {ont_detail}",
            "anchor": suggest_anchor(r["text"]),
            "score": final
        })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ---------------------- INDEX BUILD -------------------------------
def build_index(
    selected_sites: List[str],
    use_cache_sitemaps: bool,
    check_robots: bool,
    include_gz_candidates: bool,
    max_pages_per_site: int,
    max_workers: int,
    exclude_common: bool,
    extra_excludes_text: str,
    force_refresh: bool
):
    excludes = [e.strip() for e in extra_excludes_text.split(",") if e.strip()]
    if exclude_common:
        excludes = DEFAULT_EXCLUDE_SUBSTRINGS + excludes

    st.info("Discovering sitemaps… (fast mode)")
    smaps = discover_sitemaps_fast(
        selected_sites,
        use_cache=use_cache_sitemaps,
        check_robots=check_robots,
        include_gz=include_gz_candidates,
        max_workers=max_workers,
        head_timeout=HEAD_TIMEOUT
    )
    if not smaps:
        st.error("No sitemaps found for the selected sites."); return

    # Collect URLs
    all_urls = collect_urls_from_sitemaps(smaps)

    # Filter to selected domains and excludes
    allowed_domains = {urlparse(s).netloc for s in selected_sites}
    url_pool = [u for u in all_urls if urlparse(u).netloc in allowed_domains and url_allowed(u, excludes)]

    # Enforce per‑site caps
    capped: List[str] = []
    per_count: Dict[str, int] = {d: 0 for d in allowed_domains}
    for u in url_pool:
        d = urlparse(u).netloc
        if per_count[d] < max_pages_per_site:
            capped.append(u)
            per_count[d] += 1

    # Load existing rows and skip already cached unless forced
    existing_rows, _ = load_index()
    existing_by_url = {r["url"]: r for r in existing_rows}
    to_fetch = [u for u in capped if force_refresh or u not in existing_by_url]

    st.write(f"Fetching {len(to_fetch)} new pages (threads={max_workers})…")
    progress = st.progress(0.0)

    # Concurrent fetch + extract
    def fetch_and_extract(u: str):
        blob = http_get(u, timeout=REQ_TIMEOUT, first_bytes=None)
        if not blob:
            return None
        try:
            title, text, preview = smart_extract_text(blob)
            if text and len(text) >= 200:
                return {"url": u, "title": title or "(untitled)", "text": text, "preview": preview, "domain": urlparse(u).netloc}
        except Exception:
            return None
        return None

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(fetch_and_extract, u) for u in to_fetch]
        total = max(1, len(futs))
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r: results.append(r)
            progress.progress(i/total)

    # Merge rows
    merged = list(existing_rows)
    by_url = {m["url"]: m for m in merged}
    for r in results:
        by_url[r["url"]] = r
    merged = list(by_url.values())

    if not merged:
        st.warning("No articles were indexed (pages too short or blocked).")
        return

    # Save + build vector index
    save_rows(merged)
    build_lite_index(merged)
    st.success(f"Indexed {len(merged)} articles across {len(selected_sites)} site(s).")

# ---------------------- UI ---------------------------------------
st.set_page_config(page_title="Site Linker (Ultra‑light)", layout="wide")
st.title("🔗 Site Linker — Cross‑Site Related Article Finder")
st.caption("Ultra‑light build: pure‑Python TF‑IDF + pet ontology. Minimal dependencies.")

# 1) Choose sites
st.subheader("1) Choose sites to scan")
cols = st.columns(3)
selected_sites = []
for i, site in enumerate(ALL_SITES):
    with cols[i % 3]:
        if st.checkbox(site, value=True):
            selected_sites.append(site)

# 2) Speed & filters
st.subheader("2) Speed & filters")
colA, colB, colC = st.columns([1,1,2])
with colA:
    max_pages_per_site = st.slider("Max pages per site", 20, 2000, 120, help="Cap pages per site to keep builds quick.")
with colB:
    max_workers = st.slider("Concurrency (threads)", 1, 24, 12, help="Higher = faster. Be kind to origin servers.")
with colC:
    exclude_common = st.checkbox("Exclude common paths (/tag/, /category/, /page/, /feed/, /reviews/, /author/, /about/, /privacy/, /terms/, /contact/, /affiliate/, /advertise/)", value=True)

extra_excludes = st.text_input("Extra paths to exclude (comma‑separated, optional)", value="")
force_refresh = st.checkbox("Force re-crawl even if cached", value=False)

# Advanced discovery (fast mode)
with st.expander("Advanced sitemap discovery (fast mode)"):
    use_cache_sitemaps = st.checkbox("Use sitemap cache", value=True)
    check_robots = st.checkbox("Check robots.txt for sitemap hints", value=True)
    include_gz_candidates = st.checkbox("Try .gz sitemap candidates", value=False)

# Load index (for searching)
rows, pack = load_index()
st.write(f"**Indexed articles:** {len(rows)}")

# 3) Search tabs
st.subheader("3) Search for related links")
tab1, tab2 = st.tabs(["🔎 Find by URL", "📝 Find by pasted text"])

with tab1:
    url = st.text_input("Article URL to analyze", placeholder="https://dogvills.com/why-dogs-lose-mobility-as-they-age/")
    prefer_same_site = st.checkbox("Prefer same-site suggestions first", value=True)
    k = st.slider("How many suggestions?", 3, 20, 8)
    disabled = len(rows)==0
    if disabled:
        st.info("Index is empty. Build the index at the bottom first.")
    if st.button("Find related (by URL)", disabled=disabled):
        try:
            t_title, t_text, _ = extract_main_content(url)
            if not t_text:
                st.error("Couldn’t extract text from that URL.")
            else:
                domain = urlparse(url).netloc
                domain_filter = domain if prefer_same_site else None
                results = rank_related(t_title, t_text, rows, pack, same_domain_only=domain_filter)
                if prefer_same_site:
                    same = [r for r in results if r["domain"] == domain]
                    cross = [r for r in results if r["domain"] != domain]
                    results = same[:k] + cross[:k]
                table = [{
                    "Score": round(r["score"],3),
                    "Title": r["title"],
                    "URL": r["url"],
                    "Suggested Anchor": r["anchor"],
                    "Why": r["why"]
                } for r in results[:k]]
                st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tab2:
    t_title = st.text_input("(Optional) Title for your draft/article")
    t_text = st.text_area("Paste article text (intro is fine)", height=220)
    prefer_domain = st.text_input("(Optional) Prefer matches from this domain", placeholder="dogvills.com")
    k2 = st.slider("How many suggestions? ", 3, 20, 8, key="k2")
    disabled2 = len(rows)==0 or not t_text.strip()
    if len(rows)==0:
        st.info("Index is empty. Build the index at the bottom first.")
    if st.button("Find related (by text)", disabled=disabled2):
        results = rank_related(t_title, t_text, rows, pack, same_domain_only=(prefer_domain or None))
        table = [{
            "Score": round(r["score"],3),
            "Title": r["title"],
            "URL": r["url"],
            "Suggested Anchor": r["anchor"],
            "Why": r["why"]
        } for r in results[:k2]]
        st.dataframe(table, use_container_width=True)

# 4) Big Build button AT THE BOTTOM
st.markdown("---")
st.subheader("4) Build or refresh the index")
build_clicked = st.button("🔄 BUILD / REFRESH INDEX", type="primary", use_container_width=True, disabled=(len(selected_sites)==0))

if build_clicked:
    if not selected_sites:
        st.error("Pick at least one site.")
    else:
        build_index(
            selected_sites=selected_sites,
            use_cache_sitemaps=use_cache_sitemaps,
            check_robots=check_robots,
            include_gz_candidates=include_gz_candidates,
            max_pages_per_site=max_pages_per_site,
            max_workers=max_workers,
            exclude_common=exclude_common,
            extra_excludes_text=extra_excludes,
            force_refresh=force_refresh,
        )
        # Rerun so the new index count and tabs enable immediately
        st.rerun()

st.markdown("---")
st.caption("Scoring = 0.70×TF‑IDF + 0.20×ontology(candidate) + 0.10×ontology(target) − species mismatch penalty.")
