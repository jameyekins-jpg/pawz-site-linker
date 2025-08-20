# site_linker.py
# Ultra‑light build: pure‑Python TF‑IDF (no sklearn/numpy)
import os, pickle, math, re
from typing import List, Tuple, Dict, Optional

import streamlit as st
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

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
        "synonyms": ["mobility","joint health","hip dysplasia","arthritis","stiffness","lameness","pain","recover","rehab","physical therapy","cartilage","inflammation","range of motion"],
        "solutions": ["ramp","steps","stairs","harness","lift harness","traction","non-slip","booties","low-impact exercise","hydrotherapy","massage","acupuncture","orthopedic"],
        "supplements": ["glucosamine","chondroitin","msm","omega-3","fish oil","turmeric","curcumin","green lipped mussel","hyaluronic acid","collagen"]
    },
    "senior": {"synonyms": ["senior","aging","older","geriatric","longevity","age-related"]},
    "species": {"dog": ["dog","dogs","canine"], "cat": ["cat","cats","feline"]}
}

CACHE_DIR="data"; ARTICLES_PKL=os.path.join(CACHE_DIR,"articles.pkl"); LITE_PKL=os.path.join(CACHE_DIR,"lite_index.pkl")
REQUEST_TIMEOUT=20; MAX_URLS_PER_SITEMAP=5000
os.makedirs(CACHE_DIR, exist_ok=True)
STOPWORDS=set("a about above after again against all am an and any are as at be because been before being below between both but by can did do does doing down during each few for from further had has have having he her here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not of off on once only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves then there these they this those through to too under until up very was we were what when where which while who whom why with you your yours yourself yourselves".split())

def user_agent(): return {"User-Agent":"Mozilla/5.0 (compatible; SiteLinker/ultra/1.0)"}
def fetch(url:str):
    try:
        r=requests.get(url,timeout=REQUEST_TIMEOUT,headers=user_agent())
        if r.status_code==200: return r.content
    except Exception: pass
    return None

def discover_sitemaps():
    smaps=[]
    for base in SITES:
        for path in SITEMAP_CANDIDATES:
            sm_url=base.rstrip("/")+path
            content=fetch(sm_url)
            if not content: continue
            head=content[:2000].decode("utf-8","ignore").lower()
            if "<urlset" in head or "<sitemapindex" in head:
                smaps.append(sm_url); break
    return smaps

def parse_sitemap_xml(xml_bytes:bytes):
    urls,children=[],[]
    try:
        root=ET.fromstring(xml_bytes.decode("utf-8","ignore"))
        def ns_tag(name):
            if "}" in root.tag:
                ns=root.tag.split("}")[0].strip("{"); return f"{{{ns}}}{name}"
            return name
        if root.tag.endswith("sitemapindex"):
            for sm in root.findall(ns_tag("sitemap")):
                loc=sm.find(ns_tag("loc"))
                if loc is not None and loc.text: children.append(loc.text.strip())
        elif root.tag.endswith("urlset"):
            for u in root.findall(ns_tag("url")):
                loc=u.find(ns_tag("loc"))
                if loc is not None and loc.text: urls.append(loc.text.strip())
    except Exception: pass
    urls=[u for u in urls if not any(x in u for x in ["/image_sitemap","/video_sitemap"])]
    return urls,children

def collect_urls_from_sitemap(start_url:str):
    seen=set(); to_visit=[start_url]; all_urls=[]
    while to_visit:
        sm_url=to_visit.pop()
        if sm_url in seen: continue
        seen.add(sm_url)
        xml=fetch(sm_url)
        if not xml: continue
        urls,children=parse_sitemap_xml(xml)
        all_urls.extend(urls[:MAX_URLS_PER_SITEMAP])
        to_visit.extend(children)
        if len(all_urls)>=MAX_URLS_PER_SITEMAP: break
    return list(dict.fromkeys(all_urls))

def smart_extract_text(html_bytes:bytes):
    html=html_bytes.decode("utf-8","ignore")
    soup=BeautifulSoup(html,"html5lib")
    title=(soup.title.string.strip() if soup.title and soup.title.string else "")
    h1=soup.find("h1")
    if h1 and h1.get_text(strip=True): title=h1.get_text(strip=True)
    main_node=soup.find("article") or soup.find("main")
    if not main_node:
        candidates=soup.find_all(["div","section"], class_=True); best=None; best_len=0
        for c in candidates:
            cls=" ".join(c.get("class",[])).lower()
            if any(k in cls for k in ["content","entry-content","post-content","article-content","single-post"]):
                txt=c.get_text(" ",strip=True)
                if len(txt)>best_len: best=c; best_len=len(txt)
        main_node=best if best is not None else soup.body
    text=main_node.get_text(" ",strip=True) if main_node else soup.get_text(" ",strip=True)
    preview=text[:600]+("…" if len(text)>600 else "")
    return title or "(untitled)", text, preview

def extract_main_content(url:str):
    html=fetch(url)
    if not html: return None,None,None
    try: return smart_extract_text(html)
    except Exception: return None,None,None

_token_re=re.compile(r"[a-z0-9]+")
def tokenize(text:str):
    if not text: return []
    return [w for w in _token_re.findall(text.lower()) if w not in STOPWORDS and len(w)>2]
def tokens_with_bigrams(tokens:List[str]):
    if len(tokens)<2: return tokens
    bigrams=[tokens[i]+"_"+tokens[i+1] for i in range(len(tokens)-1)]
    return tokens + bigrams
def text_for_index(title,text): return ((title or "")+"\n\n"+(text or "")[:2000]).strip()

def build_lite_index(rows:List[Dict]):
    docs_tokens=[]; df={}; N=len(rows)
    for r in rows:
        toks=tokens_with_bigrams(tokenize(text_for_index(r["title"], r["text"])))
        docs_tokens.append(toks)
        for t in set(toks): df[t]=df.get(t,0)+1
    idf={t:(math.log((1+N)/(1+c))+1.0) for t,c in df.items()}
    doc_vectors=[]; doc_norms=[]
    for toks in docs_tokens:
        tf={}
        for t in toks: tf[t]=tf.get(t,0)+1
        vec={}
        for t,f in tf.items():
            w=(1.0+math.log(f))*idf.get(t,0.0)
            if w>0: vec[t]=w
        norm=math.sqrt(sum(w*w for w in vec.values())) or 1.0
        doc_vectors.append(vec); doc_norms.append(norm)
    pack={"idf":idf,"doc_vectors":doc_vectors,"doc_norms":doc_norms,"N":N}
    with open(LITE_PKL,"wb") as f: pickle.dump(pack,f)

def load_index():
    if not (os.path.exists(ARTICLES_PKL) and os.path.exists(LITE_PKL)): return [], None
    with open(ARTICLES_PKL,"rb") as f: rows=pickle.load(f)
    with open(LITE_PKL,"rb") as f: pack=pickle.load(f)
    return rows, pack
def save_rows(rows): 
    with open(ARTICLES_PKL,"wb") as f: pickle.dump(rows,f)

def cosine_sim_sparse(qvec,qnorm,dvec,dnorm):
    dot=0.0
    for t,qw in qvec.items():
        dw=dvec.get(t)
        if dw is not None: dot+=qw*dw
    return dot/(qnorm*dnorm) if qnorm and dnorm else 0.0
def make_query_vector(title,text,idf):
    toks=tokens_with_bigrams(tokenize(text_for_index(title,text)))
    tf={}
    for t in toks: tf[t]=tf.get(t,0)+1
    qvec={}
    for t,f in tf.items():
        w=(1.0+math.log(f))*idf.get(t,0.0)
        if w>0: qvec[t]=w
    qnorm=math.sqrt(sum(w*w for w in qvec.values())) or 1.0
    return qvec,qnorm

def ontology_overlap_score(text:str):
    t=(text or "").lower(); score=0.0; detail=[]
    def count_any(terms,label):
        c=sum(t.count(term.lower()) for term in terms)
        if c>0: detail.append(f"{label}:{c}")
        return c
    if "mobility" in ONTOLOGY:
        c1=count_any(ONTOLOGY["mobility"]["synonyms"],"mobility")
        c2=count_any(ONTOLOGY["mobility"]["solutions"],"aids")
        c3=count_any(ONTOLOGY["mobility"]["supplements"],"supps")
        score+=0.5*min(c1,5)+0.3*min(c2,5)+0.2*min(c3,5)
    if "senior" in ONTOLOGY:
        c4=count_any(ONTOLOGY["senior"]["synonyms"],"senior")
        score+=0.3*min(c4,5)
    return score, ", ".join(detail[:4])

def species_penalty(a_text,b_text):
    get=lambda txt,keys: sum((txt or "").lower().count(k) for k in keys)
    dA=get(a_text, ONTOLOGY["species"]["dog"]); cA=get(a_text, ONTOLOGY["species"]["cat"])
    dB=get(b_text, ONTOLOGY["species"]["dog"]); cB=get(b_text, ONTOLOGY["species"]["cat"])
    if dA>cA and cB>dB: return 0.07
    if cA>dA and dB>cB: return 0.07
    return 0.0

def suggest_anchor(text:str):
    t=(text or "").lower(); anchors=[]
    if "ramp" in t: anchors.append("dog ramps for seniors")
    if "glucosamine" in t or "supplement" in t: anchors.append("joint supplements for mobility")
    if "exercise" in t or "low-impact" in t: anchors.append("low-impact exercise for older dogs")
    if "hip dysplasia" in t: anchors.append("hip dysplasia support")
    if not anchors: anchors.append("help for senior dog mobility")
    return ", ".join(anchors[:2])

def rank_related(target_title,target_text,rows,pack,same_domain_only=None,min_words=200):
    if not rows or not pack: return []
    idf=pack["idf"]; qvec,qnorm=make_query_vector(target_title,target_text,idf)
    out=[]
    for i,r in enumerate(rows):
        if len((r["text"] or "").split())<min_words: continue
        if same_domain_only and r["domain"]!=same_domain_only: continue
        if r["text"]==target_text: continue
        dvec=pack["doc_vectors"][i]; dnorm=pack["doc_norms"][i]
        sim=cosine_sim_sparse(qvec,qnorm,dvec,dnorm)
        ont_cand, ont_detail = ontology_overlap_score(r["text"])
        ont_tgt, _ = ontology_overlap_score(target_text)
        penalty=species_penalty(target_text, r["text"])
        final=0.70*float(sim)+0.20*(min(ont_cand,5.0)/5.0)+0.10*(min(ont_tgt,5.0)/5.0)-penalty
        out.append({"title":r["title"],"url":r["url"],"domain":r["domain"],"preview":r["preview"],"sim":round(float(sim),4),"why":f"tfidf:{round(float(sim),3)} | {ont_detail}","anchor":suggest_anchor(r["text"]),"score":final})
    out.sort(key=lambda x:x["score"], reverse=True)
    return out

def build_index(sitemap_urls:List[str]):
    all_urls=[]
    st.info("Collecting URLs from sitemaps…")
    for sm in sitemap_urls:
        st.write(f"Scanning: {sm}")
        urls=collect_urls_from_sitemap(sm); st.write(f"→ Found {len(urls)} URLs")
        all_urls.extend(urls)
    all_urls=list(dict.fromkeys(all_urls)); rows=[]
    if not all_urls: st.warning("No URLs discovered. Check sitemap endpoints."); return
    progress=st.progress(0.0); total=len(all_urls)
    for i,url in enumerate(all_urls,1):
        title,text,preview=extract_main_content(url)
        if not text or len(text)<200: progress.progress(i/total); continue
        rows.append({"url":url,"title":title or "(untitled)","text":text,"preview":preview,"domain":urlparse(url).netloc})
        progress.progress(i/total)
    if not rows: st.warning("No articles were indexed (pages too short or blocked)."); return
    save_rows(rows); build_lite_index(rows); st.success(f"Indexed {len(rows)} articles.")

st.set_page_config(page_title="Site Linker (Ultra‑light)", layout="wide")
st.title("🔗 Site Linker — Cross‑Site Related Article Finder")
st.caption("Ultra‑light build: pure‑Python TF‑IDF + pet ontology. Minimal dependencies.")

with st.expander("⚙️ Index controls"):
    col1,col2,col3 = st.columns([1,1,2])
    with col1:
        if st.button("🔄 Build / Refresh Index", type="primary"):
            sitemaps=discover_sitemaps()
            if not sitemaps: st.error("Couldn’t find any sitemaps. Check that the sites expose /sitemap_index.xml or /sitemap.xml")
            else:
                build_index(sitemaps); st.experimental_rerun()
    with col2:
        if st.button("🗑 Clear Cache Files"):
            for p in [ARTICLES_PKL, LITE_PKL]:
                if os.path.exists(p): os.remove(p)
            st.success("Cache cleared. Click refresh to rebuild.")
    with col3: st.write("**Tip:** Refresh after new posts or weekly. Searches stay fast between refreshes.")

rows, pack = load_index()
st.write(f"**Indexed articles:** {len(rows)}")

tab1, tab2 = st.tabs(["🔎 Find by URL", "📝 Find by pasted text"])
with tab1:
    url = st.text_input("Article URL to analyze", placeholder="https://dogvills.com/why-dogs-lose-mobility-as-they-age/")
    prefer_same_site = st.checkbox("Prefer same-site suggestions first", value=True)
    k = st.slider("How many suggestions?", 3, 20, 8)
    if st.button("Find related (by URL)", disabled=(len(rows)==0)):
        try:
            t_title, t_text, _ = extract_main_content(url)
            if not t_text: st.error("Couldn’t extract text from that URL.")
            else:
                domain=urlparse(url).netloc; domain_filter = domain if prefer_same_site else None
                results=rank_related(t_title, t_text, rows, pack, same_domain_only=domain_filter)
                if prefer_same_site:
                    same=[r for r in results if r["domain"]==domain]; cross=[r for r in results if r["domain"]!=domain]
                    results = same[:k] + cross[:k]
                table=[{"Score":round(r["score"],3),"Title":r["title"],"URL":r["url"],"Suggested Anchor":r["anchor"],"Why":r["why"]} for r in results[:k]]
                st.dataframe(table, use_container_width=True)
        except Exception as e:
            st.error(str(e))

with tab2:
    t_title = st.text_input("(Optional) Title for your draft/article")
    t_text = st.text_area("Paste article text (intro is fine)", height=220)
    prefer_domain = st.text_input("(Optional) Prefer matches from this domain", placeholder="dogvills.com")
    k2 = st.slider("How many suggestions? ", 3, 20, 8, key="k2")
    if st.button("Find related (by text)", disabled=(len(rows)==0 or not t_text.strip())):
        results=rank_related(t_title, t_text, rows, pack, same_domain_only=(prefer_domain or None))
        table=[{"Score":round(r["score"],3),"Title":r["title"],"URL":r["url"],"Suggested Anchor":r["anchor"],"Why":r["why"]} for r in results[:k2]]
        st.dataframe(table, use_container_width=True)

st.markdown("---")
st.caption("Scoring = 0.70×TF‑IDF + 0.20×ontology(candidate) + 0.10×ontology(target) − species mismatch penalty.")
