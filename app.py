# app.py
import os
import pickle
import numpy as np
import streamlit as st
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# ------------------- CONFIG -------------------
MODEL_PATH = "word2vec.model"
VECS_PATH  = "vectors.npy"
PKL_PATH   = "sampled_products.pkl"

# ------------------- LOAD ARTIFACTS -------------------
@st.cache_resource
def load_artifacts():
    # 1. Model
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()
    model = Word2Vec.load(MODEL_PATH)

    # 2. Vectors
    if not os.path.exists(VECS_PATH):
        st.error(f"Vectors not found: {VECS_PATH}")
        st.stop()
    vectors = np.load(VECS_PATH)               # shape: (N, 100)

    # 3. Product list (with original JSON rows)
    if not os.path.exists(PKL_PATH):
        st.error(f"Sampled products not found: {PKL_PATH}")
        st.stop()
    with open(PKL_PATH, "rb") as f:
        products = pickle.load(f)

    return model, vectors, products

model, vectors, products = load_artifacts()

# ------------------- HELPERS -------------------
def product_to_text(p):
    """Same logic you used in model.py – title + description + features"""
    parts = []
    if p.get("title"): parts.append(str(p["title"]))
    desc = p.get("description", "")
    if isinstance(desc, list): desc = " ".join(desc)
    if desc: parts.append(str(desc))
    if "feature" in p and isinstance(p["feature"], list):
        parts.append(" ".join(p["feature"]))
    return " ".join(parts)

def vectorize_query(query: str) -> np.ndarray:
    """Turn a free-text query into a 100-dim vector (same preprocessing)"""
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    toks = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(query)
            if w.isalpha() and w.lower() not in stop_words]
    if not toks:
        return np.zeros(100)

    vecs = [model.wv[t] for t in toks if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

# ------------------- UI -------------------
st.title("Clothing Product Search (Word2Vec)")

query = st.text_input("Enter a product description or keywords", "red summer dress with pockets")

if query:
    q_vec = vectorize_query(query).reshape(1, -1)
    # Cosine similarity against *all* product vectors
    sims = cosine_similarity(q_vec, vectors).flatten()
    top_k = 5
    top_idx = np.argsort(sims)[-top_k:][::-1]

    st.write(f"**Top {top_k} matches** (cosine similarity):")
    for rank, idx in enumerate(top_idx, 1):
        prod = products[idx]
        sim = sims[idx]
        with st.expander(f"#{rank} – {prod.get('title', 'Untitled')} (sim={sim:.3f})"):
            st.write("**Title:**", prod.get("title"))
            st.write("**Description:**", prod.get("description"))
            st.write("**Features:**", prod.get("feature"))
            # optional: show image if you have URLs
            if prod.get("imageURL"):
                st.image(prod["imageURL"], width=200)