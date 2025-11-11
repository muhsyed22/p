# model.py – FINAL WORKING VERSION (Nov 11, 2025)
import os
import json
import gzip
import random
import numpy as np
import nltk
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests   # ← NEW: no gdown needed

print("Downloading NLTK data...")
nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet',   quiet=True)
nltk.download('omw-1.4',   quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

DATA_PATH  = "data/clothing.json.gz"
MODEL_PATH = "word2vec.model"
VECS_PATH  = "vectors.npy"
PKL_PATH   = "sampled_products.pkl"

# ←←←←← YOUR GOOGLE DRIVE FILE ID (already correct) ←←←←←
FILE_ID = "1gXAC70uJLSC5pzccsxxy2MiPwP5VOv5N"

# ============ AUTO-DOWNLOAD MODEL IF MISSING ============
if not os.path.exists(MODEL_PATH):
    print("Model not found → Downloading word2vec.model from Google Drive (one-time only)...")
    url = f"https://drive.google.com/uc?id={FILE_ID}&export=download&confirm=t"
    r = requests.get(url, stream=True)
    r.raise_for_status()
    
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            if chunk:
                f.write(chunk)
    print("Download complete! Model saved as word2vec.model")

# ============ LOAD THE MODEL ============
print("Loading Word2Vec model...")
model = Word2Vec.load(MODEL_PATH)
print("Model loaded successfully!")

# ============ CHECK IF ALREADY PROCESSED ============
if os.path.exists(VECS_PATH) and os.path.exists(PKL_PATH):
    print("Pre-computed vectors & products found → skipping training")
    product_vectors = np.load(VECS_PATH)
    import pickle
    with open(PKL_PATH, "rb") as f:
        products = pickle.load(f)
    print(f"Loaded {len(products):,} pre-processed products")
else:
    # ============ STREAM & SAMPLE DATA ============
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Put clothing.json.gz in the 'data' folder!")

    TARGET_SAMPLE = 100_000
    products = []

    print(f"Streaming and sampling {TARGET_SAMPLE:,} products...")
    with gzip.open(DATA_PATH, 'rt', encoding='utf-8') as f:
        for line in tqdm(f, desc="Sampling", unit="line"):
            if random.random() < (TARGET_SAMPLE / (len(products) + 1)):
                try:
                    data = json.loads(line)
                    products.append(data)
                    if len(products) >= TARGET_SAMPLE:
                        break
                except:
                    continue

    print(f"Sampled {len(products):,} products")

    # ============ PREPROCESSING ============
    def preprocess(text: str):
        if not text or not str(text).strip():
            return []
        tokens = nltk.word_tokenize(str(text).lower())
        return [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]

    def full_text(row):
        parts = []
        if row.get("title"): parts.append(str(row["title"]))
        desc = row.get("description", "")
        if isinstance(desc, list): desc = " ".join(desc)
        if desc: parts.append(str(desc))
        if row.get("feature"): 
            feats = row["feature"]
            if isinstance(feats, list): parts.append(" ".join(feats))
        return " ".join(parts)

    print("Building corpus...")
    corpus = [preprocess(full_text(p)) for p in tqdm(products, desc="Tokenizing", unit="product")]

    # ============ TRAIN WORD2VEC (only if needed) ============
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4,
        epochs=10
    )
    model.save(MODEL_PATH)
    print(f"New model trained & saved → {MODEL_PATH}")

    # ============ GENERATE VECTORS ============
    def get_vector(text):
        toks = preprocess(text)
        if not toks: return np.zeros(100)
        vecs = [model.wv[t] for t in toks if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(100)

    print("Generating product vectors...")
    product_vectors = np.array([get_vector(full_text(p)) for p in tqdm(products, desc="Vectorising")])
    
    np.save(VECS_PATH, product_vectors)
    print(f"Vectors saved → {VECS_PATH}")
    
    import pickle
    with open(PKL_PATH, "wb") as f:
        pickle.dump(products, f)
    print(f"Products saved → {PKL_PATH}")

print("model.py finished – Everything ready for app.py!")