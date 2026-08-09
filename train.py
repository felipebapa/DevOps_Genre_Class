"""
Training script — run once to generate model artifacts in models/.
Usage: python train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
from scipy.sparse import hstack

from src.preprocessing import load_spacy_model, preprocess_text

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DATA_PATH = Path("notebooks/KLUSTERS.xlsx")

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(DATA_PATH, engine="openpyxl")
df = df[["plot", "genres"]].dropna().reset_index(drop=True)
print(f"  {len(df)} rows loaded")

# ── Preprocess ─────────────────────────────────────────────────────────────
print("Loading spaCy model...")
nlp = load_spacy_model()

print("Preprocessing plots (this may take several minutes)...")
df["plot_lemmas"] = df["plot"].astype(str).apply(lambda x: preprocess_text(x, nlp))

def parse_genres(x):
    if isinstance(x, str):
        return eval(x)
    return x

df["genres"] = df["genres"].apply(parse_genres)

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["genres"])
print(f"  Genres: {list(mlb.classes_)}")

# ── Split 70 / 15 / 15 ────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    df["plot"], y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)
X_lemmas_train = df.loc[X_train.index, "plot_lemmas"]
X_lemmas_val   = df.loc[X_val.index,   "plot_lemmas"]
X_lemmas_test  = df.loc[X_test.index,  "plot_lemmas"]
print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

# ── Embeddings ─────────────────────────────────────────────────────────────
print("Encoding embeddings with all-MiniLM-L6-v2...")
from sentence_transformers import SentenceTransformer
emb_model = SentenceTransformer("all-MiniLM-L6-v2")
X_train_emb = emb_model.encode(X_train.tolist(), show_progress_bar=True)
X_val_emb   = emb_model.encode(X_val.tolist(),   show_progress_bar=True)
X_test_emb  = emb_model.encode(X_test.tolist(),  show_progress_bar=True)

# ── TF-IDF ─────────────────────────────────────────────────────────────────
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.90,
    sublinear_tf=True,
)
X_train_tfidf = tfidf.fit_transform(X_lemmas_train)
X_val_tfidf   = tfidf.transform(X_lemmas_val)
X_test_tfidf  = tfidf.transform(X_lemmas_test)

# ── Scale and stack ────────────────────────────────────────────────────────
print("Scaling and stacking features...")
scaler_emb = StandardScaler(with_mean=False)
X_train_emb = scaler_emb.fit_transform(X_train_emb)
X_val_emb   = scaler_emb.transform(X_val_emb)
X_test_emb  = scaler_emb.transform(X_test_emb)

X_train_final = hstack([X_train_emb, X_train_tfidf])
X_val_final   = hstack([X_val_emb,   X_val_tfidf])
X_test_final  = hstack([X_test_emb,  X_test_tfidf])

scaler_final = MaxAbsScaler()
X_train_final = scaler_final.fit_transform(X_train_final)
X_val_final   = scaler_final.transform(X_val_final)
X_test_final  = scaler_final.transform(X_test_final)

# ── Train classifier ───────────────────────────────────────────────────────
print("Training OneVsRest LogisticRegression...")
clf = OneVsRestClassifier(
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        C=2.0,
        solver="liblinear",
    )
)
clf.fit(X_train_final, y_train)

# ── Threshold tuning ───────────────────────────────────────────────────────
print("Tuning thresholds on validation set...")
y_val_proba = clf.predict_proba(X_val_final)
search_thresholds = np.arange(0.10, 0.80, 0.05)
classes = mlb.classes_
threshold_map = {}

for i, genre in enumerate(classes):
    best_thr, best_f1 = 0.5, 0.0
    for thr in search_thresholds:
        pred_i = (y_val_proba[:, i] >= thr).astype(int)
        f1 = f1_score(y_val[:, i], pred_i, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    threshold_map[genre] = best_thr

# Manual overrides matching the notebook
for g in ["Sci-Fi", "Short", "Horror", "Fantasy"]:
    if g in threshold_map:
        threshold_map[g] = max(threshold_map[g], 0.50)
if "Drama" in threshold_map:
    threshold_map["Drama"] = min(threshold_map["Drama"], 0.30)

threshold_vec = np.array([threshold_map[g] for g in classes])

# ── Evaluate on test ───────────────────────────────────────────────────────
print("\nTest set metrics:")
y_test_proba = clf.predict_proba(X_test_final)

def decode_row(probs, thr_vec, min_ratio=0.80, max_labels=3):
    active = np.where(probs >= thr_vec)[0].tolist()
    top1 = int(np.argmax(probs))
    top1_prob = probs[top1]
    if not active:
        return [top1]
    if top1 not in active:
        active = [top1] + active
    filtered = [i for i in active if i == top1 or probs[i] >= top1_prob * min_ratio]
    return sorted(filtered, key=lambda i: probs[i], reverse=True)[:max_labels]

pred_rows = []
for i in range(len(y_test_proba)):
    idxs = decode_row(y_test_proba[i], threshold_vec)
    row = np.zeros(len(classes), dtype=int)
    row[idxs] = 1
    pred_rows.append(row)

y_pred = np.vstack(pred_rows)
print(classification_report(y_test, y_pred, target_names=classes))

# ── Save artifacts ─────────────────────────────────────────────────────────
print("Saving artifacts to models/...")
joblib.dump(clf,          MODELS_DIR / "classifier.joblib")
joblib.dump(tfidf,        MODELS_DIR / "tfidf.joblib")
joblib.dump(scaler_emb,   MODELS_DIR / "scaler_emb.joblib")
joblib.dump(scaler_final, MODELS_DIR / "scaler_final.joblib")
joblib.dump(mlb,          MODELS_DIR / "mlb.joblib")
np.save(MODELS_DIR / "threshold_vec.npy", threshold_vec)

print("Done! Run: streamlit run app.py")
