from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import hstack

MODELS_DIR = Path(__file__).parent.parent / "models"

_REQUIRED = [
    "classifier.joblib",
    "tfidf.joblib",
    "scaler_emb.joblib",
    "scaler_final.joblib",
    "mlb.joblib",
    "threshold_vec.npy",
]


def models_exist() -> bool:
    return all((MODELS_DIR / f).exists() for f in _REQUIRED)


def load_artifacts() -> dict:
    return {
        "clf": joblib.load(MODELS_DIR / "classifier.joblib"),
        "tfidf": joblib.load(MODELS_DIR / "tfidf.joblib"),
        "scaler_emb": joblib.load(MODELS_DIR / "scaler_emb.joblib"),
        "scaler_final": joblib.load(MODELS_DIR / "scaler_final.joblib"),
        "mlb": joblib.load(MODELS_DIR / "mlb.joblib"),
        "threshold_vec": np.load(MODELS_DIR / "threshold_vec.npy"),
    }


def predict_genres(
    text: str,
    artifacts: dict,
    nlp,
    emb_model,
    min_ratio: float = 0.80,
    max_labels: int = 3,
) -> dict:
    """Run the full hybrid pipeline and return probabilities + predicted genres."""
    from src.preprocessing import preprocess_text

    lemmas = preprocess_text(text, nlp)

    emb = emb_model.encode([text])
    emb_scaled = artifacts["scaler_emb"].transform(emb)

    tfidf_vec = artifacts["tfidf"].transform([lemmas])

    combined = hstack([emb_scaled, tfidf_vec])
    combined_scaled = artifacts["scaler_final"].transform(combined)

    probs = artifacts["clf"].predict_proba(combined_scaled)[0]
    classes = artifacts["mlb"].classes_
    threshold_vec = artifacts["threshold_vec"]

    # Replicate decode_row from the notebook
    active = np.where(probs >= threshold_vec)[0].tolist()
    top1 = int(np.argmax(probs))
    top1_prob = probs[top1]

    if not active:
        active = [top1]
    elif top1 not in active:
        active = [top1] + active

    filtered = [
        idx for idx in active
        if idx == top1 or probs[idx] >= top1_prob * min_ratio
    ]
    filtered = sorted(filtered, key=lambda i: probs[i], reverse=True)[:max_labels]

    return {
        "probabilities": dict(zip(classes, probs.tolist())),
        "predicted": [classes[i] for i in filtered],
        "thresholds": dict(zip(classes, threshold_vec.tolist())),
        "lemmas": lemmas,
    }
