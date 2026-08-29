"""
Training script — entrena el pipeline híbrido, deja trazabilidad completa en
MLflow y registra el modelo en el Model Registry.

Uso:
    python train.py                          # entrena + loguea + registra versión nueva
    python train.py --promote-alias champion # además la marca como la que despliega el CD
    python train.py --no-mlflow              # entrena sin tocar MLflow (solo models/)

El tracking server se toma de MLFLOW_TRACKING_URI (por defecto http://localhost:5000).
"""
import argparse
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# La consola de Windows usa cp1252 y MLflow imprime emojis en sus mensajes de
# estado; sin esto el entrenamiento revienta con UnicodeEncodeError al cerrar
# la corrida, después de haber hecho todo el trabajo.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MaxAbsScaler, MultiLabelBinarizer, StandardScaler

from src.preprocessing import load_spacy_model, preprocess_text

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
DATA_PATH = Path("notebooks/KLUSTERS.xlsx")

# ── Hiperparámetros (centralizados para poder loguearlos) ──────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_TEST_SPLIT = 0.50
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TFIDF_PARAMS = {
    "max_features": 30000,
    "ngram_range": (1, 3),
    "min_df": 3,
    "max_df": 0.90,
    "sublinear_tf": True,
}
CLF_PARAMS = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "C": 2.0,
    "solver": "liblinear",
}
DECODE_MIN_RATIO = 0.80
DECODE_MAX_LABELS = 3
THRESHOLD_SEARCH = (0.10, 0.80, 0.05)
THRESHOLD_FLOOR_GENRES = ["Sci-Fi", "Short", "Horror", "Fantasy"]
THRESHOLD_FLOOR = 0.50
THRESHOLD_CAP_GENRES = ["Drama"]
THRESHOLD_CAP = 0.30


# ── CLI ────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Entrena el clasificador de géneros.")
    p.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Entrena sin registrar nada en MLflow (solo escribe models/).",
    )
    p.add_argument(
        "--no-register",
        action="store_true",
        help="Loguea la corrida pero no crea una versión en el Model Registry.",
    )
    p.add_argument(
        "--promote-alias",
        default=os.getenv("MLFLOW_PROMOTE_ALIAS", ""),
        help="Alias a asignar a la versión creada (p.ej. 'champion'). Vacío = no promover.",
    )
    p.add_argument(
        "--run-name",
        default=os.getenv("MLFLOW_RUN_NAME", ""),
        help="Nombre de la corrida en MLflow.",
    )
    p.add_argument(
        "--model-name",
        default=os.getenv("MLFLOW_MODEL_NAME", "genre-classifier"),
        help="Nombre del modelo registrado.",
    )
    p.add_argument(
        "--experiment",
        default=os.getenv("MLFLOW_EXPERIMENT_NAME", "genre-classification"),
        help="Nombre del experimento en MLflow.",
    )
    return p.parse_args()


args = parse_args()
USE_MLFLOW = not args.no_mlflow

if USE_MLFLOW:
    import mlflow
    from mlflow.models import ModelSignature
    from mlflow.tracking import MlflowClient
    from mlflow.types import ColSpec, Schema

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)

    # Primer contacto con el servidor. Se comprueba aquí, antes de cargar spaCy
    # y el dataset: un servidor inalcanzable suelta un traceback de urllib3 de
    # cien líneas que no dice ni contra qué URI se estaba intentando.
    try:
        mlflow.set_experiment(args.experiment)
    except Exception as exc:  # noqa: BLE001 - cualquier fallo aqui es de conexion
        print(f"\nERROR: no hay un servidor MLflow escuchando en {tracking_uri}", file=sys.stderr)
        print("\nRevisa, en este orden:", file=sys.stderr)
        print("  1. Que MLFLOW_TRACKING_URI apunte a donde crees.", file=sys.stderr)
        print(f"     Valor actual: {tracking_uri}", file=sys.stderr)
        if os.getenv("MLFLOW_TRACKING_URI"):
            print("     Viene de la variable de entorno. Para usar el servidor local:", file=sys.stderr)
            print('       PowerShell:  $env:MLFLOW_TRACKING_URI = "http://localhost:5000"', file=sys.stderr)
            print("       bash:        export MLFLOW_TRACKING_URI=http://localhost:5000", file=sys.stderr)
        print("  2. Que el servidor esté arriba:  docker compose up -d mlflow", file=sys.stderr)
        print("  3. Si solo quieres entrenar, sin registrar nada:", file=sys.stderr)
        print("       python train.py --no-mlflow", file=sys.stderr)
        print(f"\nDetalle: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment:   {args.experiment}")

    run = mlflow.start_run(run_name=args.run_name or None)
    print(f"MLflow run_id:       {run.info.run_id}\n")


def log_params(params: dict):
    if USE_MLFLOW:
        mlflow.log_params(params)


def log_metrics(metrics: dict):
    if USE_MLFLOW:
        mlflow.log_metrics(metrics)


def sanitize(name: str) -> str:
    """MLflow solo acepta [alfanumérico _ - . espacio /] en nombres de métrica."""
    return "".join(c if (c.isalnum() or c in "_-. /") else "_" for c in name)


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

log_params(
    {
        "data_path": str(DATA_PATH),
        "n_rows": len(df),
        "n_labels": len(mlb.classes_),
        "labels": ", ".join(mlb.classes_),
        "embedding_model": EMBEDDING_MODEL,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_test_split": VAL_TEST_SPLIT,
        "decode_min_ratio": DECODE_MIN_RATIO,
        "decode_max_labels": DECODE_MAX_LABELS,
        "threshold_search": str(THRESHOLD_SEARCH),
        "threshold_floor": THRESHOLD_FLOOR,
        "threshold_cap": THRESHOLD_CAP,
        "python_version": platform.python_version(),
        **{f"tfidf_{k}": str(v) for k, v in TFIDF_PARAMS.items()},
        **{f"clf_{k}": str(v) for k, v in CLF_PARAMS.items()},
    }
)

# ── Split 70 / 15 / 15 ────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    df["plot"], y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=VAL_TEST_SPLIT, random_state=RANDOM_STATE
)
X_lemmas_train = df.loc[X_train.index, "plot_lemmas"]
X_lemmas_val   = df.loc[X_val.index,   "plot_lemmas"]
X_lemmas_test  = df.loc[X_test.index,  "plot_lemmas"]
print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

log_params(
    {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
    }
)

# ── Embeddings ─────────────────────────────────────────────────────────────
print(f"Encoding embeddings with {EMBEDDING_MODEL}...")
from sentence_transformers import SentenceTransformer

emb_model = SentenceTransformer(EMBEDDING_MODEL)
X_train_emb = emb_model.encode(X_train.tolist(), show_progress_bar=True)
X_val_emb   = emb_model.encode(X_val.tolist(),   show_progress_bar=True)
X_test_emb  = emb_model.encode(X_test.tolist(),  show_progress_bar=True)

# ── TF-IDF ─────────────────────────────────────────────────────────────────
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(**TFIDF_PARAMS)
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

log_params(
    {
        "n_features_total": X_train_final.shape[1],
        "n_features_embedding": X_train_emb.shape[1],
        "n_features_tfidf": X_train_tfidf.shape[1],
    }
)

# ── Train classifier ───────────────────────────────────────────────────────
print("Training OneVsRest LogisticRegression...")
clf = OneVsRestClassifier(LogisticRegression(**CLF_PARAMS))
clf.fit(X_train_final, y_train)

# ── Threshold tuning ───────────────────────────────────────────────────────
print("Tuning thresholds on validation set...")
y_val_proba = clf.predict_proba(X_val_final)
search_thresholds = np.arange(*THRESHOLD_SEARCH)
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
for g in THRESHOLD_FLOOR_GENRES:
    if g in threshold_map:
        threshold_map[g] = max(threshold_map[g], THRESHOLD_FLOOR)
for g in THRESHOLD_CAP_GENRES:
    if g in threshold_map:
        threshold_map[g] = min(threshold_map[g], THRESHOLD_CAP)

threshold_vec = np.array([threshold_map[g] for g in classes])

# ── Evaluate on test ───────────────────────────────────────────────────────
print("\nTest set metrics:")
y_test_proba = clf.predict_proba(X_test_final)

def decode_row(probs, thr_vec, min_ratio=DECODE_MIN_RATIO, max_labels=DECODE_MAX_LABELS):
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
report_text = classification_report(y_test, y_pred, target_names=classes)
print(report_text)

# ── Métricas agregadas y por género ────────────────────────────────────────
metrics = {
    "f1_micro": f1_score(y_test, y_pred, average="micro", zero_division=0),
    "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "f1_weighted": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "f1_samples": f1_score(y_test, y_pred, average="samples", zero_division=0),
    "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
    "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    "precision_micro": precision_score(y_test, y_pred, average="micro", zero_division=0),
    "recall_micro": recall_score(y_test, y_pred, average="micro", zero_division=0),
    "subset_accuracy": accuracy_score(y_test, y_pred),
    "hamming_loss": hamming_loss(y_test, y_pred),
}

prec_g, rec_g, f1_g, support_g = precision_recall_fscore_support(
    y_test, y_pred, average=None, zero_division=0
)
for i, genre in enumerate(classes):
    key = sanitize(genre)
    metrics[f"f1_{key}"] = float(f1_g[i])
    metrics[f"precision_{key}"] = float(prec_g[i])
    metrics[f"recall_{key}"] = float(rec_g[i])
    metrics[f"support_{key}"] = float(support_g[i])
    metrics[f"threshold_{key}"] = float(threshold_vec[i])

log_metrics(metrics)
print(f"\n  f1_macro={metrics['f1_macro']:.4f}  f1_micro={metrics['f1_micro']:.4f}")

# ── Save artifacts ─────────────────────────────────────────────────────────
print("Saving artifacts to models/...")
joblib.dump(clf,          MODELS_DIR / "classifier.joblib")
joblib.dump(tfidf,        MODELS_DIR / "tfidf.joblib")
joblib.dump(scaler_emb,   MODELS_DIR / "scaler_emb.joblib")
joblib.dump(scaler_final, MODELS_DIR / "scaler_final.joblib")
joblib.dump(mlb,          MODELS_DIR / "mlb.joblib")
np.save(MODELS_DIR / "threshold_vec.npy", threshold_vec)

# ── Registro en MLflow ─────────────────────────────────────────────────────
if USE_MLFLOW:
    import scipy
    import sklearn

    from src.mlflow_model import GenreClassifier, build_artifacts_map

    print("\nLogging report and thresholds to MLflow...")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "classification_report.txt").write_text(report_text, encoding="utf-8")
        pd.DataFrame({"genre": classes, "threshold": threshold_vec}).to_csv(
            tmp_path / "thresholds.csv", index=False
        )
        (tmp_path / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        mlflow.log_artifacts(str(tmp_path), artifact_path="evaluation")

    mlflow.set_tags(
        {
            "pipeline": "hybrid-embeddings-tfidf",
            "task": "multilabel-genre-classification",
            "embedding_model": EMBEDDING_MODEL,
        }
    )

    signature = ModelSignature(
        inputs=Schema([ColSpec("string", "plot")]),
        outputs=Schema(
            [
                ColSpec("string", "predicted_genres"),
                ColSpec("string", "probabilities"),
            ]
        ),
    )

    pip_requirements = [
        f"mlflow=={mlflow.__version__}",
        f"scikit-learn=={sklearn.__version__}",
        f"scipy=={scipy.__version__}",
        f"numpy=={np.__version__}",
        f"pandas=={pd.__version__}",
        f"joblib=={joblib.__version__}",
        "sentence-transformers",
        "spacy",
        (
            "https://github.com/explosion/spacy-models/releases/download/"
            "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
        ),
    ]

    print("Logging pyfunc model bundle to MLflow...")
    model_info = mlflow.pyfunc.log_model(
        name="model",
        python_model=GenreClassifier(),
        artifacts=build_artifacts_map(MODELS_DIR),
        code_paths=["src"],
        signature=signature,
        input_example=pd.DataFrame({"plot": [str(X_test.iloc[0])]}),
        pip_requirements=pip_requirements,
        registered_model_name=None if args.no_register else args.model_name,
    )

    if not args.no_register:
        version = model_info.registered_model_version
        client = MlflowClient()
        client.set_model_version_tag(
            args.model_name, version, "f1_macro", f"{metrics['f1_macro']:.4f}"
        )
        client.set_model_version_tag(
            args.model_name, version, "f1_micro", f"{metrics['f1_micro']:.4f}"
        )
        print(f"\nRegistered: {args.model_name} version {version}")

        if args.promote_alias:
            client.set_registered_model_alias(
                args.model_name, args.promote_alias, version
            )
            print(
                f"Alias @{args.promote_alias} -> {args.model_name} v{version} "
                "(el CD desplegara esta version)"
            )
        else:
            print(
                "Version registrada sin alias: el CD sigue desplegando la anterior.\n"
                "Para promoverla, vuelve a correr con --promote-alias champion "
                "o asigna el alias desde la UI de MLflow."
            )

    mlflow.end_run()

print("\nDone! Run: streamlit run app.py")
