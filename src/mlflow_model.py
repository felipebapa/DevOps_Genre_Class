"""
Envoltorio `mlflow.pyfunc` del pipeline híbrido.

El modelo no es un único estimador de scikit-learn: son seis artefactos
(clasificador, TF-IDF, dos scalers, el binarizador de etiquetas y el vector de
thresholds) que solo tienen sentido juntos. Empaquetarlos como un `PythonModel`
permite:

  * registrarlos como UNA sola versión en el Model Registry,
  * que el CD descargue el bundle completo por alias (``models:/nombre@champion``),
  * y servir el modelo con ``mlflow models serve`` sin código adicional.

Los nombres de archivo dentro del bundle son idénticos a los que espera
``src.model_utils.load_artifacts()``, de modo que el CD solo tiene que copiar
``artifacts/*`` a ``models/`` y la app Streamlit funciona sin cambios.
"""

import json
from pathlib import Path

import mlflow.pyfunc

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# clave lógica -> nombre del archivo en models/
ARTIFACT_FILES = {
    "classifier": "classifier.joblib",
    "tfidf": "tfidf.joblib",
    "scaler_emb": "scaler_emb.joblib",
    "scaler_final": "scaler_final.joblib",
    "mlb": "mlb.joblib",
    "threshold_vec": "threshold_vec.npy",
}


def build_artifacts_map(models_dir: Path) -> dict[str, str]:
    """Mapa {clave: ruta} que se le pasa a ``mlflow.pyfunc.log_model``."""
    return {key: str(models_dir / name) for key, name in ARTIFACT_FILES.items()}


class GenreClassifier(mlflow.pyfunc.PythonModel):
    """Recibe sinopsis en texto y devuelve géneros predichos + probabilidades."""

    def load_context(self, context):
        import joblib
        import numpy as np
        from sentence_transformers import SentenceTransformer

        from src.preprocessing import load_spacy_model

        paths = context.artifacts
        self._artifacts = {
            "clf": joblib.load(paths["classifier"]),
            "tfidf": joblib.load(paths["tfidf"]),
            "scaler_emb": joblib.load(paths["scaler_emb"]),
            "scaler_final": joblib.load(paths["scaler_final"]),
            "mlb": joblib.load(paths["mlb"]),
            "threshold_vec": np.load(paths["threshold_vec"]),
        }
        self._nlp = load_spacy_model()
        self._emb_model = SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    def _as_texts(model_input) -> list[str]:
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            column = "plot" if "plot" in model_input.columns else model_input.columns[0]
            return model_input[column].astype(str).tolist()
        if isinstance(model_input, str):
            return [model_input]
        return [str(x) for x in model_input]

    def predict(self, context, model_input, params=None):
        import pandas as pd

        from src.model_utils import predict_genres

        rows = []
        for text in self._as_texts(model_input):
            result = predict_genres(
                text,
                self._artifacts,
                self._nlp,
                self._emb_model,
            )
            rows.append(
                {
                    "predicted_genres": ", ".join(result["predicted"]),
                    "probabilities": json.dumps(
                        {k: round(v, 6) for k, v in result["probabilities"].items()}
                    ),
                }
            )
        return pd.DataFrame(rows)
