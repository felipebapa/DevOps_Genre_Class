import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src import model_utils
from src.model_utils import _REQUIRED, models_exist, predict_genres

# Tests de models_exist()

def test_models_exist_false_when_directory_missing(tmp_path, monkeypatch):
    """Si la carpeta models/ no existe o está vacía, debe devolver False."""
    fake_dir = tmp_path / "models_vacio"
    monkeypatch.setattr(model_utils, "MODELS_DIR", fake_dir)
    assert models_exist() is False


def test_models_exist_true_when_all_files_present(tmp_path, monkeypatch):
    """Si todos los artefactos requeridos existen, debe devolver True."""
    fake_dir = tmp_path / "models_completo"
    fake_dir.mkdir()
    for filename in _REQUIRED:
        (fake_dir / filename).write_text("dummy")
    monkeypatch.setattr(model_utils, "MODELS_DIR", fake_dir)
    assert models_exist() is True


def test_models_exist_false_when_one_file_missing(tmp_path, monkeypatch):
    """Si falta un solo artefacto, debe devolver False (no basta con la mayoría)."""
    fake_dir = tmp_path / "models_incompleto"
    fake_dir.mkdir()
    for filename in _REQUIRED[:-1]:  # se salta el último archivo a propósito
        (fake_dir / filename).write_text("dummy")
    monkeypatch.setattr(model_utils, "MODELS_DIR", fake_dir)
    assert models_exist() is False


# Tests de predict_genres() usando artefactos y modelos "falsos" (mocks)
# No requiere tener modelos entrenados reales ni spaCy descargado.

class _FakeScaler:
    """Simula un scaler de sklearn: transform() devuelve el input tal cual."""
    def transform(self, X):
        return X


class _FakeTfidf:
    def transform(self, texts):
        # sklearn's TfidfVectorizer.transform() siempre devuelve sparse;
        # replicamos eso aquí porque scipy.sparse.hstack lo requiere
        # para combinar correctamente con el array denso de embeddings.
        return csr_matrix(np.ones((1, 3)))


class _FakeClassifier:
    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, X):
        return np.array([self._probs])


class _FakeMultiLabelBinarizer:
    def __init__(self, classes):
        self.classes_ = np.array(classes)


class _FakeEmbeddingModel:
    def encode(self, texts):
        return np.zeros((1, 4))


@pytest.fixture
def fake_artifacts():
    classes = ["Action", "Comedy", "Drama"]
    probs = [0.9, 0.3, 0.85]  # Action y Drama superan el threshold
    threshold_vec = np.array([0.5, 0.5, 0.5])
    return {
        "clf": _FakeClassifier(probs),
        "tfidf": _FakeTfidf(),
        "scaler_emb": _FakeScaler(),
        "scaler_final": _FakeScaler(),
        "mlb": _FakeMultiLabelBinarizer(classes),
        "threshold_vec": threshold_vec,
    }


def test_predict_genres_returns_expected_keys(monkeypatch, fake_artifacts):
    # Evita depender de spaCy real: parcheamos preprocess_text
    monkeypatch.setattr(
        "src.preprocessing.preprocess_text", lambda text, nlp: "fake lemma text"
    )

    result = predict_genres(
        text="A hero saves the city while making everyone laugh.",
        artifacts=fake_artifacts,
        nlp=None,
        emb_model=_FakeEmbeddingModel(),
    )

    assert set(result.keys()) == {"probabilities", "predicted", "thresholds", "lemmas"}
    assert result["lemmas"] == "fake lemma text"


def test_predict_genres_selects_genres_above_threshold(monkeypatch, fake_artifacts):
    monkeypatch.setattr(
        "src.preprocessing.preprocess_text", lambda text, nlp: "fake lemma text"
    )

    result = predict_genres(
        text="Cualquier texto de prueba",
        artifacts=fake_artifacts,
        nlp=None,
        emb_model=_FakeEmbeddingModel(),
    )

    # Con probs = [0.9, 0.3, 0.85] y threshold 0.5, deben predecirse Action y Drama
    assert "Action" in result["predicted"]
    assert "Drama" in result["predicted"]
    assert "Comedy" not in result["predicted"]


def test_predict_genres_always_includes_top1_even_below_threshold(monkeypatch):
    """Aunque ninguna clase supere el threshold, la de mayor probabilidad debe incluirse."""
    classes = ["Action", "Comedy", "Drama"]
    probs = [0.2, 0.1, 0.15]  # ninguna supera 0.5
    threshold_vec = np.array([0.5, 0.5, 0.5])
    artifacts = {
        "clf": _FakeClassifier(probs),
        "tfidf": _FakeTfidf(),
        "scaler_emb": _FakeScaler(),
        "scaler_final": _FakeScaler(),
        "mlb": _FakeMultiLabelBinarizer(classes),
        "threshold_vec": threshold_vec,
    }
    monkeypatch.setattr(
        "src.preprocessing.preprocess_text", lambda text, nlp: "fake lemma text"
    )

    result = predict_genres(
        text="Texto ambiguo",
        artifacts=artifacts,
        nlp=None,
        emb_model=_FakeEmbeddingModel(),
    )

    # Action tiene la probabilidad más alta (0.2), debe estar presente
    assert result["predicted"] == ["Action"]