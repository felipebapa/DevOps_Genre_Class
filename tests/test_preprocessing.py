import pytest

from src.preprocessing import normalize_text, preprocess_text

# ---------------------------------------------------------------------
# Tests de normalize_text (función pura, sin dependencias externas)
# ---------------------------------------------------------------------

def test_normalize_text_collapses_whitespace():
    """Múltiples espacios/saltos de línea deben colapsar a uno solo."""
    text = "Hola    mundo\n\ncomo   estas"
    assert normalize_text(text) == "Hola mundo como estas"


def test_normalize_text_strips_leading_trailing_spaces():
    text = "   texto con espacios al borde   "
    assert normalize_text(text) == "texto con espacios al borde"


def test_normalize_text_removes_content_after_double_colon():
    """Todo lo que va después de '::' debe eliminarse (metadata tipo notebook)."""
    text = "Una sinopsis interesante::genre=Action;year=2020"
    assert normalize_text(text) == "Una sinopsis interesante"


def test_normalize_text_handles_non_string_input():
    """Debe poder manejar valores no-string (ej. floats/NaN de pandas) sin explotar."""
    result = normalize_text(123)
    assert isinstance(result, str)
    assert result == "123"


def test_normalize_text_empty_string():
    assert normalize_text("") == ""


# ---------------------------------------------------------------------
# Tests de preprocess_text (requiere el modelo de spaCy en_core_web_sm)
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def nlp():
    """Carga el modelo de spaCy una sola vez por módulo de tests.
    Si el modelo no está instalado, se hace skip en vez de fallar el CI.
    """
    try:
        from src.preprocessing import load_spacy_model
        return load_spacy_model()
    except OSError:
        pytest.skip("Modelo en_core_web_sm no está instalado en este entorno")


def test_preprocess_text_returns_string(nlp):
    result = preprocess_text("A young detective solves a mysterious murder case.", nlp)
    assert isinstance(result, str)


def test_preprocess_text_removes_punctuation_and_stopwords(nlp):
    result = preprocess_text("The cat, the dog, and the bird are friends!", nlp)
    # No debe contener signos de puntuación
    assert "," not in result
    assert "!" not in result
    # Stopwords comunes como "the" y "and" deberían filtrarse
    tokens = result.split()
    assert "the" not in tokens
    assert "and" not in tokens


def test_preprocess_text_negation_words_are_filtered_by_pos(nlp):
    """
    NOTA / hallazgo: KEEP_STOPWORDS incluye 'never', 'not', etc. para
    preservar negaciones, pero como son adverbios (ADV) y ALLOWED_POS
    solo permite {NOUN, PROPN, VERB, ADJ}, terminan filtrándose de
    todas formas por el segundo filtro (pos_ not in ALLOWED_POS).

    Este test documenta el comportamiento ACTUAL del pipeline, no
    necesariamente el deseado. Si el equipo decide que las negaciones
    sí deben conservarse, habría que ajustar preprocessing.py (por
    ejemplo, agregando 'ADV' a ALLOWED_POS o dando un trato especial
    a los tokens en KEEP_STOPWORDS antes del filtro de POS) y luego
    actualizar este test para reflejar el nuevo comportamiento.
    """
    result = preprocess_text("She was never found again.", nlp)
    tokens = result.split()
    assert "never" not in tokens  # comportamiento actual, no ideal


def test_preprocess_text_empty_input(nlp):
    result = preprocess_text("", nlp)
    assert result == ""