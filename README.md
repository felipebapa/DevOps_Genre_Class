# NLP-DGO — Clasificador de Géneros Audiovisuales

Pipeline NLP híbrido para clasificación multilabel de géneros a partir de la sinopsis de películas y series. Combina embeddings semánticos (`all-MiniLM-L6-v2`) con TF-IDF sobre lemmas extraídos con spaCy, alimentando un clasificador `OneVsRest` por género. Incluye una interfaz interactiva con Streamlit.

---

## Equipo

- Luis Jorge García Camargo
- Luis Eduardo Uribe
- Felipe Barreto Patiño

*Proyecto — DevOps, Maestría en Inteligencia Artificial*

---

## Estructura del proyecto

```
DevOps_Genre_Class/
├── .github/
│   └── workflows/
│       └── ci.yml           # Workflow de CI: lint (ruff), imports y tests (pytest) en cada push/PR
├── app.py                   # Aplicación Streamlit (3 tabs: intro, métricas, predicción)
├── train.py                 # Script de entrenamiento — genera los artefactos en models/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Pipeline de preprocesamiento con spaCy
│   └── model_utils.py       # Carga de artefactos y función de predicción
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Pruebas unitarias de normalize_text() y preprocess_text()
│   └── test_model_utils.py     # Pruebas unitarias de models_exist() y predict_genres() (con mocks)
├── notebooks/
│   ├── KLUSTERS.xlsx                                    # Dataset principal
│   ├── Test_of_AUDIOVISUAL_Class_NLP.ipynb              # Notebook de experimentación
│   ├── Last_resultados_modelo_hibrido_embeddings_tfidf.csv
│   └── thresholds_por_genero.csv
├── models/                  # Artefactos entrenados (generados por train.py, no en git)
├── .gitignore
├── .python-version
├── pyproject.toml           # Incluye dependencias de dev: pytest, ruff
├── uv.lock
└── README.md
```

---

## Guía de instalación y ejecución

### Requisitos previos

**Instalar `uv`** (gestor de paquetes y entornos de Python):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Cerrar y volver a abrir la terminal para actualizar el PATH, luego verificar:

```bash
uv --version
```

---

### Paso 1 — Clonar el repositorio

```bash
git clone <URL-del-repo>
cd nlp-dgo
```

---

### Paso 2 — Crear el entorno e instalar dependencias

```bash
uv sync
```

Crea automáticamente el `.venv` e instala todas las dependencias del `pyproject.toml`, incluyendo spaCy, SentenceTransformers, Streamlit y el modelo de inglés `en_core_web_sm`.

> La primera vez puede tardar unos minutos descargando paquetes.

---

### Paso 3 — Entrenar el modelo

```bash
uv run python train.py
```

Este script carga el dataset, genera embeddings, entrena el clasificador, ajusta los thresholds por género y guarda los artefactos en `models/`. **Solo se ejecuta una vez.**

> Tarda entre 5 y 15 minutos dependiendo del hardware.

---

### Paso 4 — Lanzar la aplicación Streamlit

```bash
uv run streamlit run app.py
```

Abre automáticamente el navegador en `http://localhost:8501`.

---

### Resumen rápido

```bash
git clone <URL-del-repo>
cd nlp-dgo
uv sync
uv run python train.py
uv run streamlit run app.py
```

---

### Seleccionar el intérprete en VS Code (opcional)

1. `Ctrl + Shift + P` → **Python: Select Interpreter**
2. Elegir `.venv\Scripts\python.exe` dentro de la carpeta del proyecto

---

### Solución de problemas comunes

| Problema | Solución |
|---|---|
| `uv: command not found` | Reiniciar la terminal tras instalar uv |
| Error al importar spaCy | Verificar que `uv sync` completó sin errores |
| `models/` no encontrado al abrir Streamlit | Ejecutar `train.py` primero |
| Puerto 8501 ocupado | `uv run streamlit run app.py --server.port 8502` |
