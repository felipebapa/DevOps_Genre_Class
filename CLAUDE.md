# CLAUDE.md

Contexto del proyecto para Claude Code. Complementa el [README.md](README.md), que es la guía para personas.

---

## Qué es

Clasificador multilabel de géneros audiovisuales a partir de la sinopsis (*plot*) en inglés. Pipeline NLP híbrido: embeddings semánticos (`all-MiniLM-L6-v2`) + TF-IDF sobre lemmas de spaCy, fusionados y alimentados a un `OneVsRestClassifier(LogisticRegression)`, con **thresholds ajustados por género**. Interfaz en Streamlit.

Proyecto académico de DevOps — Maestría en Inteligencia Artificial. Equipo: Luis Jorge García Camargo, Luis Eduardo Uribe, Felipe Barreto Patiño.

El foco de la asignatura es la **cadena DevOps/MLOps**, no la métrica del modelo.

---

## Entorno

- Windows 11, `uv` como gestor de paquetes, Python 3.14.
- Shell: PowerShell y Git Bash disponibles.
- Docker Desktop con WSL2.
- Remote: `https://github.com/felipebapa/DevOps_Genre_Class.git`, rama `main`.

---

## Comandos

```bash
uv sync                          # entorno base
uv sync --group mlops            # + cliente de MLflow (necesario para train.py)
uv run ruff check .              # lint (excluye notebooks/)
uv run pytest -q                 # 15 tests
uv run streamlit run app.py      # app en :8501

docker compose up -d             # MLflow + Jenkins en la red `mlops`
docker compose up -d mlflow      # solo el tracking server (:5000)

uv run python train.py --promote-alias champion   # entrena, registra y promueve
uv run python train.py --no-mlflow                # entrena sin tocar MLflow
```

`MLFLOW_TRACKING_URI` controla contra qué servidor trabaja `train.py` (default `http://localhost:5000`).

---

## Arquitectura

### Código

| Archivo | Rol |
|---|---|
| [src/preprocessing.py](src/preprocessing.py) | Normalización + lematización con spaCy (NER placeholders, POS permitidos, stopwords conservadas) |
| [src/model_utils.py](src/model_utils.py) | `load_artifacts()` y `predict_genres()` — incluye el `decode_row` que aplica thresholds |
| [src/mlflow_model.py](src/mlflow_model.py) | Envoltorio `mlflow.pyfunc` del bundle de 6 artefactos |
| [train.py](train.py) | Entrenamiento + logging + registro. Script lineal a nivel de módulo, no tiene `main()` |
| [app.py](app.py) | Streamlit, 3 tabs: introducción, métricas, predicción |

### El modelo NO es un solo estimador

Son **seis artefactos que solo sirven juntos**:

```
classifier.joblib  tfidf.joblib  scaler_emb.joblib
scaler_final.joblib  mlb.joblib  threshold_vec.npy
```

Por eso [src/mlflow_model.py](src/mlflow_model.py) los envuelve en un `PythonModel`: permite registrarlos como **una sola versión** en el Model Registry. Los nombres de archivo dentro del bundle son idénticos a los que espera `src.model_utils.load_artifacts()`, así que el CD solo copia `artifacts/*` → `models/` y la app no cambia.

### CI / CD

- **CI** ([.github/workflows/ci.yml](.github/workflows/ci.yml)) — GitHub Actions: ruff, imports, pytest en cada push/PR a `main`.
- **CD** ([Jenkinsfile](Jenkinsfile)) — Jenkins. **El modelo NO sale del repo: sale del Model Registry de MLflow.**

Etapas del CD:

1. `Checkout`
2. `Verify Toolchain` — Docker, construye la imagen cliente de MLflow si falta, health check del server
3. `Fetch Model from Registry` — resuelve `genre-classifier@champion`, **aplica el gate de F1**, descarga a `models/`
4. `Build Docker Image` — tags `:<BUILD_NUMBER>` y `:model-v<VERSION>`, labels con `run_id`, commit y `f1_macro`
5. `Push Docker Image` → Docker Hub (`leuribe2/devops-genre-class`)
6. `Record Deployment in MLflow` — etiqueta la versión con la imagen/build/commit que la desplegaron

Parámetros del job: `MODEL_ALIAS` (default `champion`), `MIN_F1_MACRO` (default `0.30`), `PUSH_IMAGE`.

**Rollback** = mover el alias `@champion` a otra versión y relanzar. No se revierte ningún commit.

---

## Decisiones y trampas (leer antes de tocar)

### `mlflow-skinny`, nunca `mlflow`, en el pyproject

El paquete `mlflow` completo fija `pandas<3` y este proyecto usa `pandas>=3.0.2` — `uv lock` falla con conflicto irresoluble. `mlflow-skinny` no tiene esa restricción y cubre tracking, registry y `pyfunc.log_model` (verificado end-to-end). El servidor sí usa `mlflow` completo, en [docker/mlflow/Dockerfile](docker/mlflow/Dockerfile).

Vive en el grupo `mlops`, que **no** entra en la imagen de la app (`uv sync --frozen --no-dev` no lo instala — verificado).

### `--allowed-hosts` es obligatorio

MLflow 3 responde **`403 Invalid Host header`** a cualquier `Host` fuera de su allowlist (protección anti DNS rebinding). Sin `mlflow:5000` en la lista, los contenedores del pipeline no pueden hablar con el servidor. Se configura vía `MLFLOW_ALLOWED_HOSTS` en `.env` — ver [.env.example](.env.example).

Es el error que más frena a quien levanta esto por primera vez.

### `WORKSPACE_MOUNT` en el Jenkinsfile

Los scripts del CD corren en un **contenedor efímero** (así el agente Jenkins solo necesita Docker, ni Python ni mlflow). Pero el daemon de Docker es el del host: si Jenkins corre en contenedor, `$(pwd)` es una ruta interna que el daemon no resuelve. Por eso se monta el **volumen nombrado** y se usa `-w "${WORKSPACE}"`.

| Cómo corre Jenkins | Valor |
|---|---|
| Con el `docker-compose.yml` del repo | `genre-mlops_jenkins-home:/var/jenkins_home` |
| Nativo en el host | `"${WORKSPACE}:${WORKSPACE}"` |

### `models/` ya no se versiona en git

Lo produce `train.py` localmente y lo baja el CD del registry. `.gitignore` usa `models/*` + `!models/.gitkeep` — con `models/` a secas git no desciende al directorio y la negación no funciona.

### Encoding en Windows

MLflow imprime emojis en sus mensajes de estado y la consola de Windows usa cp1252 → `UnicodeEncodeError` **al cerrar la corrida, después de entrenar 15 minutos**. `train.py` ya hace `sys.stdout.reconfigure(encoding="utf-8")`. Para otros scripts: `PYTHONUTF8=1`.

### `log_model` revienta al final en Windows

`mlflow.pyfunc.log_model` con `code_paths=["src"]` termina con
`PermissionError: [WinError 5]` al borrar su directorio temporal: MLflow copia
`src/` allí, lo importa al cargar el modelo, y Windows no borra un directorio
con módulos Python abiertos.

Lo traicionero es **cuándo** ocurre: después de registrar la versión, así que el
modelo queda guardado pero el script muere antes de asignar el alias — 15 minutos
de entrenamiento para acabar sin nada que desplegar. Pasa con y sin
`input_example` (comprobado).

`train.py` lo captura y recupera la versión por `run_id`. Si se toca ese bloque,
no quitar el `try/except`. Deja un temporal huérfano por corrida; es inofensivo.

### Contrato de `scripts/mlflow_fetch_model.py`

El Jenkinsfile distingue los exit codes, no los cambies sin actualizarlo:

| Exit | Significado |
|---|---|
| 0 | OK — artefactos + `models/MODEL_VERSION.env` |
| 1 | Error de conexión, alias inexistente o bundle incompleto |
| 2 | El modelo no pasa el gate de calidad |

`MODEL_VERSION.env` es cómo el pipeline pasa datos de la etapa de fetch a las siguientes (`MODEL_VERSION`, `MODEL_RUN_ID`, `MODEL_F1_MACRO`).

### Heredocs en Bash

Escribir archivos Python grandes con `cat <<'EOF'` en este entorno **corrompe los backslashes** y a veces rompe el parser. Usar la herramienta Write, o construir los backslashes con `chr(92)`.

---

## Traza: sesión del 2026-08-23

Punto de partida: CD que solo hacía build + push, con los `.joblib` versionados en git y sin ninguna métrica asociada al modelo desplegado.

**Decisiones del usuario:** alcance = *el Registry dirige el build* (no solo tracking, no retrain en CD); infraestructura = *docker-compose local junto a Jenkins*; Jenkins corre *en la máquina de otro integrante*.

**Implementado:**

- `docker-compose.yml` + `docker/mlflow/` + `docker/jenkins/` — infra en la red `mlops`
- `src/mlflow_model.py` — wrapper pyfunc del bundle
- `train.py` instrumentado — params, métricas agregadas y por género, artefactos de evaluación, registro y `--promote-alias`
- `scripts/mlflow_fetch_model.py` — fetch por alias + gate de calidad
- `scripts/mlflow_tag_deployment.py` — trazabilidad inversa (versión → imagen)
- `Jenkinsfile` reescrito — 6 etapas, parámetros, tags y labels con la versión del modelo
- `pyproject.toml` — grupo `mlops` con `mlflow-skinny`
- `.env.example` + compose parametrizado (`MLFLOW_ALLOWED_HOSTS`, `MLFLOW_PORT`) para el escenario de Jenkins remoto
- `models/` fuera de git; README ampliado con la sección *MLOps con MLflow*

**Verificado empíricamente** (no solo escrito):

- Servidor levantado, UI HTTP 200, health OK
- Se registró el bundle real como modelo de prueba y se ejecutó `mlflow_fetch_model.py` en un contenedor efímero igual que en Jenkins: resolvió el alias, pasó el gate, descargó los 6 artefactos, escribió `MODEL_VERSION.env`
- Los dos caminos de fallo: gate no superado → exit 2; alias inexistente → exit 1
- El modelo de prueba se borró del registry al terminar
- `ruff` limpio, 15 tests pasan, `docker compose config` válido con y sin overrides

Los bugs de `--allowed-hosts` y del `UnicodeEncodeError` salieron de esa verificación; ambos habrían roto el primer build real.

**NO verificado:** el pipeline completo corriendo en Jenkins, y un `docker build` de la app con los artefactos descargados (el build es idéntico al que ya funcionaba, solo cambia el origen de `models/`).

---

## Estado pendiente

1. **Nada está commiteado.** Todo el trabajo de MLflow está en el working tree; la salida de `models/` de git está *staged*. Jenkins hace `checkout scm`, así que sin commit + push el compañero no recibe el `Jenkinsfile` nuevo.
2. **No se ha entrenado con MLflow.** El registry está vacío: no existe aún `genre-classifier` ni el alias `@champion`. Hasta que se entrene, el CD falla a propósito en la etapa de fetch.
3. **El servidor MLflow del equipo no existe todavía** — debe levantarlo quien tiene Jenkins, con la IP correcta en `MLFLOW_ALLOWED_HOSTS`.

---

## Convenciones

- Comentarios y documentación **en español**; nombres de código en inglés.
- Los comentarios explican *por qué*, no *qué*. El código existente tiene poca densidad de comentarios salvo donde hay una trampa.
- `ruff` con `exclude = ["notebooks"]`; los notebooks no se lintean ni se tocan.
- No commitear sin que el usuario lo pida.
