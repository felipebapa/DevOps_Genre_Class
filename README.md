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
├── Jenkinsfile              # Pipeline de CD: modelo desde MLflow → build → push → trazabilidad
├── docker-compose.yml       # Infraestructura local: MLflow tracking server + Jenkins
├── Dockerfile               # Imagen de la app Streamlit
├── app.py                   # Aplicación Streamlit (3 tabs: intro, métricas, predicción)
├── train.py                 # Entrenamiento — genera models/, loguea en MLflow y registra el modelo
├── docker/
│   ├── mlflow/Dockerfile    # Tracking server; también es el cliente que usa el CD
│   └── jenkins/Dockerfile   # Jenkins + CLI de Docker
├── scripts/
│   ├── mlflow_fetch_model.py     # CD: baja el modelo @champion y aplica el gate de calidad
│   └── mlflow_tag_deployment.py  # CD: marca la versión como desplegada (imagen, build, commit)
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Pipeline de preprocesamiento con spaCy
│   ├── model_utils.py       # Carga de artefactos y función de predicción
│   └── mlflow_model.py      # Envoltorio mlflow.pyfunc del bundle de 6 artefactos
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py   # Pruebas unitarias de normalize_text() y preprocess_text()
│   └── test_model_utils.py     # Pruebas unitarias de models_exist() y predict_genres() (con mocks)
├── notebooks/
│   ├── KLUSTERS.xlsx                                    # Dataset principal
│   ├── Test_of_AUDIOVISUAL_Class_NLP.ipynb              # Notebook de experimentación
│   ├── Last_resultados_modelo_hibrido_embeddings_tfidf.csv
│   └── thresholds_por_genero.csv
├── models/                  # Artefactos del modelo — NO versionados en git: los produce
│                            # train.py y el CD los baja del Model Registry de MLflow
├── .env.example             # Config del servidor MLflow (hosts permitidos, puerto)
├── .gitignore
├── .python-version
├── pyproject.toml           # Grupos: dev (pytest, ruff) y mlops (mlflow-skinny)
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

Este script carga el dataset, genera embeddings, entrena el clasificador, ajusta los thresholds por género y guarda los artefactos en `models/`.

> Tarda entre 5 y 15 minutos dependiendo del hardware.

Para además registrar la corrida y el modelo en MLflow —que es de donde el CD toma
el modelo a desplegar— ver [MLOps con MLflow](#mlops-con-mlflow):

```bash
docker compose up -d mlflow
uv sync --group mlops
uv run python train.py --promote-alias champion
```

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

---

## MLOps con MLflow

El CD no construye la imagen con los artefactos que estén en el repo: **la construye con la versión del modelo que esté marcada como `@champion` en el Model Registry de MLflow**. Los `.joblib` ya no se versionan en git.

```
train.py  ──log──►  MLflow Tracking  ──registra──►  Model Registry
                    (params, métricas,              genre-classifier
                     reporte, bundle)                  v1, v2, v3...
                                                          │
                                                    alias @champion
                                                          │
                                                          ▼
                                     Jenkins CD ── fetch ──┘
                                          │
                                     gate de F1
                                          │
                                     docker build ──► Docker Hub
                                          │            :<build>
                                          │            :model-v<version>
                                          └──── etiqueta la versión
                                                como desplegada
```

### Qué aporta

| Antes | Ahora |
|---|---|
| Los `.joblib` viajaban en git sin métricas asociadas | Cada versión del modelo tiene params, métricas por género y su reporte |
| No se sabía qué modelo corría en una imagen | La imagen lleva tag `model-vN` y labels con el `run_id` |
| Volver atrás requería revertir commits binarios | Se cambia el alias `@champion` y se reconstruye |
| Nada impedía publicar un modelo peor | El build falla si el `f1_macro` no llega al mínimo |

---

### Paso 1 — Levantar la infraestructura

```bash
docker compose up -d
```

Levanta dos servicios en la red `mlops`:

| Servicio | URL | Qué es |
|---|---|---|
| `mlflow` | http://localhost:5000 | Tracking server + Model Registry (SQLite + artefactos en volumen) |
| `jenkins` | http://localhost:8080 | Jenkins con el CLI de Docker incluido |

> **¿Ya tienes tu propio Jenkins?** Levanta solo MLflow con `docker compose up -d mlflow` y conecta tu contenedor a la red:
> ```bash
> docker network connect mlops <nombre-del-contenedor-jenkins>
> ```
> Si tu Jenkins corre nativo en el host (no en Docker), cambia `WORKSPACE_MOUNT` en el `Jenkinsfile` a `"${WORKSPACE}:${WORKSPACE}"`.

La contraseña inicial de Jenkins:

```bash
docker exec jenkins cat /var/jenkins_home/secrets/initialAdminPassword
```

---

### Configuración de equipo: Jenkins en otra máquina

Si Jenkins no corre en tu equipo sino en el de otro integrante, **MLflow tiene que vivir en esa misma máquina**. El pipeline resuelve `http://mlflow:5000` por nombre de servicio dentro de la red `mlops`, así que el servidor debe estar en el mismo host de Docker que Jenkins. Un MLflow en la laptop de otra persona no sirve: tendría que estar encendida en cada build.

```
   Máquina con Jenkins                     Otros integrantes
  ┌───────────────────────────┐
  │  red docker `mlops`       │           train.py  ──────┐
  │  ┌─────────┐ ┌─────────┐  │                           │
  │  │ jenkins │→│ mlflow  │  │ ◄─── http://<IP-HOST>:5000┘
  │  └─────────┘ └─────────┘  │                           │
  │        http://mlflow:5000 │           navegador ──────┘
  └───────────────────────────┘              (UI)
```

**En la máquina que hospeda Jenkins:**

1. Clonar el repo y averiguar su IP en la red local (`ipconfig` en Windows, `ip a` en Linux).

2. Crear el archivo `.env` con esa IP, para que MLflow acepte las conexiones de los demás:

   ```bash
   cp .env.example .env
   ```

   Editar `MLFLOW_ALLOWED_HOSTS` y poner la IP real. Sin esto, todo acceso externo recibe `403 Invalid Host header`:

   ```
   MLFLOW_ALLOWED_HOSTS=mlflow:5000,localhost:5000,127.0.0.1:5000,192.168.1.50:5000
   ```

3. Levantar los servicios:

   ```bash
   docker compose up -d
   ```

   Si Jenkins ya existe y no se quiere reemplazar, levantar solo MLflow y conectar el Jenkins existente a la red:

   ```bash
   docker compose up -d mlflow
   docker network connect mlops <nombre-del-contenedor-jenkins>
   ```

4. Abrir el puerto 5000 en el firewall. En Windows, desde PowerShell **como administrador**:

   ```powershell
   New-NetFirewallRule -DisplayName "MLflow" -Direction Inbound -LocalPort 5000 -Protocol TCP -Action Allow
   ```

5. Ajustar `WORKSPACE_MOUNT` en el `Jenkinsfile` según cómo corra ese Jenkins:

   | Cómo corre Jenkins | Valor de `WORKSPACE_MOUNT` |
   |---|---|
   | Con este `docker-compose.yml` | `genre-mlops_jenkins-home:/var/jenkins_home` (ya está puesto) |
   | En contenedor con otro volumen | `<nombre-del-volumen>:/var/jenkins_home` |
   | Nativo en el host (sin Docker) | `"${WORKSPACE}:${WORKSPACE}"` |

   Es la ruta que el daemon de Docker usa para pasarle el workspace al contenedor que descarga el modelo; si no coincide, la etapa *Fetch Model* no encuentra los scripts.

**Desde tu máquina** (y la de cualquier otro integrante), apuntar al servidor remoto:

```bash
# Linux / macOS / Git Bash
export MLFLOW_TRACKING_URI=http://192.168.1.50:5000

# PowerShell
$env:MLFLOW_TRACKING_URI = "http://192.168.1.50:5000"
```

Y entrenar normalmente. La corrida y la versión quedan en el servidor compartido, visibles para todo el equipo:

```bash
uv sync --group mlops
uv run python train.py --promote-alias champion
```

La UI queda en `http://192.168.1.50:5000` para todos.

> El servidor no lleva autenticación. Está bien para una LAN de confianza o la sustentación; no lo expongan a internet tal cual.

---

### Paso 2 — Entrenar y registrar el modelo

Instalar el cliente de MLflow (grupo aparte, no entra en la imagen de la app):

```bash
uv sync --group mlops
```

Entrenar. Cada corrida queda registrada como una **nueva versión** del modelo:

```bash
uv run python train.py
```

Entrenar y además promoverla para que el CD la despliegue:

```bash
uv run python train.py --promote-alias champion
```

Opciones útiles:

| Flag | Efecto |
|---|---|
| `--promote-alias champion` | Asigna el alias que el CD despliega |
| `--no-register` | Loguea la corrida sin crear versión en el registry (experimentos) |
| `--no-mlflow` | Entrena sin tocar MLflow, solo escribe `models/` |
| `--run-name "..."` | Nombra la corrida en la UI |

El tracking server se toma de `MLFLOW_TRACKING_URI` (por defecto `http://localhost:5000`).

Lo que queda registrado en cada corrida:

- **Params** — hiperparámetros de TF-IDF y del clasificador, tamaños de split, dimensiones de features, modelo de embeddings, versión de Python.
- **Métricas** — `f1_micro/macro/weighted/samples`, precisión y recall, `subset_accuracy`, `hamming_loss`, y por cada género su `f1`, `precision`, `recall`, `support` y `threshold`.
- **Artefactos** — `classification_report.txt`, `thresholds.csv`, `metrics.json` y el bundle completo del modelo empaquetado como `mlflow.pyfunc`.

---

### Paso 3 — Promover una versión

El alias `@champion` es el único interruptor de despliegue. Se puede mover desde la UI (**Models → genre-classifier → versión → Aliases**) o por código:

```bash
uv run python -c "
from mlflow.tracking import MlflowClient
MlflowClient('http://localhost:5000').set_registered_model_alias('genre-classifier','champion','3')
"
```

**Rollback**: apuntar `@champion` a la versión anterior y relanzar el pipeline. No hay que revertir ningún commit.

---

### Paso 4 — Ejecutar el CD

El `Jenkinsfile` corre estas etapas:

1. **Checkout**
2. **Verify Toolchain** — comprueba Docker, construye la imagen cliente de MLflow si falta y verifica que el tracking server responda.
3. **Fetch Model from Registry** — resuelve `genre-classifier@champion`, **valida el gate de calidad** y descarga los artefactos a `models/`.
4. **Build Docker Image** — construye con tags `:<BUILD_NUMBER>` y `:model-v<VERSION>`, y labels con el `run_id`, el commit y el `f1_macro`.
5. **Push Docker Image** — publica ambos tags en Docker Hub.
6. **Record Deployment in MLflow** — etiqueta la versión del modelo con la imagen, el build y el commit que la desplegaron.

Parámetros del job (Jenkins los expone a partir de la segunda ejecución):

| Parámetro | Default | Para qué |
|---|---|---|
| `MODEL_ALIAS` | `champion` | Desplegar otro alias, p. ej. `challenger` |
| `MIN_F1_MACRO` | `0.30` | Umbral del gate de calidad |
| `PUSH_IMAGE` | `true` | Desmarcar para construir sin publicar |

Los scripts del pipeline corren en un **contenedor efímero** con el cliente de MLflow, así que el agente Jenkins solo necesita Docker — ni Python ni `mlflow` instalados.

Si el modelo no pasa el gate, el build se detiene antes de construir nada y el log dice exactamente qué métrica falló.

---

### Trazabilidad en ambos sentidos

Desde una imagen, saber qué modelo lleva dentro:

```bash
docker inspect leuribe2/devops-genre-class:model-v3 \
  --format '{{json .Config.Labels}}'
```

Desde MLflow, saber qué imagen está corriendo una versión: los tags `deployed_image`, `jenkins_build_url` y `git_commit` de la versión en el registry.

---

### Notas de implementación

- **`mlflow-skinny`, no `mlflow`.** El paquete completo fija `pandas<3` y este proyecto usa `pandas>=3`. El cliente skinny no tiene esa restricción y cubre tracking, registry y `pyfunc.log_model`. El servidor corre aparte con `mlflow` completo ([docker/mlflow/Dockerfile](docker/mlflow/Dockerfile)).
- **El modelo se registra como `mlflow.pyfunc`.** No es un solo estimador de sklearn sino seis artefactos que solo sirven juntos, así que [src/mlflow_model.py](src/mlflow_model.py) los envuelve en un `PythonModel`. Los nombres de archivo dentro del bundle son los mismos que espera `src/model_utils.load_artifacts()`, por lo que la app Streamlit no cambió en nada.
- **`--allowed-hosts` en el compose.** MLflow 3 responde `403 Invalid Host header` a cualquier Host que no esté en su allowlist. Sin `mlflow:5000` en esa lista, los contenedores del pipeline no pueden hablar con el servidor.
- **`models/` ya no está en git.** Lo genera `train.py` localmente y lo baja el CD desde el registry.

---

### Solución de problemas de MLflow

| Problema | Solución |
|---|---|
| `403 Invalid Host header` | Falta el host en `--allowed-hosts` del servicio `mlflow` en `docker-compose.yml` |
| `Registered model alias champion not found` | Ninguna versión está promovida: `uv run python train.py --promote-alias champion` |
| Jenkins no resuelve `http://mlflow:5000` | El contenedor no está en la red: `docker network connect mlops jenkins` |
| `403 Invalid Host header` desde otra máquina | Añadir esa IP a `MLFLOW_ALLOWED_HOSTS` en el `.env` y `docker compose up -d mlflow` |
| No se alcanza la UI remota | Falta abrir el puerto 5000 en el firewall del host que hospeda MLflow |
| El build falla en el gate | El modelo `@champion` es peor que `MIN_F1_MACRO`; promover otra versión o ajustar el umbral |
| `UnicodeEncodeError` al entrenar en Windows | Ya mitigado en `train.py`; si aparece en otro script, exportar `PYTHONUTF8=1` |
