import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP-DGO | Clasificador de Géneros",
    page_icon="🎬",
    layout="wide",
)

DATA_DIR = Path("notebooks")
RESULTS_CSV = DATA_DIR / "Last_resultados_modelo_hibrido_embeddings_tfidf.csv"

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["📖  Introducción", "📊  Métricas del Modelo", "🎬  Predicción de Géneros"]
)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — INTRODUCCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.title("Clasificador de Géneros Audiovisuales")
    st.subheader("Pipeline NLP Híbrido — Embeddings Semánticos + TF-IDF sobre Lemmas")

    st.markdown(
        """
        Este sistema predice automáticamente los géneros de una película o serie a partir
        de su **sinopsis (plot)** en inglés, combinando comprensión semántica profunda con
        señales léxicas.

        ---
        """
    )

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### Dos ramas de señal")
        st.markdown(
            """
            | Rama | Herramienta | Qué captura |
            |---|---|---|
            | **Semántica** | `all-MiniLM-L6-v2` | Significado y contexto profundo |
            | **Léxica** | TF-IDF sobre lemmas | Vocabulario característico por género |

            Las dos ramas se **fusionan** antes de alimentar al clasificador, de modo que
            el modelo aprovecha tanto la semántica como los patrones de vocabulario.
            """
        )

        st.markdown("### Pipeline paso a paso")
        st.markdown(
            """
            1. **Preprocesamiento con spaCy (`en_core_web_sm`)**
               - Normalización unicode y limpieza de metadatos (`::autor`)
               - NER: nombres → `_PERSON_`, fechas → `_DATE_`, números → `_NUMBER_`, etc.
               - Lematización filtrada por PoS: solo `NOUN`, `PROPN`, `VERB`, `ADJ`
               - Stopwords eliminadas (excepciones: *no, not, never, without, against*)

            2. **Extracción de features**
               - Embeddings de 384 dimensiones con `SentenceTransformer`
               - TF-IDF: hasta 30 000 features, n-gramas 1–3, `sublinear_tf=True`

            3. **Escalado y fusión**
               - `StandardScaler` sobre embeddings (preserva escala relativa)
               - Concatenación `hstack([embeddings, tfidf])`
               - `MaxAbsScaler` sobre la matriz final combinada

            4. **Clasificación multilabel**
               - `OneVsRestClassifier(LogisticRegression)` — un clasificador por género
               - `class_weight="balanced"` para manejar el desbalance de clases
               - **Threshold por género**: ajustado sobre el set de validación (máx. F1)

            5. **Decodificación de predicciones**
               - Siempre se incluye el género de mayor probabilidad
               - Se agregan géneros con prob ≥ 80% de la del género principal
               - Máximo 3 géneros por título
            """
        )

    with col_b:
        st.markdown("### Diagrama del pipeline")
        st.image(
            "https://mermaid.ink/img/pako"
            + "/eyJjb2RlIjoiZmxvd2NoYXJ0IFREXG4gICAgQVtQbG90IC8gU2lub3BzaXNdIC0tPiBCW1ByZXByb2Nlc2FtaWVudG8gc3BhQ3ldXG4gICAgQiAtLT4gQ1tMZW1hc11cbiAgICBCIC0tPiBEW1RleHRvIG5vcm1hbGl6YWRvXVxuICAgIEMgLS0-IEVbVEYtSURGXVxuICAgIEQgLS0-IEZbU2VudGVuY2VUcmFuc2Zvcm1lcl1cbiAgICBFIC0tPiBHW0hTdGFja11cbiAgICBGIC0tPiBHXG4gICAgRyAtLT4gSFtNYXhBYnNTY2FsZXJdXG4gICAgSCAtLT4gSVtPbmVWc1Jlc3QgTG9nUmVnXVxuICAgIEkgLS0-IEpbVGhyZXNob2xkIHBvciBnw6luZXJvXVxuICAgIEogLS0-IEtbR8OpbmVyb3MgcHJlZGljaG9zXVxuIiwibWVybWFpZCI6e30sInVwZGF0ZUVkaXRvciI6ZmFsc2V9",
            caption="Pipeline híbrido de clasificación",
            use_container_width=True,
        )

        st.markdown("### Dataset y splits")
        st.markdown(
            """
            | Set | Proporción | Uso |
            |---|---|---|
            | **Train** | 70% | Ajuste de parámetros |
            | **Validación** | 15% | Calibración de thresholds |
            | **Test** | 15% | Evaluación final |

            - **Fuente:** `KLUSTERS.xlsx`
            - **Géneros:** 27 categorías (Action, Comedy, Drama, Horror, Sci-Fi…)
            - **Modelo de embeddings:** `all-MiniLM-L6-v2` (Hugging Face)
            """
        )

        st.markdown("### Equipo")
        st.markdown(
            """
            - Luis Jorge García Camargo
            - Luis Eduardo Uribe
            - Felipe Barreto Patiño

            *Proyecto Final — Procesamiento de Lenguaje Natural*
            *Maestría en Inteligencia Artificial*
            """
        )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.title("Métricas del Modelo")

    if not RESULTS_CSV.exists():
        st.warning(
            "No se encontró el archivo de resultados. "
            "Ejecuta el notebook o `train.py` primero."
        )
    else:
        @st.cache_data
        def load_metrics():
            df = pd.read_csv(RESULTS_CSV)

            def parse_col(s):
                return [g.strip() for g in str(s).split(",") if g.strip()]

            true_genres = df["true_genres"].apply(parse_col).tolist()
            pred_genres = df["predicted_genres"].apply(parse_col).tolist()

            all_genres = sorted(
                {g for genres in true_genres + pred_genres for g in genres}
            )

            mlb = MultiLabelBinarizer(classes=all_genres)
            y_true = mlb.fit_transform(true_genres)
            y_pred = mlb.transform(pred_genres)

            prec, rec, f1, sup = precision_recall_fscore_support(
                y_true, y_pred, zero_division=0
            )
            metrics_df = pd.DataFrame(
                {
                    "Género": all_genres,
                    "Precision": prec.round(2),
                    "Recall": rec.round(2),
                    "F1-Score": f1.round(2),
                    "Support": sup.astype(int),
                }
            )

            avgs = {
                avg: precision_recall_fscore_support(
                    y_true, y_pred, average=avg, zero_division=0
                )
                for avg in ("micro", "macro", "weighted")
            }
            return metrics_df, avgs

        metrics_df, avgs = load_metrics()

        # ── Summary cards ──────────────────────────────────────────────────
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Micro F1",    f"{avgs['micro'][2]:.3f}")
        c2.metric("Macro F1",    f"{avgs['macro'][2]:.3f}")
        c3.metric("Weighted F1", f"{avgs['weighted'][2]:.3f}")
        c4.metric("Micro P",     f"{avgs['micro'][0]:.3f}")
        c5.metric("Micro R",     f"{avgs['micro'][1]:.3f}")
        c6.metric("Géneros",     str(len(metrics_df)))

        st.markdown("---")

        # ── Bar chart + table ──────────────────────────────────────────────
        col_chart, col_table = st.columns([1.6, 1])

        with col_chart:
            sorted_df = metrics_df.sort_values("F1-Score", ascending=True)
            colors = [
                "#2ecc71" if v >= 0.70 else "#f39c12" if v >= 0.50 else "#e74c3c"
                for v in sorted_df["F1-Score"]
            ]
            fig = go.Figure(
                go.Bar(
                    x=sorted_df["F1-Score"],
                    y=sorted_df["Género"],
                    orientation="h",
                    marker_color=colors,
                    text=sorted_df["F1-Score"].map("{:.2f}".format),
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="F1-Score por Género",
                xaxis_title="F1-Score",
                xaxis_range=[0, 1.15],
                height=700,
                margin={"l": 10, "r": 80, "t": 50, "b": 20},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("### Detalle por género")

            def color_f1(val):
                if val >= 0.70:
                    return "background-color:#c6efce;color:#276221"
                elif val >= 0.50:
                    return "background-color:#ffeb9c;color:#9c5700"
                return "background-color:#ffc7ce;color:#9c0006"

            styled = (
                metrics_df.sort_values("F1-Score", ascending=False)
                .style
                .map(color_f1, subset=["F1-Score"])
                .format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-Score": "{:.2f}"})
            )
            st.dataframe(styled, use_container_width=True, height=700)

        st.markdown("---")

        # ── Precision vs Recall scatter ────────────────────────────────────
        st.markdown("### Precision vs. Recall por género")
        fig2 = px.scatter(
            metrics_df,
            x="Recall",
            y="Precision",
            text="Género",
            size="Support",
            color="F1-Score",
            color_continuous_scale="RdYlGn",
            hover_data={"F1-Score": ":.2f", "Support": True},
            title="Precision vs. Recall  (tamaño = support · color = F1)",
            range_x=[-0.05, 1.05],
            range_y=[-0.05, 1.05],
        )
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.add_shape(
            type="line", x0=0, y0=0, x1=1, y1=1,
            line={"color": "gray", "dash": "dash", "width": 1},
        )
        fig2.update_layout(height=520, plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

        # ── Thresholds table ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Thresholds aprendidos por género")
        thr_path = DATA_DIR / "thresholds_por_genero.csv"
        if thr_path.exists():
            thr_df = pd.read_csv(thr_path).sort_values("best_threshold", ascending=False)
            fig3 = px.bar(
                thr_df,
                x="genre",
                y="best_threshold",
                title="Umbral de clasificación por género",
                labels={"genre": "Género", "best_threshold": "Threshold"},
                color="best_threshold",
                color_continuous_scale="Blues",
            )
            fig3.add_hline(y=0.5, line_dash="dash", line_color="red",
                           annotation_text="0.5 baseline")
            fig3.update_layout(height=400, xaxis_tickangle=-45,
                               plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.title("Predicción de Géneros")

    from src.model_utils import load_artifacts, models_exist, predict_genres

    if not models_exist():
        st.warning(
            "No se encontraron los artefactos del modelo entrenado. "
            "Ejecuta el script de entrenamiento primero:"
        )
        st.code("python train.py", language="bash")
        st.info(
            "El entrenamiento tarda ~5-10 minutos dependiendo del hardware. "
            "Una vez completado, recarga esta página."
        )
    else:
        @st.cache_resource(show_spinner="Cargando spaCy...")
        def get_nlp():
            from src.preprocessing import load_spacy_model
            return load_spacy_model()

        @st.cache_resource(show_spinner="Cargando SentenceTransformer...")
        def get_emb_model():
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")

        @st.cache_resource(show_spinner="Cargando modelo clasificador...")
        def get_artifacts():
            return load_artifacts()

        nlp = get_nlp()
        emb_model = get_emb_model()
        artifacts = get_artifacts()

        st.markdown(
            "Ingresa la sinopsis de una película o serie **en inglés** para predecir sus géneros."
        )

        # ── Sample plots ───────────────────────────────────────────────────
        SAMPLES = {
            "(ninguno)": "",
            "Acción / Thriller": (
                "A former CIA operative is forced out of retirement when his daughter is "
                "kidnapped by a powerful criminal organization. Using his deadly skills and "
                "relentless determination, he hunts down the kidnappers across Europe."
            ),
            "Comedia romántica": (
                "Two awkward strangers are repeatedly thrown together by fate during a series "
                "of increasingly ridiculous misadventures in New York City. Despite their best "
                "efforts to avoid each other, they slowly realize they may have been falling "
                "in love all along."
            ),
            "Terror sobrenatural": (
                "A family moves into a remote farmhouse only to discover it was built on an "
                "ancient burial ground. As supernatural events escalate and members of the "
                "family begin to act strangely, the youngest daughter must uncover the dark "
                "secret before it consumes them all."
            ),
            "Ciencia ficción": (
                "In the year 2157, a crew of astronauts aboard a damaged spacecraft must "
                "navigate through a wormhole to find a new habitable planet for mankind. "
                "When they discover an alien signal, they realize they are not alone in "
                "the universe."
            ),
            "Drama histórico": (
                "Based on a true story. During World War II, a brilliant mathematician "
                "leads a secret team tasked with breaking the enemy's unbreakable encryption "
                "machine. Racing against time, he must overcome military skepticism and his "
                "own personal demons to change the course of history."
            ),
        }

        sample_key = st.selectbox("Cargar un ejemplo:", list(SAMPLES.keys()))
        default_text = SAMPLES[sample_key]

        user_text = st.text_area(
            "Sinopsis:",
            value=default_text,
            height=160,
            placeholder="Enter a movie or TV show plot in English...",
        )

        col_btn, col_opts = st.columns([1, 3])
        with col_btn:
            classify = st.button(
                "Clasificar", type="primary", disabled=not user_text.strip()
            )
        with col_opts:
            max_labels = st.slider("Máx. géneros a predecir", 1, 5, 3)
            min_ratio = st.slider("Ratio mínimo vs. género principal (%)", 50, 100, 80) / 100

        if classify and user_text.strip():
            with st.spinner("Procesando sinopsis..."):
                result = predict_genres(
                    user_text.strip(),
                    artifacts,
                    nlp,
                    emb_model,
                    min_ratio=min_ratio,
                    max_labels=max_labels,
                )

            probs = result["probabilities"]
            thresholds = result["thresholds"]
            predicted = result["predicted"]

            # ── Predicted genres ───────────────────────────────────────────
            st.markdown("### Géneros predichos")
            if predicted:
                genre_cols = st.columns(len(predicted))
                for col, genre in zip(genre_cols, predicted):
                    col.metric(label=genre, value=f"{probs[genre]:.1%}")
            else:
                st.info("No se predijo ningún género con la configuración actual.")

            st.markdown("---")

            # ── Full probability chart ─────────────────────────────────────
            st.markdown("### Probabilidades por género")

            genres_sorted = sorted(probs, key=probs.get, reverse=True)
            prob_vals = [probs[g] for g in genres_sorted]
            thr_vals = [thresholds[g] for g in genres_sorted]
            above_thr = [probs[g] >= thresholds[g] for g in genres_sorted]
            bar_colors = ["#2ecc71" if a else "#bdc3c7" for a in above_thr]

            fig = go.Figure()

            # Probability bars
            fig.add_trace(
                go.Bar(
                    x=prob_vals,
                    y=genres_sorted,
                    orientation="h",
                    marker_color=bar_colors,
                    name="Probabilidad",
                    text=[f"{p:.1%}" for p in prob_vals],
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.1%}<extra></extra>",
                )
            )

            # Threshold tick markers
            fig.add_trace(
                go.Scatter(
                    x=thr_vals,
                    y=genres_sorted,
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 14,
                        "color": "crimson",
                        "line": {"width": 2, "color": "crimson"},
                    },
                    name="Umbral",
                    hovertemplate="%{y} umbral: %{x:.2f}<extra></extra>",
                )
            )

            fig.update_layout(
                xaxis_title="Probabilidad",
                xaxis_range=[0, 1.18],
                height=750,
                showlegend=True,
                legend={"x": 0.82, "y": 0.02},
                margin={"l": 10, "r": 90, "t": 20, "b": 20},
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Lemmas expander ────────────────────────────────────────────
            with st.expander("Ver lemmas extraídos por spaCy"):
                st.code(result["lemmas"] or "(sin lemmas)", language="text")
