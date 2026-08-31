"""
Descarga desde el Model Registry de MLflow la versión apuntada por un alias
(por defecto `@champion`), valida que cumpla el gate de calidad y deja los
artefactos en `models/` para que el `docker build` los hornee en la imagen.

Lo ejecuta el Jenkinsfile dentro de un contenedor efímero, así que el agente
Jenkins no necesita Python ni mlflow instalados.

Salida:
  models/*.joblib, models/threshold_vec.npy   artefactos del modelo
  models/MODEL_VERSION.env                    metadatos que Jenkins lee para taguear

Códigos de salida:
  0  todo bien
  1  error de conexión / alias inexistente / artefactos incompletos
  2  el modelo no pasó el gate de métricas
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

# Lo que la app necesita para arrancar (src/model_utils._REQUIRED)
REQUIRED_FILES = [
    "classifier.joblib",
    "tfidf.joblib",
    "scaler_emb.joblib",
    "scaler_final.joblib",
    "mlb.joblib",
    "threshold_vec.npy",
]

GATE_EXIT_CODE = 2


def parse_args():
    p = argparse.ArgumentParser(description="Trae el modelo champion desde MLflow.")
    p.add_argument("--model-name", default="genre-classifier")
    p.add_argument("--alias", default="champion")
    p.add_argument("--dest", default="models", help="Directorio destino.")
    p.add_argument(
        "--min-f1-macro",
        type=float,
        default=0.0,
        help="Gate: falla el build si f1_macro de la corrida es menor a este valor.",
    )
    p.add_argument(
        "--min-f1-micro",
        type=float,
        default=0.0,
        help="Gate: falla el build si f1_micro de la corrida es menor a este valor.",
    )
    return p.parse_args()


def resolve_version(client, model_name, alias):
    try:
        return client.get_model_version_by_alias(model_name, alias)
    except Exception as exc:  # noqa: BLE001 - queremos un mensaje accionable
        print(f"ERROR: no se pudo resolver '{model_name}@{alias}': {exc}", file=sys.stderr)
        print(
            "\nRevisa que exista una versión promovida:\n"
            "  python train.py --promote-alias champion\n"
            "o asigna el alias desde la UI de MLflow (Models -> versión -> Aliases).",
            file=sys.stderr,
        )
        sys.exit(1)


def check_gate(client, run_id, args):
    """Compara las métricas de la corrida que produjo el modelo contra el gate."""
    metrics = client.get_run(run_id).data.metrics
    gates = [
        ("f1_macro", args.min_f1_macro),
        ("f1_micro", args.min_f1_micro),
    ]

    failed = []
    for name, minimum in gates:
        if minimum <= 0:
            continue
        value = metrics.get(name)
        if value is None:
            failed.append(f"  {name}: la corrida no registró esta métrica")
        elif value < minimum:
            failed.append(f"  {name}: {value:.4f} < mínimo requerido {minimum:.4f}")
        else:
            print(f"  gate {name}: {value:.4f} >= {minimum:.4f}  OK")

    if failed:
        print("\nERROR: el modelo no pasa el gate de calidad:", file=sys.stderr)
        print("\n".join(failed), file=sys.stderr)
        sys.exit(GATE_EXIT_CODE)

    return metrics


def download_bundle(model_name, alias, dest: Path):
    """Baja el modelo pyfunc y aplana su carpeta `artifacts/` en `dest`."""
    uri = f"models:/{model_name}@{alias}"
    with tempfile.TemporaryDirectory() as tmp:
        print(f"Descargando {uri} ...")
        local_root = Path(mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp))

        bundle_dir = local_root / "artifacts"
        if not bundle_dir.is_dir():
            print(
                f"ERROR: el modelo no contiene la carpeta 'artifacts/' esperada "
                f"(contenido: {[p.name for p in local_root.iterdir()]})",
                file=sys.stderr,
            )
            sys.exit(1)

        dest.mkdir(parents=True, exist_ok=True)
        for item in bundle_dir.iterdir():
            if item.is_file():
                shutil.copy2(item, dest / item.name)

    missing = [f for f in REQUIRED_FILES if not (dest / f).exists()]
    if missing:
        print(f"ERROR: faltan artefactos en el bundle: {missing}", file=sys.stderr)
        sys.exit(1)


def write_env_file(dest: Path, model_name, alias, version, metrics):
    lines = [
        f"MODEL_NAME={model_name}",
        f"MODEL_ALIAS={alias}",
        f"MODEL_VERSION={version.version}",
        f"MODEL_RUN_ID={version.run_id}",
        f"MODEL_F1_MACRO={metrics.get('f1_macro', 0.0):.4f}",
        f"MODEL_F1_MICRO={metrics.get('f1_micro', 0.0):.4f}",
    ]
    (dest / "MODEL_VERSION.env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dest = Path(args.dest)

    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    client = MlflowClient()

    version = resolve_version(client, args.model_name, args.alias)
    print(
        f"Resuelto: {args.model_name}@{args.alias} -> "
        f"version {version.version} (run {version.run_id})"
    )

    metrics = check_gate(client, version.run_id, args)
    download_bundle(args.model_name, args.alias, dest)
    write_env_file(dest, args.model_name, args.alias, version, metrics)

    print(f"\nArtefactos listos en {dest}/ para el docker build:")
    for f in sorted(REQUIRED_FILES):
        size_kb = (dest / f).stat().st_size / 1024
        print(f"  {f:<24} {size_kb:>10.1f} KB")


if __name__ == "__main__":
    main()
