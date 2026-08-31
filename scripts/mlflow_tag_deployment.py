"""
Cierra el ciclo de trazabilidad: después de publicar la imagen, escribe en la
versión del modelo qué imagen y qué build de Jenkins la desplegaron.

Con esto, desde la UI de MLflow se puede responder "¿qué imagen está corriendo
este modelo?" y desde el tag de la imagen "¿qué corrida lo entrenó?".

Lo ejecuta el Jenkinsfile en un contenedor efímero tras el push.
"""

import argparse
import sys
from datetime import UTC, datetime

from mlflow.tracking import MlflowClient


def parse_args():
    p = argparse.ArgumentParser(description="Marca la versión del modelo como desplegada.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--image", required=True, help="Imagen publicada, con tag.")
    p.add_argument("--build-number", default="", help="BUILD_NUMBER de Jenkins.")
    p.add_argument("--build-url", default="", help="BUILD_URL de Jenkins.")
    p.add_argument("--git-commit", default="", help="Commit del repo que construyó la imagen.")
    return p.parse_args()


def main():
    args = parse_args()
    client = MlflowClient()

    tags = {
        "deployed_image": args.image,
        "deployed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    if args.build_number:
        tags["jenkins_build"] = args.build_number
    if args.build_url:
        tags["jenkins_build_url"] = args.build_url
    if args.git_commit:
        tags["git_commit"] = args.git_commit

    try:
        for key, value in tags.items():
            client.set_model_version_tag(args.model_name, args.version, key, value)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR al etiquetar la versión: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"{args.model_name} v{args.version} etiquetado como desplegado:")
    for key, value in tags.items():
        print(f"  {key} = {value}")


if __name__ == "__main__":
    main()
