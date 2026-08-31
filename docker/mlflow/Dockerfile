# Imagen del tracking server de MLflow.
# Se usa para dos cosas:
#   1. Levantar el servidor (docker compose up -d mlflow)
#   2. Ejecutar los scripts del CD como contenedor efímero, para que el agente
#      Jenkins no necesite tener Python ni mlflow instalados.
FROM python:3.12-slim

ARG MLFLOW_VERSION=3.15.1

RUN pip install --no-cache-dir \
        "mlflow==${MLFLOW_VERSION}" \
        "psycopg2-binary==2.9.10"

RUN mkdir -p /mlflow/store /mlflow/artifacts

WORKDIR /mlflow

EXPOSE 5000
