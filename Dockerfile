FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.0@sha256:606e70c71c852d03f611b1e56a195d08648507018a7057fab82c4974c4eae105 /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY app.py ./
COPY src ./src
COPY models ./models

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
