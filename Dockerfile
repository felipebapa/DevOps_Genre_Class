FROM python:3.14-slim

WORKDIR /app

# System packages potentially required by Python dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from its official container image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency definitions first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime dependencies
RUN uv sync --frozen --no-dev

# Copy application source
COPY app.py train.py ./
COPY src ./src

# Copy the dataset required by train.py
COPY notebooks/KLUSTERS.xlsx ./notebooks/KLUSTERS.xlsx

# Train the model and generate models/*
RUN uv run python train.py

# Streamlit port
EXPOSE 8501

# Run the web application
CMD ["uv", "run", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
