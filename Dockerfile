FROM python:3.11-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md ./
COPY cutible/ cutible/

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["cutible-api", "--host", "0.0.0.0", "--port", "8000"]
