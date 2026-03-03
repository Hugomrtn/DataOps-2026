FROM --platform=linux/amd64 ghcr.io/astral-sh/uv:debian-slim

WORKDIR /app

COPY uv.lock .
COPY pyproject.toml .

RUN uv sync

COPY app.py .
COPY model.joblib .

CMD ["uv", "run", "fastapi", "run", "api.py"]
