FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
COPY data ./data
COPY models ./models

RUN pip install --no-cache-dir --upgrade pip uv && uv pip install --system -r requirements.txt

EXPOSE 8000 8501
