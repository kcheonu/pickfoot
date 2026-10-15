FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md alembic.ini ./
COPY src ./src
COPY migrations ./migrations
RUN pip install --no-cache-dir . \
    && mkdir -p /app/models /app/data/persistent /app/data/raw \
    && useradd --create-home --uid 10001 pickfoot \
    && chown -R pickfoot:pickfoot /app

USER pickfoot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/api/v1/health/ready', timeout=3)"
CMD ["uvicorn", "pickfoot.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
