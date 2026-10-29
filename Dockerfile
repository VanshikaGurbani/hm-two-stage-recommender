# Serving image. Data is mounted at runtime (it is large and licensed); see README.
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml ./
COPY hmrec ./hmrec
RUN pip install --no-cache-dir torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir ".[serve]"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s CMD python -c "import httpx;httpx.get('http://127.0.0.1:8000/health').raise_for_status()"
# mount the three Kaggle CSVs at /app/data, e.g.  docker run -p 8000:8000 -v ./data:/app/data hmrec
CMD ["uvicorn", "hmrec.api:app", "--host", "0.0.0.0", "--port", "8000"]
