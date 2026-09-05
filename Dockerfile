# Draft Room API — self-contained image serving projections, the exact optimiser and the
# draft/weekly engines over HTTP. A prebuilt model board is copied in, so the image builds
# fast, deterministically and offline; refresh it anytime with `eldraft build`.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY pyproject.toml README.md ./
COPY eldraft ./eldraft
RUN pip install --no-cache-dir ".[api]"

# prebuilt model board + research results, so the API serves immediately, offline and
# deterministically. Refresh anytime with `eldraft build`.
COPY data/board.json ./data/board.json
COPY out/*.json ./out/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=3).status==200 else 1)"
CMD ["uvicorn", "eldraft.api:app", "--host", "0.0.0.0", "--port", "8000"]
