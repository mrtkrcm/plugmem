FROM python:3.12-slim
WORKDIR /app
# Copy only what the package install needs to keep the image lean and the
# build cache layer stable when only docs/tests change.
COPY pyproject.toml README.md LICENSE ./
COPY plugmem ./plugmem
RUN pip install --no-cache-dir .
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')"
EXPOSE 8080
CMD ["uvicorn", "plugmem.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
