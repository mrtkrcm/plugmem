FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -e .
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')"
EXPOSE 8080
CMD ["uvicorn", "plugmem.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
