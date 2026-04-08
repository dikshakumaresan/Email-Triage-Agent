# Dockerfile
# ----------
# Containerizes the Email Triage Environment for HuggingFace Spaces.
#
# Build:  docker build -t email-triage-env .
# Run:    docker run -p 7860:7860 email-triage-env

# ---- Base image ----
# Use slim Python 3.11 to keep image size small
FROM python:3.11-slim

# ---- Labels ----
LABEL name="email-triage-env"
LABEL version="1.0.0"
LABEL description="OpenEnv Email Triage Environment"

# ---- Environment variables ----
# Prevents Python from buffering stdout/stderr (important for logs)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Port HuggingFace Spaces expects
ENV PORT=7860

# ---- Working directory ----
WORKDIR /app

# ---- Install dependencies ----
# Copy requirements first (Docker caches this layer if requirements don't change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Copy source code ----
COPY emails.py .
COPY tasks.py .
COPY graders.py .
COPY environment.py .
COPY server.py .
COPY inference.py .
COPY openenv.yaml .

# ---- Expose port ----
EXPOSE 7860

# ---- Health check ----
# Docker will ping /health every 30s to confirm the server is alive
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" \
  || exit 1

# ---- Start command ----
# Runs the FastAPI server on port 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
