# Use slim Python 3.11 for a smaller image footprint
FROM python:3.11-slim

# Labels for metadata
LABEL name="email-triage-env"
LABEL version="1.0.0"
LABEL description="OpenEnv Email Triage Environment"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=7860

# Working directory
WORKDIR /app

# Install dependencies
# We copy requirements.txt first to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THE CRITICAL FIX ---
# This copies EVERYTHING from your repo into the Docker image.
# This ensures pyproject.toml and openenv.yaml are present for 'openenv validate'.
COPY . .

# Expose the port HuggingFace expects
EXPOSE 7860

# Health check to ensure the FastAPI server is running
# Note: Ensure 'requests' is in your requirements.txt
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()" \
  || exit 1

# Start command
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
