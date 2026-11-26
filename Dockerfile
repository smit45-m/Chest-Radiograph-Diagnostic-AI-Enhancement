# Stage 1: Build Frontend (React + Vite + Tailwind v4)
FROM node:22-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python Runtime with CPU-Optimized PyTorch & Clinical Ensemble
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/appuser \
    TORCH_HOME=/app/storage/torch_cache \
    CXR_DB_PATH=/app/storage/cxr_clinical.db \
    CXR_ENSEMBLE_MODE=full \
    GEMINI_FLASH_MODEL=gemini-3.5-flash-lite \
    GROQ_LLM_MODEL=openai/gpt-oss-20b

WORKDIR /app

# Install minimal runtime libraries for OpenCV, PIL, and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-optimized PyTorch first (keeps container lean & fast on AWS EC2)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

# Install remaining clinical imaging, FastAPI, Gemini & Groq dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root app user & persistent storage/cache directories (separate from /app/data/benchmark_test_set)
RUN groupadd -r appuser && useradd -r -g appuser -m appuser \
    && mkdir -p /app/storage/torch_cache /app/outputs/benchmark_results /home/appuser/.torchxrayvision

# Copy application source, configs, benchmark samples, evaluation report & compiled SPA
COPY api/ ./api/
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/benchmark_test_set/ ./data/benchmark_test_set/
COPY outputs/benchmark_results/full_evaluation_report.json ./outputs/benchmark_results/
COPY outputs/benchmark_results/nih_generalization_study.json ./outputs/benchmark_results/
COPY --from=frontend-builder /build/web/ ./web/

RUN chown -R appuser:appuser /app /home/appuser

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=8s --start-period=90s --retries=3 \
  CMD curl -f http://localhost:8000/api/v1/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
