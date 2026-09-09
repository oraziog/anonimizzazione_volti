# ── Builder ─────────────────────────────────────────────────────────────────
# ort 2.0.0-rc.13 requires Rust ≥ 1.88. The `retraining` feature additionally
# needs Python (Debian bookworm ships 3.11, which pyo3 0.21 supports) because
# the training script and PyTorch run through an embedded interpreter.
FROM rust:1.88-bookworm AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-dev python3-venv pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Build the service (spec §6). CPU ONNX Runtime binaries are fetched by ort's
# download-binaries feature at build time; PYO3_NO_PYTHON avoids surprises
# with stray interpreters.
# Default: nightly retraining. Extend via build arg (e.g. `FEATURES=retraining,s3`
# for the S3 backend, or `retraining,cuda` for GPU — see Dockerfile.gpu).
ARG FEATURES=retraining
ENV PYO3_PYTHON=python3
RUN cargo build --release --features "$FEATURES"

# ── Runtime ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU wheels (required by python/retrain.py for the nightly
# fine-tuning + ONNX export).
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir torchvision pillow

WORKDIR /app

# Binary + the ONNX Runtime shared library copied by ort's copy-dylibs.
COPY --from=builder /app/target/release/anonimizzazione_volti /usr/local/bin/anonimizzazione_volti
COPY --from=builder /app/target/release/libonnxruntime.so* /usr/local/lib/
COPY --from=builder /app/python/retrain.py /app/python/retrain.py

ENV LD_LIBRARY_PATH=/usr/local/lib \
    RETRAIN_SCRIPT_PATH=/app/python/retrain.py \
    RUST_LOG=info

# Persistent volumes (models survive restarts; the database and datasets too).
VOLUME ["/app/models", "/app/data", "/app/dataset_falsi_positivi", "/app/dataset_seed"]

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

ENTRYPOINT ["anonimizzazione_volti"]
