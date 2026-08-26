FROM python:3.14-slim AS builder

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install EXACTLY what uv.lock pins — never re-resolve at build time.
# (Incident 2026-07-07: an unpinned `pip install .` here silently picked up
# fastmcp 3.4.3, whose new Host-header guard 421'd all of production.)
# uv.lock pins linux torch to the CPU wheel index via tool.uv.sources, so
# the extra-index-url is required for pip to find the +cpu build; it saves
# ~5 GB of nvidia-cuda-* packages nebula-1 (CPU-only VM) can't use.
RUN pip install --no-cache-dir uv && \
    uv export --frozen --no-dev --no-emit-project -o /tmp/requirements.txt && \
    pip install --no-cache-dir --require-hashes --extra-index-url https://download.pytorch.org/whl/cpu \
        -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-deps .

# Pre-download the embedding model at build time
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

FROM python:3.14-slim

WORKDIR /app

# Copy installed packages and model cache from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/mcp-stolperfalle /usr/local/bin/mcp-stolperfalle
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface
COPY --from=builder /app/src ./src

# Create non-root user with a real HOME so HuggingFace / FastMCP caches resolve.
RUN addgroup --system mcp && \
    adduser --system --ingroup mcp --home /home/mcp mcp && \
    mkdir -p /data && chown mcp:mcp /data && \
    mkdir -p /home/mcp/.cache && \
    mv /root/.cache/huggingface /home/mcp/.cache/huggingface && \
    chown -R mcp:mcp /home/mcp

USER mcp

# Default env for Docker
ENV HOME=/home/mcp \
    HF_HOME=/home/mcp/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/mcp/.cache/huggingface \
    TRANSPORT=http \
    HOST=0.0.0.0 \
    CQ_LOCAL_DB_PATH=/data/stolperstein.db

EXPOSE 8716

# Unauthenticated /health returns 200 when the server is up and migrations
# have finished. No bearer token needed → healthcheck logs don't flood
# with 401s.
# Probe with the PUBLIC Host header, not localhost — so the healthcheck fails
# exactly when external traffic would (the fastmcp 3.4.3 host-guard 421 blind
# spot: localhost stayed 200 while the public hostname was rejected).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python3 -c "import urllib.request,sys; \
req=urllib.request.Request('http://localhost:8716/health', \
headers={'Host':'mcp-stolperfalle.cdit-dev.de'}); \
urllib.request.urlopen(req, timeout=3); sys.exit(0)"

CMD ["mcp-stolperfalle"]
