# syntax=docker/dockerfile:1.7
# One image for the three services (pipeline, api, dashboard); multi-arch (amd64 + arm64 Ampere A1).
FROM ghcr.io/astral-sh/uv:0.13.7 AS uv

FROM python:3.12-slim-bookworm
ARG TARGETARCH
ARG SUPERCRONIC_VERSION=v0.2.51
ARG SUPERCRONIC_SHA256_amd64=cc232108374e8e2b537045f2d5eaa1229f619db314c3ba6a14d38b35dc0aa123
ARG SUPERCRONIC_SHA256_arm64=4ae22576a78223d8ccdf5f3b3660a5cbb5288416bf0ca96afc4a36d195932b94
# Provenance and protocol state, filled in by CI at build time (docs/adr/0006).
ARG GIT_SHA=dev
ARG PROTOCOL_TAGS=""
ARG FROZEN_CONFIG_SHA256=""

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_COMPILE_BYTECODE_TIMEOUT=600 \
    UV_LINK_MODE=copy \
    UV_NATIVE_TLS=1 \
    UV_SYSTEM_CERTS=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TZ=UTC

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl rclone \
 && rm -rf /var/lib/apt/lists/*

# Optional extra root certificate, for a build behind a TLS-inspecting proxy or antivirus
# (docs/operations.md): docker build --secret id=extra_ca,src=<root.pem> .
# Without the secret (CI) this step does nothing.
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -s /run/secrets/extra_ca ]; then \
      cp /run/secrets/extra_ca /usr/local/share/ca-certificates/extra-ca.crt \
      && update-ca-certificates; \
    fi

RUN set -eux; \
    curl -fsSLo /usr/local/bin/supercronic \
      "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/supercronic-linux-${TARGETARCH}"; \
    eval "expected=\${SUPERCRONIC_SHA256_${TARGETARCH}}"; \
    echo "${expected}  /usr/local/bin/supercronic" | sha256sum -c -; \
    chmod +x /usr/local/bin/supercronic

COPY --from=uv /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY config ./config
COPY dashboards ./dashboards
COPY scripts ./scripts
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev \
 && chmod +x scripts/*.sh

RUN useradd --create-home --uid 1000 ofs && mkdir -p /data && chown ofs /data
USER ofs
ENV OFS_ROOT=/app \
    OFS_DATA_DIR=/data \
    OFS_CODE_VERSION=${GIT_SHA} \
    OFS_PROTOCOL_TAGS=${PROTOCOL_TAGS} \
    OFS_FROZEN_CONFIG_SHA256=${FROZEN_CONFIG_SHA256} \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/matplotlib
VOLUME ["/data"]
CMD ["ofs", "--help"]
