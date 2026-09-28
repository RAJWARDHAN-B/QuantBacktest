# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---- builder: build wheels for the project and all dependencies ----
FROM base AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels ".[dev]"

# ---- system: OS packages (LibreOffice for Excel recalc + PDF export) ----
FROM base AS system
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libreoffice-calc \
        libreoffice-impress \
        fonts-liberation \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 analyst

# ---- runtime: CLI image ----
FROM system AS runtime
WORKDIR /app
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels ib-agent
COPY config ./config
RUN mkdir -p /app/outputs /app/.cache && chown -R analyst:analyst /app
ENV IB_AGENT_CONFIG_DIR=/app/config \
    IB_AGENT_CACHE_DIR=/app/.cache \
    IB_AGENT_OUTPUT_DIR=/app/outputs
USER analyst
ENTRYPOINT ["ib-agent"]
CMD ["--help"]

# ---- dev: runtime + test/lint tooling; source is bind-mounted ----
FROM runtime AS dev
USER root
RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels \
    pip install --no-index --find-links=/wheels "ib-agent[dev]"
ENV PYTHONPATH=/app/src \
    RUFF_CACHE_DIR=/tmp/ruff-cache \
    MYPY_CACHE_DIR=/tmp/mypy-cache
USER analyst
ENTRYPOINT []
CMD ["pytest", "-p", "no:cacheprovider"]
