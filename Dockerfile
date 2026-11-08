# Image for the prediction API. It holds only what serving needs: the core packages and the
# "serve" group from pyproject.toml. Training and development tools are left out.
FROM python:3.13-slim

# uv installs the exact versions from uv.lock, so the image matches what was tested.
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface

# A user without root rights runs the server, so a bug in the app cannot change the system.
RUN useradd --create-home app \
    && mkdir -p "$HF_HOME" \
    && chown -R app:app /home/app

WORKDIR /app

# Dependencies come first so Docker reuses this layer until pyproject.toml or uv.lock changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --group serve --no-install-project

# The app user owns the files, so the image does not depend on the file permissions of the
# machine that builds it.
COPY --chown=app:app src ./src
COPY --chown=app:app configs/serving.yaml ./configs/serving.yaml

USER app
EXPOSE 8000

# The first start downloads the model, so the check waits for it (see start-period).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "src.serving.app:create_default_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
