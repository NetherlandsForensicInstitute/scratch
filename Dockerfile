FROM python:3.13

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Do not write .pyc files to disk
ENV PYTHONDONTWRITEBYTECODE=1
# Do not buffer stdout and stderr
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . .

# Configure UV virtual env
# ENV ENV PIP_NO_CACHE_DIR=1
# ENV UV_NO_CACHE=1
# ENV UV_LINK_MODE=copy

RUN uv sync --all-packages --frozen
