# scratch

API and tooling for forensic ballistics comparison.

Provides a FastAPI service and core algorithms to compare toolmarks on bullets (striations) and cartridge cases (
impressions).

## Quickstart

```bash
git clone https://github.com/NetherlandsForensicInstitute/scratch.git
cd scratch

uv venv
uv sync
uv run fastapi dev src/main.py
```

API will be available at: http://localhost:8000/docs (Swagger UI)

## Architecture

The system is split into:

- **API layer (`src/`)** – HTTP interface, file handling, orchestration
- **Core library (`scratch-core`)** – algorithms and signal processing [
  `scratch-core` documentation](./packages/scratch-core/README.md)
- **Stats modules** – external statistical evaluation

// Add here the draw.io diagram

# Development Environment

This project uses **[Just](https://github.com/casey/just)** as a
command runner. Dependencies are managed with
**[uv](https://github.com/astral-sh/uv)**, and code quality tools are handled
via **[pre-commit](https://pre-commit.com/)**.

### Setup

Before continuing, make sure to have **[Just](https://github.com/casey/just)** and
**[uv](https://github.com/astral-sh/uv)** installed.

#### UV

It could be that you need to use a private package index proxy, if that is the
case add the following

```toml
# ~/.config/uv/uv.toml
[[index]]
url = "https://<your domain>/<route>/simple"
default = true
```

```bash
# Create and sync a virtual environment
uv venv
uv sync
source ./.venv/bin/activate
```

```bash
# Install pre-commit
pre-commit install
pre-commit install-hooks
just check-quality # will run pre-commit for all files (not required)
```

### Usage

#### Start API Server

```bash
just api
```

### Project Structure

This project tree
follows: [fastapi-best-pactices](https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#fastapi-best-practices-)