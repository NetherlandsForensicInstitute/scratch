# scratch

API and tooling for forensic ballistics comparison.

Provides a FastAPI service and core algorithms to compare toolmarks on bullets (striations) and cartridge cases (
impressions).

## Quickstart

```bash
git clone https://github.com/NetherlandsForensicInstitute/scratch.git
cd scratch

uv venv
uv sync --all-packages --frozen
uv run fastapi dev src/main.py
```

API will be available at: http://localhost:8000/docs (Swagger UI)

## Architecture

The system is split into:

- **API layer (`src/`)** – HTTP interface, file handling, orchestration, for documentation of the rest endpoitns, check
  /docs in the api
- **Core library (`scratch-core`)** – algorithms and signal processing [
  `scratch-core` documentation](./packages/scratch-core/README.md)
- **Stats modules** – external statistical evaluation

# Development

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

#### pre-commit

Install and set up pre-commit hooks:

```bash
# Install pre-commit
pre-commit install
pre-commit install-hooks
```

### Usage

#### Start API Server

```bash
just api
```

#### formatting

```bash
just check-quality
```

#### static typechecking

for type checking we use `Pyright`

```bash
just check-static
```

for more `just` commands available in this project run:

```bash
just help
```

### Project Structure

The backend is built with FastAPI and loosely follows the structure
from [fastapi-best-pactices](https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#fastapi-best-practices-)

This service is designed to operate as a worker/slave service for a larger Java application, which acts as the master
system.
Because of this architecture, the service itself does not maintain its own database. All state and orchestration are
handled by the Java application.

Instead, this service focuses solely on processing tasks (such as file conversion). It uses a local temp directory for
intermediate file handling.

When a file is processed successfully, the API returns a URL.
This URL can then be used to retrieve the generated or converted file.