# scratch

API and tooling for forensic ballistics comparison.

Provides a FastAPI service and core algorithms to compare toolmarks on bullets (striations) and cartridge cases
(impressions).

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

- **API layer (`src/`)** – HTTP interface, file handling, orchestration.
  For documentation of the REST endpoints, check `/docs` in the running API.
- **Core library (`scratch-core`)** – algorithms and signal processing.
  See [`scratch-core` documentation](./packages/scratch-core/README.md).
- **Stats modules** – likelihood ratio calculation and statistical evaluation (external package).

## Development

This project uses **[Just](https://github.com/casey/just)** as a command runner.
Dependencies are managed with **[uv](https://github.com/astral-sh/uv)**, and code quality tools are handled
via **[pre-commit](https://pre-commit.com/)**.

### Setup

Before continuing, make sure to have **[Just](https://github.com/casey/just)** and
**[uv](https://github.com/astral-sh/uv)** installed.

#### uv

It could be that you need to use a private package index proxy. If that is the case, add the following:

```toml
# ~/.config/uv/uv.toml
[[index]]
url = "https://<your domain>/<route>/simple"
default = true
```

#### pre-commit

Install and set up pre-commit hooks:

```bash
pre-commit install
pre-commit install-hooks
```

### Usage

#### Start API Server

```bash
just api
```

#### Formatting

```bash
just check-quality
```

#### Static Typechecking

For type checking we use `Pyright`:

```bash
just check-static
```

For all available `just` commands, run:

```bash
just help
```

### Project Structure

The backend is built with FastAPI and loosely follows the structure from
[fastapi-best-practices](https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#fastapi-best-practices-).

This service is designed to operate as a worker service for a larger Java application, which acts as the
coordinating system. Because of this architecture, the service itself does not maintain its own database.
All state and orchestration are handled by the Java application.

This service focuses solely on processing tasks such as mark comparison and score calculation. It uses a local
temp directory for intermediate file handling.

When a file is processed successfully, the API returns a URL. This URL can then be used to retrieve the
generated or converted file.
