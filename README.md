# scratch

## 🧑‍💻 Development Environment

This project uses **[devenv](https://devenv.sh/getting-started/)** for
environment management and **[Just](https://github.com/casey/just)** as a
command runner. Dependencies are managed with
**[uv](https://github.com/astral-sh/uv)**, and code quality tools are handled
via **[pre-commit](https://pre-commit.com/)**.

Using `devenv` ensures a fully reproducible development setup — including Python
version, dependencies, and tooling — across all machines.

### 🚀 Recommended setup (with devenv)

#### pre-requirements

- [devenv](https://devenv.sh/getting-started/)

```bash
devenv shell
```

Or allow your project to start isolated shell at the point of directory entry

```bash
direnv allow
```

This will:

- Enter the `devenv` shell with the correct environment
- Automatically install **UV**
- Sync all project dependencies
- Install and configure **pre-commit** hooks

You can now start developing right away.

### 🧰 Alternative setup (without devenv)

If you prefer not to use `devenv`, you can still set things up manually:

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

#### pre-commit

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: "v6.0.0"
    hooks:
      - id: check-case-conflict
      - id: check-merge-conflict
      - id: check-toml
      - id: check-yaml
      - id: check-json
      - id: pretty-format-json
        args: [--autofix, --no-sort-keys]
      - id: end-of-file-fixer
      - id: trailing-whitespace

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: "v0.14.1"
    hooks:
      - id: ruff-check
        args: [--exit-non-zero-on-fix]
      - id: ruff-format
```

```bash
# create and copy pre-commit-config.yaml
pre-commit install
pre-commit install-hooks
just check # will run pre-commit for all files (not required)
```

> [!NOTE] Using `devenv` is highly recommended — it guarantees a consistent,
> isolated, and reproducible development environment.

## Usage

### Start API Server

```bash
just api
```

## Project Structure

```mermaid
flowchart LR

  %% ===== Styles =====
  classDef pkg fill:#f5f5f5,stroke:#bbb,stroke-width:1px,color:#333;
  classDef public fill:#e6f7ff,stroke:#1890ff,stroke-width:1px;
  classDef internal fill:#f6ffed,stroke:#52c41a,stroke-width:1px;
  classDef datastore fill:#fff7e6,stroke:#fa8c16,stroke-width:1.5px,stroke-dasharray:3 2;
  classDef optional fill:#fff0f6,stroke:#eb2f96,stroke-dasharray:2 2;
  classDef extTeam fill:#f0f5ff,stroke:#2f54eb;

  %% ===== API Workspace =====
  subgraph API[API]
    direction TB
    api_rest[REST]:::public
    api_sockets[Sockets]:::optional
    api_models[Public Models]:::internal
  end
  class API pkg

  %% ===== Core Workspace =====
  subgraph CORE[Core]
    direction TB
    core_pre[Pre-processors]:::internal
    core_proc[Processors]:::internal
    core_comp[Comparators]:::internal
    core_common[Common]:::public
    core_store[Datastore]:::datastore
  end
  class CORE pkg

  %% ===== Stats Workspace =====
  subgraph STATS[Stats]
    direction TB
    stats_pkg[Stats modules]:::extTeam
  end
  class STATS pkg

  %% ===== Dependencies / Policies =====
  %% API talks only to Core (no direct Stats or Datastore access)
  api_rest --> core_proc
  api_rest --> core_pre
  api_rest --> core_comp
  api_rest --> api_models

  %% Optional Sockets (still to be determined)
  api_sockets -.-> core_pre

  %% Core uses Stats
  core_proc --> stats_pkg
  core_comp --> stats_pkg

  %% Core modules depend on Common
  core_pre --> core_common
  core_proc --> core_common
  core_comp --> core_common
  core_store --> core_common

  %% Modules that need storage access → Datastore
  core_pre --> core_store
  core_proc --> core_store
  core_comp --> core_store
```
