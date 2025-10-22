# scratch

## Starting API

Run the REST API with

### Just (recommended)

```bash
just api
```

### UV:

```bash
uv run fastapi dev src/api/__main__.py
```

### Python:

```bash
python -m fastapi dev src/api/__main__.py
```

## 🧑‍💻 Development Environment

This project uses **[devenv](https://devenv.sh/)** for environment management
and **[Just](https://github.com/casey/just)** as a command runner. Dependencies
are managed with **[uv](https://github.com/astral-sh/uv)**, and code quality
tools are handled via **[pre-commit](https://pre-commit.com/)**.

Using `devenv` ensures a fully reproducible development setup — including Python
version, dependencies, and tooling — across all machines.

### 🚀 Recommended setup (with devenv)

If you have [Nix](https://nixos.org/) and [devenv](https://devenv.sh/)
installed:

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
- configure **PDM**

You can now start developing right away.

### 🧰 Alternative setup (without devenv)

If you prefer not to use `devenv`, you can still set things up manually:

- Ensure you have UV or PDM installed
- Ensure you have Just installed
- copy #pre-commit configuration to your local root directory

```bash
# Create and sync a virtual environment
uv run pre-commit install
just install
```

> [Note!] Using `devenv` is highly recommended — it guarantees a consistent,
> isolated, and reproducible development environment.

### PDM

If you must use **PDM**, go right ahead; your devenv shell already has it
available

If you are running a manually install then please configure **PDM** before
embarking on a new PDM adventure

```bash
pdm config use_uv true
pdm config python.install_root $(uv python dir --color never)
```
