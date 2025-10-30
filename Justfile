# List all tasks (default)
help:
  @echo "{{YELLOW}}Yellow: means change, {{RED}}Red: deleting, {{BLUE}}Blue: command execution{{NORMAL}}"
  @echo "Just docs: https://just.systems/man/en/introduction.html"
  @echo "cheatsheet: https://cheatography.com/linux-china/cheat-sheets/justfile/"
  @just --list

# Install the virtual environment and per-commit hooks
install:
  echo "{{BLUE}}{{BOLD}}{{ITALIC}}Creating virtual environment using uv"
  uv sync --all-packages --frozen

# update the virtual environment and per-commit hooks
update:
  echo "{{YELLOW}}{{BOLD}}{{ITALIC}}Updating virtual environment using uv"
  uv lock --upgrade
  uv sync --all-packages
  uv run pre-commit autoupdate

# Run code formatter
format:
  @echo "{{YELLOW}}{{BOLD}}{{ITALIC}}Format python files"
  uv run ruff format

# Run project quality (invoke pre-commit on all files)
check-quality:
  @echo "\n{{BLUE}}{{BOLD}}{{ITALIC}}Checking code quality: Running pre-commit"
  uv run pre-commit run -a

# Run python static code check
check-static:
  @echo "\n{{BLUE}}{{BOLD}}{{ITALIC}}Static type checking: Running ty"
  uv run ty check

  @echo "\n{{BLUE}}{{BOLD}}{{ITALIC}}Checking for obsolete dependencies: Running deptry"
  uv run deptry src

# Run all Project tests
test:
  uv run pytest -m 'not contract_testing' --cov --cov-config=pyproject.toml --cov-report=xml

# Run all endpoints health checks
smoke-test:
  @echo "{{BLUE}}{{BOLD}}{{ITALIC}}Testing code: Running the contract testing"
  just api & sleep 2
  uv run pytest -m 'contract_testing'
  kill `lsof -t -i:8000` 2>/dev/null || true

# Removes version control system dirty files
clean:
  @echo "{{RED}}{{BOLD}}{{ITALIC}}Deleting all dirty files"
  git clean -xfd

# Start API development server
api:
  @echo "{{BLUE}}{{BOLD}}{{ITALIC}}Starting FastAPI development server"
  uv run fastapi dev src/main.py
