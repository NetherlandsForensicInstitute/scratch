
# List all tasks (default)
help:
    @echo "{{ YELLOW }}Yellow: means change, {{ RED }}Red: deleting, {{ BLUE }}Blue: command execution{{ NORMAL }}"
    @echo "Just docs: https://just.systems/man/en/introduction.html"
    @echo "cheatsheet: https://cheatography.com/linux-china/cheat-sheets/justfile/"
    @just --list

# Install the virtual environment and per-commit hooks
install:
    @echo "{{ BLUE }}{{ BOLD }}{{ ITALIC }}Creating virtual environment using uv"
    uv sync --all-packages --frozen

# update the virtual environment and per-commit hooks
update:
    @echo "{{ YELLOW }}{{ BOLD }}{{ ITALIC }}Updating virtual environment using uv"
    uv lock --upgrade
    uv sync --all-packages
    uv run pre-commit autoupdate

# Run code formatter
format:
    @echo "{{ YELLOW }}{{ BOLD }}{{ ITALIC }}Format python files"
    uv run ruff format

# Run project quality (invoke pre-commit on all files)
check-quality:
    @echo "\n{{ BLUE }}{{ BOLD }}{{ ITALIC }}Checking code quality: Running pre-commit"
    uv run pre-commit run --all

# Run python static code check
check-static:
    @echo "\n{{ BLUE }}{{ BOLD }}{{ ITALIC }}Static type checking: Running ty"
    uv run ty check

    @echo "\n{{ BLUE }}{{ BOLD }}{{ ITALIC }}Checking for obsolete dependencies: Running deptry"
    uv run deptry src

# Run all Project tests with coverage report if given (html or xml)
test report="":
  rep={{report}} && uv run pytest -m 'not contract_testing' ${rep:+--cov --cov-report=$rep} --cov-config=.coveragerc

# Run API tests with coverage report if given (html or xml)
api-test report="":
  rep={{report}} && uv run pytest tests -m 'not contract_testing' ${rep:+--cov --cov-report=$rep}

# Run scratch core tests with coverage report if given (html or xml)
core-test report="":
  rep={{report}} && uv run pytest packages/scratch-core -m 'not contract_testing' ${rep:+--cov --cov-report=$rep}

# test-contract REST API
test-contract:
    uv run pytest -m 'contract_testing'

# Run all endpoints health checks
smoke-test artifact="" host="0.0.0.0" port="8000":
    @echo "{{ BLUE }}{{ BOLD }}{{ ITALIC }}Testing code: Running the contract testing{{ NORMAL }}"
    @just api-bg {{ artifact }}
    @echo "{{ BLUE }}{{ BOLD }}{{ ITALIC }}Waiting for API to be ready...{{NORMAL}}"
    @timeout 10 bash -c 'until curl -fs http://{{ host }}:{{ port }}/docs > /dev/null; do sleep 1; done'
    @just test-contract
    @if [ "{{os_family()}}" = "unix" ]; then \
        kill $(cat api.pid); \
    else \
        taskkill //PID $(cat api.pid) //F 2>nul; \
    fi
    @rm -f api.pid

# Removes version control system dirty files
clean:
    @echo "{{ RED }}{{ BOLD }}{{ ITALIC }}Deleting all dirty files"
    git clean -xfd

# Build an executable for the REST API
build:
    @echo "\n{{ BLUE }}{{ BOLD }}{{ ITALIC }}Building the REST API to an executable"
    uv run pyinstaller --onefile src/main.py --clean \
    --hidden-import=numpy \
    --hidden-import=numpy.core \
    --hidden-import=numpy.core._methods \
    --hidden-import=numpy.core._dtype_ctypes

# Start API development server
api:
    @echo "{{ BLUE }}{{ BOLD }}{{ ITALIC }}Starting FastAPI development server"
    uv run fastapi dev src/main.py

# Start API server in the background
api-bg artifact="":
    cmd=(just api); [ -n "{{ artifact }}" ] && cmd=(./dist/{{ artifact }}); \
    ${cmd[@]} >/dev/null 2>&1 & echo $! > api.pid
    @echo "{{ BLUE }}{{ BOLD }}{{ ITALIC }} API started in the background"
    @cat api.pid

# list or run github job locally
ci job="":
  [ -z "{{job}}" ] && act --list || act --job {{job}} --quiet

# run coverage difference between current branch and main
cov-diff:
  [ -f coverage.xml ] || just test xml
  @echo "{{BLUE}}{{BOLD}}{{ITALIC}}Getting coverage difference against main"
  uv run diff-cover coverage.xml \
     --diff-range-notation '..' \
     --fail-under 80 \
     --format markdown:diff_coverage.md
