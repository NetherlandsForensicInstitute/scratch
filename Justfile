# Helper function to style echo messages
log message style="magenta":
    #!/bin/bash
    case "{{ style }}" in
        blue) echo -e "{{ BLUE }}{{ BOLD }}{{ ITALIC }}{{ message }}{{ NORMAL }}" ;;
        magenta) echo -e "{{ MAGENTA }}{{ BOLD }}{{ ITALIC }}{{ message }}{{ NORMAL }}" ;;
        yellow) echo -e "{{ YELLOW }}{{ BOLD }}{{ ITALIC }}{{ message }}{{ NORMAL }}" ;;
        red) echo -e "{{ RED }}{{ BOLD }}{{ ITALIC }}{{ message }}{{ NORMAL }}" ;;
        *) just log "Yellow: change" "yellow" \
            && just log "Blue: Additive" "blue" \
            && just log "Red: subtractive" "red" \
            && just log "Magenta: command"
    esac

# List all tasks (default)
help: (log "" "")
    @echo "Just docs: https://just.systems/man/en/introduction.html"
    @echo "cheatsheet: https://cheatography.com/linux-china/cheat-sheets/justfile/"
    @just --list

# Install the virtual environment and per-commit hooks
install: (log "Creating virtual environment using uv")
    uv sync --all-packages --frozen

# update the virtual environment and per-commit hooks
update: (log "Updating virtual environment using uv" "yellow")
    uv lock --upgrade
    uv sync --all-packages
    uv run pre-commit autoupdate

# Run code formatter
format: (log "Format python files" "yellow")
    uv run ruff format

# Run project quality (invoke pre-commit on all files)
check-quality: (log "Checking code quality: Running pre-commit")
    uv run pre-commit run --all

# Run python static code check
check-static:
    @just log "\nStatic type checking: Running ty"
    uv run ty check

    @just log "\nChecking for obsolete dependencies: Running deptry"
    uv run deptry src

# Run all Project tests with coverage report if given
test $report="":
    uv run pytest -m 'not contract_testing' ${report:+--cov --cov-report $report}

# test-contract REST API
test-contract: (log "Running contract tests...")
    uv run pytest -m 'contract_testing'

# Run all endpoints health checks
smoke-test artifact="" host="0.0.0.0" port="8000": (api-bg artifact) (log "Waiting for API to be ready...")
    @timeout 10 bash -c 'until curl -fs http://{{ host }}:{{ port }}/docs > /dev/null; do sleep 1; done'
    @just test-contract
    @if [ "{{ os_family() }}" = "unix" ]; then \
        kill $(cat api.pid); \
    else \
        taskkill //PID $(cat api.pid) //F 2>nul; \
    fi
    @rm -f api.pid

# Removes version control system dirty files
clean: (log "Delete all dirty files" "red")
    git clean -xfd

# Build an executable for the REST API
build: (log "\nBuilding the REST API to an executable" "blue")
    uv run pyinstaller --onefile src/main.py --clean \
    --hidden-import=numpy \
    --hidden-import=numpy.core \
    --hidden-import=numpy.core._methods \
    --hidden-import=numpy.core._dtype_ctypes

# Start API development server
api: (log "Starting FastAPI development server")
    uv run fastapi dev src/main.py

# Start API server in the background
api-bg artifact="":
    @cmd=(just api); [ -n "{{ artifact }}" ] && cmd=(./dist/{{ artifact }}); \
    ${cmd[@]} >/dev/null 2>&1 & echo $! > api.pid
    @just log "API started in the background"
    @cat api.pid

# list or run github job locally
ci job="":
    [ -z "{{ job }}" ] && act --list || act --job {{ job }} --quiet

# run coverage difference between current branch and main
cov-diff:
    [ -f coverage.xml ] || just test xml
    @just log "Getting coverage difference against main"
    uv run diff-cover coverage.xml \
       --diff-range-notation '..' \
       --fail-under 80 \
       --format markdown:diff_coverage.md
