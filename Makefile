# ===========================
# Default: help section
# ===========================

info: do-show-commands

# ===========================
# Main commands
# ===========================

check-quality: do-check-quality
check-static: do-check-static
format: do-format
install: do-install
tests: do-tests
update: do-update

# Docker
start: do-docker-start
stop: do-docker-stop
logs: do-docker-logs
shell: do-docker-shell

# ===========================
# Snippets
# ===========================

docker-compose-exec = docker compose exec

# ===========================
# Recipes
# ===========================

do-show-commands:
	@echo "=== Make commands ==="
	@echo "Project commands:"
	@echo "    make install                   Make the project ready for development."
	@echo "    make upgrade                   Upgrade third party python libraries."
	@echo "Formatting:"
	@echo "    make format               	  Run formatting."
	@echo "Tests:"
	@echo "    make tests                     Run unit and feature tests."
	@echo "Docker"
	@echo "    make start                     Start all containers."
	@echo "    make stop                      Stop all containers."
	@echo "    make shell                     Open shell in the app container."
	@echo "    make logs                      Tail the container logs."


do-install:
	@echo "=== Preparing project ==="
	docker compose up -d --build
	@${docker-compose-exec} app sh -c "test -f .pre-commit-config.yaml || cp .pre-commit-config.example .pre-commit-config.yaml"
	@${docker-compose-exec} app sh -c "uv sync --all-packages --frozen"

do-update:
	@echo "=== Upgrading dependencies ==="
	@${docker-compose-exec} app sh -c "uv lock --upgrade"
	@${docker-compose-exec} app sh -c "uv sync --all-packages"
	@${docker-compose-exec} app sh -c "uv run pre-commit autoupdate"

do-tests:
	@echo "=== Running tests ==="
	@${docker-compose-exec} app sh -c "uv run pytest -m 'not contract_testing'"

# Docker
do-docker-start:
	@echo "=== Starting containers ==="
	docker compose up -d
	@echo "=== Application running at http://127.0.0.1:8000 ==="

do-docker-stop:
	@echo "=== Stopping containers ==="
	docker compose stop

do-docker-logs:
	@echo "=== Watch logs ==="
	docker compose --file docker-compose.yaml logs -f --tail 5

do-docker-shell:
	@echo "=== Start shell in app container ==="
	docker compose --file docker-compose.yaml exec app bash