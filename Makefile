.PHONY: install dev lint test run clean help

# Default Python version
PYTHON ?= python3.11

help:
	@echo "Eager Learner (EL) - Available commands:"
	@echo ""
	@echo "  make install    - Install dependencies with uv"
	@echo "  make dev        - Install with dev dependencies"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make test       - Run tests"
	@echo "  make run        - Run the API server"
	@echo "  make clean      - Clean up cache files"
	@echo "  make docker-up  - Start services with Docker Compose"
	@echo "  make docker-down - Stop Docker services"

install:
	uv sync

dev:
	uv sync --all-extras

lint:
	uv run ruff check packages/
	uv run ruff format --check packages/
	uv run mypy packages/core/src/el_core packages/api/src/el_api --ignore-missing-imports

format:
	uv run ruff check --fix packages/
	uv run ruff format packages/

test:
	uv run pytest packages/ -v

run:
	uv run el-api

run-dev:
	ENV=development uv run el-api

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f el-api
