.PHONY: run test fmt lint

run:
	POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON=false PYTHONPATH=src poetry run uvicorn el_agent.app:app --host 0.0.0.0 --port 8000

test:
	POETRY_VIRTUALENVS_PREFER_ACTIVE_PYTHON=false PYTHONPATH=src poetry run pytest -q

fmt:
	poetry run black src tests
	poetry run ruff --fix src tests

lint:
	poetry run ruff src tests
	poetry run black --check src tests


