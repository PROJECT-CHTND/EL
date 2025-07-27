PYTHON ?= python3.11
VENV ?= .venv

.PHONY: install test typecheck ci clean

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && \
		pip install -r requirements.txt -r dev-requirements.txt

test:
	. $(VENV)/bin/activate && pytest -q

typecheck:
	. $(VENV)/bin/activate && mypy agent

ci: test typecheck

clean:
	rm -rf $(VENV) 