PYTHON ?= python3.11
VENV ?= .venv

.PHONY: install test typecheck ci clean setup-indices prompt-eval report

install:
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -U pip && \
		pip install -r requirements.txt -r dev-requirements.txt

test:
	. $(VENV)/bin/activate && python -m pytest -q

typecheck:
	. $(VENV)/bin/activate && mypy agent

ci: test typecheck

clean:
	rm -rf $(VENV) 

setup-indices:
	@if [ ! -x $(VENV)/bin/python ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	if ! $(VENV)/bin/python -c 'import sys; import pathlib; p=pathlib.Path("$(VENV)/lib"); print(p); exit(0 if sys.version_info[:2]==(3,11) else 1)'; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	$(VENV)/bin/python -m pip install -U pip; \
	$(VENV)/bin/python -m pip install qdrant-client==1.9.0 elasticsearch==8.13.0; \
	ES_URL=$${ES_URL:-http://localhost:9200} QDRANT_URL=$${QDRANT_URL:-http://localhost:6333} $(VENV)/bin/python scripts/setup_indices.py

prompt-eval:
	@if [ ! -x $(VENV)/bin/python ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	if ! $(VENV)/bin/python -c 'import sys; import pathlib; p=pathlib.Path("$(VENV)/lib"); print(p); exit(0 if sys.version_info[:2]==(3,11) else 1)'; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	$(VENV)/bin/python -m pip install -U pip; \
	$(VENV)/bin/python -m pip install -r requirements.txt; \
	P=$${P:?Specify P=<prompt>}; C=$${C:?Specify C=<case path>}; \
	$(VENV)/bin/python scripts/prompt_eval.py --prompt $$P --case $$C

report:
	@if [ ! -x $(VENV)/bin/python ]; then \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	if ! $(VENV)/bin/python -c 'import sys; import pathlib; p=pathlib.Path("$(VENV)/lib"); print(p); exit(0 if sys.version_info[:2]==(3,11) else 1)'; then \
		rm -rf $(VENV); \
		$(PYTHON) -m venv $(VENV); \
	fi; \
	$(VENV)/bin/python scripts/score_aggregator.py