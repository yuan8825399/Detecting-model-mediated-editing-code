PY ?= python

.PHONY: help install test lint audit pipeline figures clean

help:
	@echo "make install    install core dependencies"
	@echo "make test       run the fast test suite"
	@echo "make lint       ruff check"
	@echo "make audit      source-isolation audit (~1 min)"
	@echo "make pipeline   full analysis pipeline (~10 min, excludes optional stages)"
	@echo "make figures    rebuild Figures 1-9 from committed results"
	@echo "make clean      remove generated intermediates"

install:
	$(PY) -m pip install -r requirements.txt
	$(PY) -m pip install -e .

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check provenance analysis scripts tests

audit:
	$(PY) analysis/40_isolation_audit.py

pipeline:
	$(PY) scripts/run_pipeline.py --stage core

figures:
	$(PY) scripts/run_pipeline.py --only typology,figures,figure-palcp,figure-coverage

clean:
	rm -rf processed .pytest_cache
