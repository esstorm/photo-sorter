.PHONY: help install run test clean

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help:
	@echo "Available targets:"
	@echo "  make install  - create venv and install dependencies"
	@echo "  make run      - start the app on http://localhost:8765"
	@echo "  make test     - run the test suite"
	@echo "  make clean    - remove venv and caches"

$(VENV)/bin/activate: requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt
	@touch $(VENV)/bin/activate

install: $(VENV)/bin/activate

run: install
	$(PYTHON) app.py

test: install
	$(PIP) install -q pytest httpx
	$(PYTHON) -m pytest tests/

clean:
	rm -rf $(VENV) .pytest_cache **/__pycache__
