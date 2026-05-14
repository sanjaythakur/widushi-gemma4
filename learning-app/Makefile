PYTHON ?= python3
VENV   ?= .venv
PIP    := $(VENV)/bin/pip
PY     := $(VENV)/bin/python

APP_IMAGE       := widushi-app
APP_DOCKERFILE  := docker/Dockerfile.app
COMPOSE_FILE    := docker/docker-compose.yml
WORKSPACE_ROOT  := ..

.PHONY: help venv install run lint test build run-docker stop-docker clean install-asound

help:
	@echo "Widushi targets:"
	@echo "  make install         - create .venv and install runtime+dev deps"
	@echo "  make run             - run the app on the host (uses .venv)"
	@echo "  make lint            - ruff check"
	@echo "  make test            - pytest smoke tests"
	@echo "  make build           - docker build the app image"
	@echo "  make run-docker      - docker compose up (headless on Mac)"
	@echo "  make stop-docker     - docker compose down"
	@echo "  make install-asound  - install Pi /etc/asound.conf (USB mic+speaker)"
	@echo "  make clean           - remove caches and the venv"

$(VENV)/bin/python:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip

venv: $(VENV)/bin/python

install: venv
	$(PIP) install -e ".[dev]"

run: venv
	$(PY) -m app.main

lint: venv
	$(VENV)/bin/ruff check app tests

test: venv
	$(VENV)/bin/pytest -q

build:
	docker build -t $(APP_IMAGE) -f $(APP_DOCKERFILE) $(WORKSPACE_ROOT)

run-docker:
	docker compose -f $(COMPOSE_FILE) up --build

stop-docker:
	docker compose -f $(COMPOSE_FILE) down

install-asound:
	@echo "Installing /etc/asound.conf (USB mic on card 0, USB speaker on card 1)."
	@echo "Verify card numbers match your Pi first:"
	@aplay -l 2>/dev/null  || echo "  (aplay not installed)"
	@arecord -l 2>/dev/null || echo "  (arecord not installed)"
	sudo install -m 0644 docker/asound.conf /etc/asound.conf
	@echo "Done. Test with: speaker-test -c2 -twav  &&  arecord -d 2 /tmp/m.wav && aplay /tmp/m.wav"

clean:
	rm -rf $(VENV) .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
