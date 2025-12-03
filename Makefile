.ONESHELL:
ENV_PREFIX=$(shell python -c "if __import__('pathlib').Path('.venv/bin/pip').exists(): print('.venv/bin/')")
.SHELLFLAGS := -e -c # exit if error

.PHONY: help
help:             ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@fgrep "##" Makefile | fgrep -v fgrep


.PHONY: show
show:             ## Show the current environment.
	@echo "Current environment:"
	@echo "Running using $(ENV_PREFIX)"
	@$(ENV_PREFIX)python -V
	@$(ENV_PREFIX)python -m site


.PHONY: fmt
fmt:              ## Format code using black & isort.
	$(ENV_PREFIX)isort webots-rl/
	$(ENV_PREFIX)isort main.py
	$(ENV_PREFIX)black -l 120  webots-rl/
	$(ENV_PREFIX)black -l 120  main.py

.PHONY: lint
lint:             ## Run pep8, black.
	$(ENV_PREFIX)flake8 --max-line-length 120 --ignore=E402,W503,W605 webots-rl/
	$(ENV_PREFIX)flake8 --max-line-length 120 --ignore=E402,W503,W605 main.py
	$(ENV_PREFIX)black -l 120 --check webots-rl/
	$(ENV_PREFIX)black -l 120 --check main.py

.PHONY: install
install:          ## Install the project.
	@echo "Don't forget to run 'make virtualenv' if you got errors."
	@pip install -r requirements-dev.txt
	@pip-compile requirements.in
	$(ENV_PREFIX)pip install -r requirements.txt

.PHONY: clean
clean:            ## Clean working directory.
	@rm -rf __pycache__
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf output/logs/*
	@rm -rf output/train/*
	@rm -rf webots-rl/worlds/.*.wbproj
	@rm -rf webots-rl/worlds/*_run_*

.PHONY: virtualenv
virtualenv:       ## Create virtual environment.
	@echo "Creating virtualenv ..."
	@rm -rf .venv
	@python3 -m venv .venv
	@./.venv/bin/pip install -U pip
	@echo
	@echo "!!! Please run 'source .venv/bin/activate' to enable the environment !!!"
