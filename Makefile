
help:
	@echo "install - install the package"
	@echo "check - run linters and formatters"
	@echo "test - run tests"
	@echo "benchmark - run benchmarks"
	@echo "coverage - run coverage"
	@echo "build - build the package"
	@echo "dist - build and upload the package"
	@echo "docs - build the docyard documentation site to dist-docs"
	@echo "clean - clean the project"

clean:
	rm -rf build dist *.egg-info .pytest_cache .tox .coverage .hypothesis .mypy_cache .mypy .ruff .ruff_cache .pytest_cache .pytest .benchmarks .benchmarks_cache wheelhouse
	find . '(' \
		    -name ".mypy_cache" -o -name "__pycache__" -o -name "*.egg-info" \
		 -o -name ".pytest_cache" -o -name ".tox" -o -name ".coverage" \
		 -o -name ".hypothesis" -o -name ".mypy" -o -name ".ruff" \
		 -o -name ".ruff_cache" -o -name ".pytest" -o -name ".benchmarks" \
		 -o -name ".benchmarks_cache" -o -name "wheelhouse" \
		')' -type d -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	
install:
	uv pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation ./

check:
	uv run ruff check --fix .

test: check
	uv run pytest -s -v -n auto --dist=loadfile --junitxml=tests.xml --no-cov --benchmark-disable
	
benchmark:
	uv run pytest -s -v -n 0 --no-cov benchmarks

coverage:
	uv run pytest --cov=sources --cov-report=html --cov-report=xml --benchmark-disable

build:
	python -m build --wheel

dist:
	uv run twine check dist/*
	uv run twine upload dist/*

docs:
	nix run .#docs-build

.PHONY: help install check test benchmark coverage dist build clean compile docs