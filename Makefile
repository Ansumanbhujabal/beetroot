.PHONY: install lint types test cov eval check clean

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

types:
	uv run mypy src

test:
	uv run pytest -q

cov:
	uv run pytest --cov --cov-report=term-missing --cov-report=html

eval:
	BEATROOT_OFFLINE=1 uv run beatroot eval system
	BEATROOT_OFFLINE=1 uv run beatroot eval components

check: lint types cov eval

clean:
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov beatroot.db
