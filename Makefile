.PHONY: format check-format test lint check-all typing all

# Format code
format:
	uv run isort tests/ src/
	uv run black tests/ src/

# Check formatting (without making changes)  
check-format:
	uv run isort --check-only --diff tests/ src/
	uv run black --check --diff tests/ src/

lint:
	uv run flake8 tests/ src/

test:
	uv run pytest

# Run all pre-commit checks manually
check-all:
	uv run pre-commit run --all-files

all: format test lint
	@echo "All checks passed!"