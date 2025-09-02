# Enhanced Earth Foundation Model Makefile

.PHONY: help install install-dev test lint format clean demo train distributed-train

# Default target
help:
	@echo "Enhanced Earth Foundation Model"
	@echo "Available commands:"
	@echo "  install          - Install the package"
	@echo "  install-dev      - Install with development dependencies"
	@echo "  test             - Run tests"
	@echo "  lint             - Run linting"
	@echo "  format           - Format code"
	@echo "  clean            - Clean build artifacts"
	@echo "  demo             - Run demo"
	@echo "  train            - Start training"
	@echo "  distributed-train - Start distributed training"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

install-flash:
	pip install -e ".[flash]"

install-distributed:
	pip install -e ".[distributed]"

install-all:
	pip install -e ".[dev,flash,distributed]"

# Development
test:
	pytest tests/ -v

lint:
	flake8 src/ scripts/ tests/
	mypy src/

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

check-format:
	black --check src/ scripts/ tests/
	isort --check-only src/ scripts/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Running
demo:
	python scripts/demo.py

train:
	python scripts/train.py --config configs/global_model.yaml

train-small:
	python scripts/train.py --config configs/small_model.yaml

distributed-train:
	torchrun --nproc_per_node=4 scripts/train.py --config configs/global_model.yaml --distributed

# Docker
docker-build:
	docker build -t enhanced-earth-foundation .

docker-run:
	docker run --gpus all -it enhanced-earth-foundation

# Documentation
docs:
	cd docs && make html

docs-serve:
	cd docs/_build/html && python -m http.server 8000

# Data preparation
prepare-data:
	python scripts/prepare_data.py --config configs/data_config.yaml

download-samples:
	python scripts/download_sample_data.py

# Model utilities
export-model:
	python scripts/export_model.py --checkpoint checkpoints/best_model.pt --format onnx

benchmark:
	python scripts/benchmark.py --config configs/benchmark.yaml

# Development workflow
dev-setup: install-dev
	@echo "Development environment setup complete"

pre-commit: format lint test
	@echo "Pre-commit checks passed"

ci: check-format lint test
	@echo "CI checks passed"