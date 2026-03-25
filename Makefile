# KIRA — Makefile
# Run any command with: make <target>
# Example: make generate-data

.PHONY: help setup generate-data ingest eval-chunking run-pipeline test clean

help:
	@echo ""
	@echo "KIRA commands"
	@echo "─────────────────────────────────────"
	@echo "  make setup           Install dependencies"
	@echo "  make generate-data   Generate all synthetic data"
	@echo "  make ingest          Chunk + embed + load into vector store"
	@echo "  make eval-chunking   Benchmark chunking strategies with Ragas"
	@echo "  make run-pipeline    Run the full agent pipeline"
	@echo "  make test            Run all tests"
	@echo "  make clean           Remove generated data and caches"
	@echo ""

setup:
	pip install -r requirements.txt

generate-data:
	python scripts/generate_data.py

ingest:
	python scripts/ingest_docs.py

eval-chunking:
	python ingestion/eval_chunking.py

run-pipeline:
	python scripts/run_pipeline.py

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf data/chunks/* data/embeddings/* data/chroma/*
	@echo "Cleaned generated artifacts. Raw data preserved."