.PHONY: setup data train eval batch-score app test clean

setup:
	pip install -r requirements.txt

env:
	cp .env.example .env
	@echo "Fill in your tokens in .env before running make data"

data:
	python -m data.download
	python -m data.preprocess

