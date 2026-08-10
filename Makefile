.PHONY: setup data train eval batch-score app test clean

setup:
	pip install -r requirements.txt

env:
	cp .env.example .env
	@echo "Fill in your tokens in .env before running make data"

data:
	python -m data.download
	python -m data.preprocess

train:
	python -m model.train

eval:
	python -m eval.evaluate

batch-score:
	python -m eval.batch_score

app:
	python -m app.app

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf data/raw data/processed model/checkpoints eval/results
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
