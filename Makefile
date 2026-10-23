.PHONY: install test lint simulate train serve package docker clean
PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

lint:
	ruff check src tests sagemaker scripts

simulate:
	income-imputation simulate --n 3000 --n-sim 10 --out artifacts/metrics.json

train:
	income-imputation train --method norm --out artifacts/model/model.joblib

serve: train
	MODEL_DIR=artifacts/model uvicorn income_imputation.serve:app --port 8080

package:
	bash scripts/package_source.sh

docker:
	docker build -t income-imputation:latest .

clean:
	rm -rf build artifacts .pytest_cache .ruff_cache
