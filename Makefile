.PHONY: help install install-dev lint format test eval-rag eval-deep gate cost drift ui clean

help:
	@echo "make install      - install full pipeline deps (deepeval, ragas, ...)"
	@echo "make install-dev  - install lint/test tooling only (offline)"
	@echo "make lint         - ruff check"
	@echo "make format       - ruff format + autofix"
	@echo "make test         - offline unit suite (no API keys)"
	@echo "make eval-rag     - RAGAS eval over the golden dataset (needs GROQ_API_KEY)"
	@echo "make eval-deep    - DeepEval batch eval (needs GROQ_API_KEY)"
	@echo "make gate         - run the CI quality gate over a results file"
	@echo "make cost         - LiteLLM cost tracking over the golden dataset"
	@echo "make drift        - drift detection vs 7-day baseline"
	@echo "make ui           - run the Build-vs-Measure demo UI on :8502 (beside Project 1)"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest

eval-rag:
	python evals/runners/run_ragas.py --output results/ragas_nightly.json

eval-deep:
	python evals/runners/run_deepeval.py --output results/deepeval_nightly.json

gate:
	python -m evals.gate --results results/ragas_nightly.json

cost:
	python evals/runners/run_cost.py --output results/cost_nightly.json

drift:
	python scripts/drift_detector.py --current results/ragas_nightly.json

ui:
	streamlit run app/streamlit_app.py --server.port 8502

clean:
	rm -rf .pytest_cache .ruff_cache .deepeval __pycache__ */__pycache__ */*/__pycache__
