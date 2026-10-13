SRC_DIRS = app
TEST_SRC_DIRS = tests

init_venv: install_uv
	uv venv

install_uv:
	pip install uv==0.9.2

install: install_uv
	uv sync

check: install
	uv run ruff check ${SRC_DIRS} ${TEST_SRC_DIRS}

format: install
	uv run ruff check --select I --fix ${SRC_DIRS} ${TEST_SRC_DIRS}
	uv run ruff format ${SRC_DIRS} ${TEST_SRC_DIRS}

test: install
	uv run pytest -s

admin-ui: install
	uv run streamlit run ui/admin_app.py

eval: install
	uv run --group eval python -m evals.rag_eval

load-test: install
	uv run --group eval locust -f load_tests/locustfile.py ChatUser --host http://localhost:8000

.PHONY: install_uv init_venv install check format test admin-ui eval load-test
