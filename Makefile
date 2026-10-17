SRC_DIRS = app
TEST_SRC_DIRS = tests

init_venv: install_uv
	uv venv

# Installs uv only when it's missing; `python3 -m pip` works where no bare `pip` is on PATH.
install_uv:
	@command -v uv >/dev/null 2>&1 || python3 -m pip install uv==0.9.2

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

# The evaluations run on the host, where Milvus is published on localhost -
# .env.development's VECTOR_DB_URI (http://milvus:19530) only resolves inside Docker.
EVAL_MILVUS_URI ?= http://localhost:19530
API_URL ?= http://localhost:8000

eval: install
	VECTOR_DB_URI=$(EVAL_MILVUS_URI) uv run --group eval python -m evals.rag_eval

eval-chat: install
	VECTOR_DB_URI=$(EVAL_MILVUS_URI) uv run --group eval python -m evals.rag_eval --api-url $(API_URL)

eval-evidently: install
	VECTOR_DB_URI=$(EVAL_MILVUS_URI) uv run --group eval python -m evals.evidently_eval --api-url $(API_URL)

load-test: install
	uv run --group eval locust -f load_tests/locustfile.py ChatUser --host http://localhost:8000

.PHONY: install_uv init_venv install check format test admin-ui eval eval-chat eval-evidently load-test
