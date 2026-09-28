# parking-reservation-chatbot

## Getting Started

Clone the repository using SSH:

```bash
# Default (no SSH host alias configured)
git clone git@github.com:FastSnakeEyes/parking-reservation-chatbot.git

# If you have a github-personal host alias set up in your SSH config
git clone git@github-personal:FastSnakeEyes/parking-reservation-chatbot.git
```

## Linting & Formatting

Run Ruff from the project root:

```bash
uv run ruff check .
```

Check formatting without changing files:

```bash
uv run ruff format --check .
```

Automatically fix safe linting problems:

```bash
uv run ruff check . --fix
```

Format the code:

```bash
uv run ruff format .
```

Recommended full sequence:

```bash
uv run ruff check . --fix
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
```