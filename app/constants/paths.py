from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # the repository root
APP_DIR = BASE_DIR / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DOCS_DIR = BASE_DIR / "data" / "vilnius-parking-data" / "static"
