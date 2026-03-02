"""App config and paths. Extend with env vars (e.g. DATA_DIR) when needed."""
from pathlib import Path

# Root of the backend package (for resolving data paths)
BACKEND_DIR = Path(__file__).resolve().parent
# Where Bible CSV and indexes will live (create when adding corpus)
DATA_DIR = BACKEND_DIR / "data"
