import logging
import os
from pathlib import Path
from typing import Dict, Any

# Setup elite professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
PICKLE_DIR = ROOT_DIR / "pickle"
OUTPUT_DIR = ROOT_DIR / "output"
DB_PATH = ROOT_DIR / "forecast_iq.db"

# Ensure essential directories exist
for directory in [DATA_DIR, PICKLE_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
