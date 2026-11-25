"""1b-қадам: барлау талдауы (EDA) — графиктер reports/figures/eda_*.png, статистика reports/metrics/eda_stats.json.

Толық түсіндірмесі бар нұсқасы: notebooks/01_EDA.ipynb

Іске қосу:
    .venv\\Scripts\\python -I scripts/01b_eda.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from src.eda import run_all  # noqa: E402

if __name__ == "__main__":
    for k, v in run_all().items():
        print(f"{k}: {v}")
