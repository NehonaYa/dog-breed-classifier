"""Жобаның барлық жолдары мен тұрақтылары бір жерде."""
import sys
from pathlib import Path

# Windows консолі (cp1251/cp866) қазақ әріптерін (ә, ғ, қ, ң, ө, ұ, ү, һ, і) шығара алмауы мүмкін
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parents[1]

# Деректер (Stanford Dogs, ресми көз: vision.stanford.edu)
DATA_RAW = ROOT / "data" / "raw" / "stanford_dogs"
IMAGES_DIR = DATA_RAW / "Images"
ANNOT_DIR = DATA_RAW / "Annotation"
FIXED_IMAGES_DIR = ROOT / "data" / "processed" / "fixed_images"  # түзетілген (RGB JPEG) файлдар
SPLITS_CSV = ROOT / "data" / "processed" / "splits.csv"
DATA_QUALITY_JSON = ROOT / "data" / "processed" / "data_quality.json"

# Аралық нәтижелер (git-ке кірмейді)
ARTIFACTS = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS / "features"
PRED_DIR = ARTIFACTS / "predictions"
HISTORY_DIR = ARTIFACTS / "histories"

# Модельдер және есептер
MODELS_DIR = ROOT / "models"
FINAL_DIR = MODELS_DIR / "final"   # сайт қолданатын соңғы модель (бас + белгілер тізімі + ақпарат)
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

SEED = 42
VAL_PER_CLASS = 15          # ресми train бөлігінен әр класс бойынша валидацияға алынатын сурет саны
NUM_CLASSES = 120

for _d in (FEATURES_DIR, PRED_DIR, HISTORY_DIR, MODELS_DIR, FIG_DIR, METRICS_DIR, SPLITS_CSV.parent):
    _d.mkdir(parents=True, exist_ok=True)
