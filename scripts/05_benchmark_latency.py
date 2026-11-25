"""5-қадам: бір суретті өңдеу уақытын (latency) өлшеу — TensorFlow, CPU, batch = 1.

Сайттағы жағдайға жақын: бір сурет -> негізгі модель -> классификатор.
Әділ салыстыру үшін компьютер бос тұрғанда (басқа ауыр процесс жоқ кезде) іске қосу керек.

Іске қосу:
    .venv\\Scripts\\python -I scripts/05_benchmark_latency.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402
import numpy as np  # noqa: E402

from src.backbones import BACKBONES, build_extractor  # noqa: E402
from src.config import FEATURES_DIR, METRICS_DIR, MODELS_DIR  # noqa: E402
from src.data_utils import save_json  # noqa: E402

RUNS = 20


def median_ms(fn, x) -> float:
    fn(x)  # алғашқы шақыру (графты құру) есепке кірмейді
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        fn(x)
        times.append((time.perf_counter() - t0) * 1000)
    return round(float(np.median(times)), 1)


def main() -> None:
    rng = np.random.default_rng(0)
    result = {}
    for name, cfg in BACKBONES.items():
        if not (FEATURES_DIR / name / "test.npy").exists():  # тек эксперименттерде қолданылған модельдер
            continue
        model = build_extractor(name)
        x = cfg["preprocess"](rng.uniform(0, 255, (1, cfg["size"], cfg["size"], 3)).astype(np.float32))
        result[name] = median_ms(model.predict_on_batch, x)
        print(f"{name:22s} {result[name]:7.1f} мс", flush=True)
        keras.backend.clear_session()
    cnn_path = MODELS_DIR / "baseline_cnn.keras"
    if cnn_path.exists():
        cnn = keras.models.load_model(cnn_path)
        size = cnn.input_shape[1]
        x = rng.uniform(0, 255, (1, size, size, 3)).astype(np.float32)
        result["baseline_cnn"] = median_ms(cnn.predict_on_batch, x)
        print(f"{'baseline_cnn':22s} {result['baseline_cnn']:7.1f} мс")
    save_json(result, METRICS_DIR / "latency.json")


if __name__ == "__main__":
    main()
