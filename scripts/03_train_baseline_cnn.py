"""3-қадам: бастапқы (baseline) модель — нөлден оқытылатын шағын CNN (transfer learning-сіз).

Мақсаты: алдын ала үйретілген модельсіз, тек 10 мыңдай суретпен 120 тұқымды ажырату
қаншалықты қиын екенін көрсету. Transfer learning-тің пайдасы осы модельмен салыстыру арқылы өлшенеді.

Архитектура (≈0.4 млн параметр): 4 × [Conv3x3 -> BatchNorm -> ReLU -> MaxPool] -> GAP -> Dropout -> Dense(120)
Кіріс: 72x72-ге кішірейтілген сурет, оқуда кездейсоқ 64x64 кесінді + көлденең айналдыру.
(GPU-сыз CPU-да оқыту уақытын қысқарту үшін шағын өлшем таңдалды: 112 px-те бір эпоха ~6 мин алды.)

Іске қосу:
    .venv\\Scripts\\python -I scripts/03_train_baseline_cnn.py
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import keras  # noqa: E402
import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

from src.config import ARTIFACTS, HISTORY_DIR, MODELS_DIR, PRED_DIR, SEED  # noqa: E402
from src.data_utils import load_splits, save_json  # noqa: E402

CACHE_SIZE, CROP = 72, 64
CACHE_DIR = ARTIFACTS / f"cnn_cache_{CACHE_SIZE}"


def load_resized(paths: list[str]) -> np.ndarray:
    def load(path):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (CACHE_SIZE, CACHE_SIZE), antialias=True)
        return tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    ds = tf.data.Dataset.from_tensor_slices(paths).map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    return np.concatenate([b.numpy() for b in ds])


def get_split(df, split: str) -> tuple[np.ndarray, np.ndarray]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{split}.npy"
    sub = df[df["split"] == split]
    if path.exists():
        x = np.load(path)
    else:
        print(f"{split}: суреттерді {CACHE_SIZE}x{CACHE_SIZE} өлшемге келтіру ...", flush=True)
        x = load_resized(sub["abspath"].tolist())
        np.save(path, x)
    return x, sub["label"].to_numpy(np.int64)


def build_cnn(num_classes: int = 120) -> keras.Model:
    inp = keras.Input(shape=(CACHE_SIZE, CACHE_SIZE, 3))
    x = layers.RandomFlip("horizontal")(inp)
    x = layers.RandomCrop(CROP, CROP)(x)  # оқуда кездейсоқ кесінді, бағалауда бүкіл сурет 64x64-ке кішірейеді
    x = layers.Rescaling(1.0 / 255)(x)
    for filters in (32, 64, 128, 256):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="baseline_cnn")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    keras.utils.set_random_seed(SEED)
    df = load_splits()
    x_tr, y_tr = get_split(df, "train")
    x_val, y_val = get_split(df, "val")
    x_te, y_te = get_split(df, "test")
    print(f"train={x_tr.shape} val={x_val.shape} test={x_te.shape}", flush=True)

    model = build_cnn()
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")])
    model.summary(print_fn=lambda s, **_: print(s))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, min_delta=0.001, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    t0 = time.perf_counter()
    hist = model.fit(x_tr.astype(np.float32), y_tr, validation_data=(x_val.astype(np.float32), y_val),
                     epochs=args.epochs, batch_size=args.batch, callbacks=callbacks, verbose=2)
    train_sec = time.perf_counter() - t0

    model.save(MODELS_DIR / "baseline_cnn.keras")
    model_id = "baseline_cnn"
    # Бағалауда (training=False) кездейсоқ кеңейту өшеді
    np.save(PRED_DIR / f"{model_id}__val.npy", model.predict(x_val.astype(np.float32), verbose=0))
    np.save(PRED_DIR / f"{model_id}__test.npy", model.predict(x_te.astype(np.float32), verbose=0))
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    save_json({"model_id": model_id, "backbone": None, "head": "cnn_from_scratch",
               "train_seconds": round(train_sec, 1), "epochs_run": len(history["loss"]),
               "best_epoch": int(np.argmin(history["val_loss"])) + 1, "params": int(model.count_params()),
               "history": history}, HISTORY_DIR / f"{model_id}.json")
    print(f"Дайын: {train_sec / 60:.1f} мин, val_acc max={max(history['val_accuracy']):.4f}")


if __name__ == "__main__":
    main()
