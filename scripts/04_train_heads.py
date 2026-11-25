"""4-қадам: сақталған белгілердің үстінде классификаторларды оқыту.

Әр негізгі модель (backbone) үшін үш классификатор:
  article — мақаладағы MLP басы (Adam 1e-3, batch 32, 25 эпохаға дейін, EarlyStopping patience=10);
  linear  — Dropout + Dense(120) (біздің ұсынысымыз);
  logreg  — scikit-learn LogisticRegression (классикалық ML, C параметрі val бойынша таңдалады).

Барлық модель бірдей train бөлігінде оқиды, гиперпараметрлер тек val бойынша таңдалады,
ал test бөлігі тек соңғы бағалауға қолданылады.

Іске қосу:
    .venv\\Scripts\\python -I scripts/04_train_heads.py
    .venv\\Scripts\\python -I scripts/04_train_heads.py --backbones inception_resnet_v2 --heads linear
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import keras  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, log_loss  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.backbones import BACKBONES  # noqa: E402
from src.config import FEATURES_DIR, HISTORY_DIR, MODELS_DIR, PRED_DIR, SEED  # noqa: E402
from src.data_utils import save_json  # noqa: E402
from src.heads import HEADS  # noqa: E402

HEAD_SETTINGS = {
    # мақаладағы баптаулар
    "article": {"lr": 1e-3, "batch": 32, "epochs": 25, "patience": 10, "reduce_lr": False},
    # біздің баптаулар: ұзағырақ оқыту + оқу қарқынын азайту
    "linear": {"lr": 1e-3, "batch": 64, "epochs": 60, "patience": 8, "reduce_lr": True},
}
LOGREG_C_GRID = [0.03, 0.3]


def load_split(backbone: str, split: str, with_flip: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    d = FEATURES_DIR / backbone
    x = np.load(d / f"{split}.npy").astype(np.float32)
    y = np.load(d / f"{split}_y.npy").astype(np.int64)
    flip_path = d / f"{split}_flip.npy"
    x_flip = np.load(flip_path).astype(np.float32) if with_flip and flip_path.exists() else None
    return x, y, x_flip


def save_preds(model_id: str, split: str, probs: np.ndarray) -> None:
    np.save(PRED_DIR / f"{model_id}__{split}.npy", probs.astype(np.float32))


def train_keras_head(backbone, head, data) -> None:
    (x_tr, y_tr), (x_val, y_val), test_sets = data
    cfg = HEAD_SETTINGS[head]
    model_id = f"{backbone}__{head}"
    keras.utils.set_random_seed(SEED)
    model = HEADS[head](x_tr.shape[1])
    model.compile(optimizer=keras.optimizers.Adam(cfg["lr"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy", keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5")])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["patience"], min_delta=0.001,
                                               restore_best_weights=True)]
    if cfg["reduce_lr"]:
        callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5))
    t0 = time.perf_counter()
    hist = model.fit(x_tr, y_tr, validation_data=(x_val, y_val), epochs=cfg["epochs"], batch_size=cfg["batch"],
                     callbacks=callbacks, verbose=0, shuffle=True)
    train_sec = time.perf_counter() - t0
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    best_epoch = int(callbacks[0].best_epoch) + 1  # EarlyStopping қалпына келтірген эпоха (min_delta ескерілген)

    (MODELS_DIR / "heads").mkdir(exist_ok=True)
    model.save(MODELS_DIR / "heads" / f"{model_id}.keras")
    save_preds(model_id, "val", model.predict(x_val, verbose=0))
    for name, x in test_sets.items():
        save_preds(model_id, name, model.predict(x, verbose=0))
    save_json({"model_id": model_id, "backbone": backbone, "head": head, "settings": cfg,
               "train_seconds": round(train_sec, 1), "epochs_run": len(history["loss"]),
               "best_epoch": best_epoch, "params": int(model.count_params()), "history": history},
              HISTORY_DIR / f"{model_id}.json")
    print(f"  {model_id}: val_acc={max(history['val_accuracy']):.4f}  эпоха={len(history['loss'])} "
          f"(ең жақсы {best_epoch})  {train_sec:.0f} с", flush=True)


def train_logreg(backbone, data) -> None:
    (x_tr, y_tr), (x_val, y_val), test_sets = data
    model_id = f"{backbone}__logreg"
    best = None
    t0 = time.perf_counter()
    search = []
    for c in LOGREG_C_GRID:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=300, tol=1e-3))
        clf.fit(x_tr, y_tr)
        p_val = clf.predict_proba(x_val)
        score = log_loss(y_val, p_val, labels=np.arange(p_val.shape[1]))
        acc = accuracy_score(y_val, p_val.argmax(1))
        search.append({"C": c, "val_log_loss": round(float(score), 4), "val_accuracy": round(float(acc), 4)})
        if best is None or score < best[0]:
            best = (score, c, clf, p_val)
    train_sec = time.perf_counter() - t0
    _, c, clf, p_val = best
    (MODELS_DIR / "heads").mkdir(exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "heads" / f"{model_id}.joblib")
    save_preds(model_id, "val", p_val)
    for name, x in test_sets.items():
        save_preds(model_id, name, clf.predict_proba(x))
    save_json({"model_id": model_id, "backbone": backbone, "head": "logreg", "best_C": c, "search": search,
               "train_seconds": round(train_sec, 1)}, HISTORY_DIR / f"{model_id}.json")
    print(f"  {model_id}: C={c}  {search}  {train_sec:.0f} с", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--heads", nargs="+", default=["article", "linear", "logreg"])
    ap.add_argument("--no-flip", action="store_true", help="айналдырылған train белгілерін қоспау")
    args = ap.parse_args()

    for backbone in args.backbones:
        if not (FEATURES_DIR / backbone / "train.npy").exists():
            print(f"[{backbone}] белгілер жоқ — өткізіп жіберемін")
            continue
        x_tr, y_tr, x_tr_flip = load_split(backbone, "train", with_flip=not args.no_flip)
        if x_tr_flip is not None:  # кеңейту: айналдырылған суреттер — жаңа оқу мысалдары
            x_tr, y_tr = np.concatenate([x_tr, x_tr_flip]), np.concatenate([y_tr, y_tr])
        x_val, y_val, x_val_flip = load_split(backbone, "val", with_flip=True)
        x_te, _, x_te_flip = load_split(backbone, "test", with_flip=True)
        test_sets = {"test": x_te}  # болжамы сақталатын қосымша жиындар (TTA үшін айналдырылған нұсқалар)
        if x_te_flip is not None and x_val_flip is not None:
            test_sets["test_flip"] = x_te_flip
            test_sets["val_flip"] = x_val_flip
        print(f"[{backbone}] train={x_tr.shape} val={x_val.shape} test={x_te.shape}", flush=True)
        data = ((x_tr, y_tr), (x_val, y_val), test_sets)
        for head in args.heads:
            if head == "logreg":
                train_logreg(backbone, data)
            else:
                train_keras_head(backbone, head, data)


if __name__ == "__main__":
    main()
