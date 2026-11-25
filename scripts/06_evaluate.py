"""6-қадам: барлық модельдерді бірдей test бөлігінде бағалау, салыстыру және қателерді талдау.

Маңызды ереже: соңғы модель test бойынша емес, VAL бойынша таңдалады (test тек соңғы
бағалауға арналған, әйтпесе нәтиже асыра бағаланады).

Нәтижелер:
  reports/metrics/comparison.csv|json   — барлық модельдердің метрикалары
  reports/metrics/final_model.json      — таңдалған модель және оның көрсеткіштері
  reports/metrics/per_class.csv, confused_pairs.csv, summary.json
  reports/figures/*.png                 — графиктер

Іске қосу:
    .venv\\Scripts\\python -I scripts/06_evaluate.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src import plot_style  # noqa: E402
from src.backbones import BACKBONES  # noqa: E402
from src.config import (DATA_QUALITY_JSON, FEATURES_DIR, FIG_DIR, FINAL_DIR, HISTORY_DIR,  # noqa: E402
                        METRICS_DIR, MODELS_DIR, PRED_DIR)
from src.data_utils import class_names, load_json, load_splits, save_json  # noqa: E402
from src.metrics import (classification_metrics, confused_pairs, per_class_table,  # noqa: E402
                         reliability_curve)

C = plot_style.C
HEAD_TITLES = {"article": "мақаладағы MLP басы", "linear": "сызықтық бас", "logreg": "LogisticRegression"}
# Мақаланы қайталау: сол InceptionResNetV2 + сол MLP басы + сол оқыту баптаулары.
# (Дәл 331px, /255 нұсқасы — "irv2_article" — уақыт үнемдеу үшін іске қосылмады, бірақ код дайын.)
ARTICLE_REPLICA_ID = "inception_resnet_v2__article"
ARTICLE_REPORTED_VAL_ACC = 0.9023  # GeeksforGeeks мақаласындағы валидация дәлдігі


def load_pred(model_id: str, split: str):
    p = PRED_DIR / f"{model_id}__{split}.npy"
    return np.load(p) if p.exists() else None


LATENCY = load_json(METRICS_DIR / "latency.json") if (METRICS_DIR / "latency.json").exists() else {}


def timing(backbone: str) -> dict:
    """Белгілерді шығару уақыты мен параметр саны (02-қадам) + бір сурет уақыты (05-қадам)."""
    p = FEATURES_DIR / backbone / "timing.json"
    t = load_json(p) if p.exists() else {}
    t["latency_ms_batch1"] = LATENCY.get(backbone)
    return t


def collect_models(y_tr, y_val, y_te) -> list[dict]:
    entries = []
    prior = np.bincount(y_tr, minlength=120) / len(y_tr)
    entries.append({"id": "dummy", "title": "Ең жиі класс (кездейсоқ деңгей)", "family": "baseline",
                    "backbone": None, "head": None, "val": np.tile(prior, (len(y_val), 1)),
                    "test": np.tile(prior, (len(y_te), 1)), "params": 0, "latency_ms": 0.0})

    hist = load_json(HISTORY_DIR / "baseline_cnn.json") if (HISTORY_DIR / "baseline_cnn.json").exists() else None
    if hist and load_pred("baseline_cnn", "test") is not None:
        entries.append({"id": "baseline_cnn", "title": "Нөлден оқытылған CNN (baseline)", "family": "baseline",
                        "backbone": None, "head": "cnn", "val": load_pred("baseline_cnn", "val"),
                        "test": load_pred("baseline_cnn", "test"), "params": hist["params"],
                        "train_seconds": hist["train_seconds"], "latency_ms": LATENCY.get("baseline_cnn")})

    for bb, cfg in BACKBONES.items():
        t = timing(bb)
        zs = FEATURES_DIR / bb / "test_imagenet120.npy"
        if zs.exists():
            zt, zv = np.load(zs), np.load(FEATURES_DIR / bb / "val_imagenet120.npy")
            entries.append({"id": f"zeroshot__{bb}", "title": f"{cfg['title']}: ImageNet басы, оқытусыз",
                            "family": "zero-shot", "backbone": bb, "head": "imagenet",
                            "val": zv / zv.sum(1, keepdims=True), "test": zt / zt.sum(1, keepdims=True),
                            "params": t.get("backbone_params"), "latency_ms": t.get("latency_ms_batch1"),
                            "train_seconds": 0.0})
        for head, htitle in HEAD_TITLES.items():
            mid = f"{bb}__{head}"
            p_te, p_val = load_pred(mid, "test"), load_pred(mid, "val")
            if p_te is None:
                continue
            h = load_json(HISTORY_DIR / f"{mid}.json")
            base = {"family": "transfer", "backbone": bb, "head": head,
                    "params": (t.get("backbone_params") or 0) + h.get("params", 0),
                    "latency_ms": t.get("latency_ms_batch1"), "train_seconds": h.get("train_seconds"),
                    "feature_seconds": t.get("train", {}).get("seconds")}
            entries.append({**base, "id": mid, "title": f"{cfg['title']} + {htitle}", "val": p_val, "test": p_te})
            p_te_f, p_val_f = load_pred(mid, "test_flip"), load_pred(mid, "val_flip")
            if p_te_f is not None and p_val_f is not None:
                lat = base["latency_ms"] * 2 if base["latency_ms"] else None
                entries.append({**base, "id": f"{mid}+tta", "latency_ms": lat,
                                "title": f"{cfg['title']} + {htitle} + TTA",
                                "val": (p_val + p_val_f) / 2, "test": (p_te + p_te_f) / 2})
    return entries


def fig_model_comparison(table: pd.DataFrame, final_id: str) -> None:
    t = table[~table["id"].str.endswith("+tta") | (table["id"] == final_id)].sort_values("test_accuracy")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(t) + 1.2))
    colors = []
    for mid in t["id"]:
        colors.append(C["blue"] if mid == final_id else C["orange"] if mid == ARTICLE_REPLICA_ID else C["gray"])
    ax.barh(t["title"], t["test_accuracy"] * 100, color=colors, height=0.72, edgecolor=C["surface"], linewidth=2)
    for y, (mid, acc) in enumerate(zip(t["id"], t["test_accuracy"])):
        if mid in (final_id, ARTICLE_REPLICA_ID, "baseline_cnn", "dummy") or mid.startswith("zeroshot__inception"):
            ax.text(acc * 100 + 0.8, y, f"{acc * 100:.1f}%", va="center", fontsize=9, color=C["ink"])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Test дәлдігі (accuracy), %")
    ax.grid(axis="y", visible=False)
    ax.set_title("Модельдерді салыстыру: test бөлігіндегі дәлдік")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=C["blue"], label="Таңдалған модель (val бойынша)"),
                       Patch(color=C["orange"], label="Мақаланы қайталау"),
                       Patch(color=C["gray"], label="Басқа модельдер")],
              loc="lower right", fontsize=9)
    fig.savefig(FIG_DIR / "model_comparison.png")
    plt.close(fig)


def fig_accuracy_vs_latency(table: pd.DataFrame, final_id: str) -> None:
    t = table[(table["head"] == "linear") & (~table["id"].str.endswith("+tta"))].dropna(subset=["latency_ms"])
    if t.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    final_bb = final_id.split("__")[0]
    for _, r in t.iterrows():
        is_final = r["backbone"] == final_bb
        ax.scatter(r["latency_ms"], r["test_accuracy"] * 100, s=40 + r["params"] / 4e5,
                   color=C["blue"] if is_final else C["gray"], edgecolor=C["surface"], linewidth=2, zorder=3)
        ax.annotate(f"{BACKBONES[r['backbone']]['title']}\n{r['params'] / 1e6:.1f} млн параметр",
                    (r["latency_ms"], r["test_accuracy"] * 100), textcoords="offset points", xytext=(8, -4),
                    fontsize=8.5, color=C["ink2"])
    ax.set_xscale("log")
    ax.set_xlabel("Бір суретті өңдеу уақыты, мс (CPU, log шкала)")
    ax.set_ylabel("Test дәлдігі, %")
    ax.set_title("Дәлдік пен жылдамдық арасындағы келісім (сызықтық бас)")
    fig.savefig(FIG_DIR / "accuracy_vs_latency.png")
    plt.close(fig)


def fig_training_curves(model_ids: list[tuple[str, str]]) -> None:
    hs = []
    for mid, title in model_ids:
        p = HISTORY_DIR / f"{mid}.json"
        if p.exists() and "history" in (h := load_json(p)):  # LogReg-те эпоха жоқ
            hs.append((title, h["history"]))
    if not hs:
        return
    fig, axes = plt.subplots(2, len(hs), figsize=(4.2 * len(hs), 6.4), squeeze=False)
    for j, (title, h) in enumerate(hs):
        ep = np.arange(1, len(h["loss"]) + 1)
        for i, (key, lab) in enumerate([("accuracy", "Дәлдік"), ("loss", "Қате (loss)")]):
            ax = axes[i, j]
            ax.plot(ep, h[key], color=C["blue"], label="train")
            ax.plot(ep, h[f"val_{key}"], color=C["orange"], label="val")
            best = int(np.argmin(h["val_loss"]))
            ax.scatter([best + 1], [h[f"val_{key}"][best]], color=C["orange"], s=36, zorder=3,
                       edgecolor=C["surface"], linewidth=2)
            ax.set_xlabel("Эпоха")
            ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(title, fontsize=10.5)
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Оқу қисықтары (нүкте — ең төмен val loss, сол салмақтар сақталды)", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curves.png")
    plt.close(fig)


def fig_confusion_subset(y, pred, names, pairs_df) -> None:
    involved = []
    for _, r in pairs_df.head(10).iterrows():
        for n in (r["true"], r["predicted"]):
            if n not in involved:
                involved.append(n)
    idx = [names.index(n) for n in involved]
    cm = np.zeros((len(idx), len(idx)))
    for a, ia in enumerate(idx):
        m = y == ia
        for b, ib in enumerate(idx):
            cm[a, b] = (pred[m] == ib).mean() if m.any() else 0
    fig, ax = plt.subplots(figsize=(0.55 * len(idx) + 3, 0.5 * len(idx) + 2))
    im = ax.imshow(cm, cmap=plot_style.seq_cmap(), vmin=0, vmax=1)
    ax.set_xticks(range(len(idx)), involved, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(idx)), involved, fontsize=8.5)
    ax.grid(False)
    for a in range(len(idx)):
        for b in range(len(idx)):
            if cm[a, b] >= 0.05:
                ax.text(b, a, f"{cm[a, b]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if cm[a, b] > 0.5 else C["ink"])
    ax.set_xlabel("Болжанған тұқым")
    ax.set_ylabel("Шын тұқым")
    ax.set_title("Ең жиі шатастырылатын тұқымдар (жол бойынша үлес)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.savefig(FIG_DIR / "confusion_top.png")
    plt.close(fig)


def fig_per_class(per_class: pd.DataFrame) -> None:
    worst = per_class.sort_values("f1").head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.barh(worst["breed"], worst["f1"], color=C["blue"], height=0.7, edgecolor=C["surface"], linewidth=2)
    for y_, v in enumerate(worst["f1"]):
        ax.text(v + 0.01, y_, f"{v:.2f}", va="center", fontsize=8.5, color=C["ink2"])
    ax.set_xlim(0, 1.05)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("F1 (test)")
    ax.set_title("Ең нашар танылатын 15 тұқым")
    fig.savefig(FIG_DIR / "per_class_worst.png")
    plt.close(fig)


def fig_misclassified(test_df, y, probs, names, n: int = 12) -> list[dict]:
    pred = probs.argmax(1)
    conf = probs.max(1)
    wrong = np.where(pred != y)[0]
    order = wrong[np.argsort(-conf[wrong])][:n]
    cols = 4
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.3))
    examples = []
    for ax, i in zip(axes.ravel(), order):
        r = test_df.iloc[i]
        with Image.open(r["abspath"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"Шын: {names[y[i]]}\nБолжам: {names[pred[i]]} ({conf[i]:.0%})", fontsize=8.5,
                     color=C["ink"])
        examples.append({"path": r["path"], "true": names[y[i]], "pred": names[pred[i]],
                         "confidence": round(float(conf[i]), 3), "n_dogs": int(r["n_dogs"]),
                         "bbox_frac_max": round(float(r["bbox_frac_max"]), 3)})
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Модель ең сенімді түрде қателескен test суреттері", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "misclassified_examples.png")
    plt.close(fig)
    return examples


def fig_confidence(y, probs) -> None:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 21)
    axes[0].hist([conf[correct], conf[~correct]], bins=bins, color=[C["blue"], C["orange"]],
                 label=[f"дұрыс ({correct.sum()})", f"қате ({(~correct).sum()})"],
                 edgecolor=C["surface"], linewidth=1)  # бағаналар қатар тұрады, бірін-бірі жаппайды
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Модельдің сенімділігі (ең үлкен ықтималдық)")
    axes[0].set_ylabel("Сурет саны (log)")
    axes[0].set_title("Дұрыс және қате жауаптардағы сенімділік")
    axes[0].legend()
    rc = reliability_curve(y, probs, 10)
    axes[1].plot([0, 1], [0, 1], color=C["axis"], linewidth=1)
    axes[1].plot([r[0] for r in rc], [r[1] for r in rc], color=C["blue"], marker="o", markersize=6,
                 markeredgecolor=C["surface"], markeredgewidth=2)
    axes[1].set_xlabel("Орташа сенімділік")
    axes[1].set_ylabel("Нақты дәлдік")
    axes[1].set_title("Калибрлеу (диагональ — мінсіз калибрлеу)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confidence_calibration.png")
    plt.close(fig)


def fig_error_vs_bbox(test_df, y, pred) -> dict:
    d = test_df[["bbox_frac_max", "n_dogs"]].copy()
    d["correct"] = (pred == y)
    d["size_bin"] = pd.cut(d["bbox_frac_max"], [0, 0.1, 0.25, 0.5, 0.75, 1.0],
                           labels=["<10%", "10–25%", "25–50%", "50–75%", ">75%"])
    by_size = d.groupby("size_bin", observed=True)["correct"].agg(["mean", "size"])
    d["dogs_bin"] = np.where(d["n_dogs"] >= 3, "3+", d["n_dogs"].astype(str))
    by_dogs = d.groupby("dogs_bin")["correct"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, tab, xl, title in [(axes[0], by_size, "Иттің кадрдағы ауданы (bounding box үлесі)",
                                "Ит кадрда кішкентай болса, қате көбейе ме?"),
                               (axes[1], by_dogs, "Суреттегі ит саны", "Бір суретте бірнеше ит болса")]:
        ax.bar(tab.index.astype(str), tab["mean"] * 100, color=C["blue"], width=0.6,
               edgecolor=C["surface"], linewidth=2)
        for x_, (m, s) in enumerate(zip(tab["mean"], tab["size"])):
            ax.text(x_, m * 100 + 1, f"{m * 100:.1f}%\n(n={s})", ha="center", fontsize=8.5, color=C["ink2"])
        ax.set_ylim(0, 110)
        ax.set_xlabel(xl)
        ax.set_ylabel("Test дәлдігі, %")
        ax.grid(axis="x", visible=False)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "error_vs_bbox.png")
    plt.close(fig)
    return {"by_size": {str(k): {"accuracy": round(float(v["mean"]), 4), "n": int(v["size"])}
                        for k, v in by_size.iterrows()},
            "by_dogs": {str(k): {"accuracy": round(float(v["mean"]), 4), "n": int(v["size"])}
                        for k, v in by_dogs.iterrows()}}


def main() -> None:
    plot_style.apply()
    df = load_splits()
    names = class_names(pretty=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    y_tr = df.loc[df["split"] == "train", "label"].to_numpy()
    y_val = df.loc[df["split"] == "val", "label"].to_numpy()
    y_te = test_df["label"].to_numpy()

    entries = collect_models(y_tr, y_val, y_te)
    rows = []
    for e in entries:
        mt = classification_metrics(y_te, e["test"])
        mv = classification_metrics(y_val, e["val"]) if e["val"] is not None else {}
        rows.append({"id": e["id"], "title": e["title"], "family": e["family"], "backbone": e["backbone"],
                     "head": e["head"], **{f"test_{k}": v for k, v in mt.items()},
                     "val_accuracy": mv.get("accuracy"), "val_log_loss": mv.get("log_loss"),
                     "params": e.get("params"), "latency_ms": e.get("latency_ms"),
                     "train_seconds": e.get("train_seconds"), "feature_seconds": e.get("feature_seconds")})
    table = pd.DataFrame(rows)

    # Соңғы модельді VAL бойынша таңдау (transfer отбасынан)
    cand = table[table["family"] == "transfer"].sort_values(["val_accuracy", "val_log_loss"],
                                                            ascending=[False, True])
    final = cand.iloc[0]
    final_id = final["id"]
    table["selected"] = table["id"] == final_id
    table.sort_values(["family", "test_accuracy"], ascending=[True, False]).to_csv(
        METRICS_DIR / "comparison.csv", index=False, float_format="%.4f")
    save_json(table.round(4).replace({np.nan: None}).to_dict(orient="records"), METRICS_DIR / "comparison.json")
    print(table[["id", "test_accuracy", "test_top5_accuracy", "test_macro_f1", "test_log_loss",
                 "val_accuracy"]].sort_values("test_accuracy", ascending=False).to_string(index=False))
    print(f"\nТаңдалған модель (val бойынша): {final_id}")

    probs = next(e["test"] for e in entries if e["id"] == final_id)
    pred = probs.argmax(1)
    per_class = per_class_table(y_te, pred, names)
    per_class.to_csv(METRICS_DIR / "per_class.csv", index=False, float_format="%.4f")
    pairs_df = confused_pairs(y_te, pred, names, top=20)
    pairs_df.to_csv(METRICS_DIR / "confused_pairs.csv", index=False)

    # Графиктер
    fig_model_comparison(table, final_id)
    fig_accuracy_vs_latency(table, final_id)
    best_bb, best_head = final["backbone"], final["head"]
    curve_head = best_head if best_head != "logreg" else "linear"  # LogReg эпохамен оқымайды
    curves = [(ARTICLE_REPLICA_ID, "Мақаланы қайталау (InceptionResNetV2 + MLP басы)"),
              (f"{best_bb}__{curve_head}", f"{BACKBONES[best_bb]['title']} + {HEAD_TITLES[curve_head]}"),
              ("baseline_cnn", "Нөлден оқытылған CNN")]
    fig_training_curves(curves)
    fig_confusion_subset(y_te, pred, names, pairs_df)
    fig_per_class(per_class)
    examples = fig_misclassified(test_df, y_te, probs, names)
    fig_confidence(y_te, probs)
    bbox_stats = fig_error_vs_bbox(test_df, y_te, pred)

    # Сайт үшін: "суретте ит бар ма" шегі (ImageNet ит кластарының ықтималдық қосындысы)
    dog_path = FEATURES_DIR / best_bb / "test_dogness.npy"
    dogness = np.load(dog_path) if dog_path.exists() else None
    conf = probs.max(1)
    correct = pred == y_te
    thresholds = [{"threshold": t, "share_above": round(float((conf >= t).mean()), 4),
                   "accuracy_above": round(float(correct[conf >= t].mean()), 4),
                   "accuracy_below": round(float(correct[conf < t].mean()), 4) if (conf < t).any() else None}
                  for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)]
    low_conf = 0.5  # бұдан төмен сенімділікте модель жартысынан көбінде қателеседі (калибрлеу графигі)
    final_info = {
        "model_id": final_id, "title": final["title"], "backbone": best_bb, "head": best_head,
        "tta": final_id.endswith("+tta"), "input_size": BACKBONES[best_bb]["size"],
        "test_metrics": {k.removeprefix("test_"): round(float(final[k]), 4) for k in final.index
                         if k.startswith("test_")},
        "val_accuracy": round(float(final["val_accuracy"]), 4),
        "params": int(final["params"]),
        "latency_ms": None if pd.isna(final["latency_ms"]) else float(final["latency_ms"]),
        "low_confidence_threshold": low_conf,
        "accuracy_when_conf_ge_threshold": round(float(correct[conf >= low_conf].mean()), 4),
        "share_conf_ge_threshold": round(float((conf >= low_conf).mean()), 4),
        "accuracy_when_conf_lt_threshold": round(float(correct[conf < low_conf].mean()), 4),
        "confidence_thresholds": thresholds,
        "dogness_threshold": 0.50,
        "dogness_test_percentiles": ({p: round(float(np.percentile(dogness, p)), 4) for p in (1, 5, 10, 50)}
                                     if dogness is not None else None),
    }
    save_json(final_info, METRICS_DIR / "final_model.json")

    # Сайт үшін соңғы модельді models/final/ қалтасына экспорттау
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    for old in FINAL_DIR.glob("head.*"):  # алдыңғы таңдаудың файлдары қалып қоймауы керек
        old.unlink()
    head_id = f"{best_bb}__{best_head}"
    ext = ".joblib" if best_head == "logreg" else ".keras"
    shutil.copy2(MODELS_DIR / "heads" / f"{head_id}{ext}", FINAL_DIR / f"head{ext}")
    # Экспорт тексерісі: көшірілген бас сақталған val болжамдарын дәл қайталауы керек
    x_val = np.load(FEATURES_DIR / best_bb / "val.npy").astype(np.float32)
    if ext == ".joblib":
        p_check = joblib.load(FINAL_DIR / "head.joblib").predict_proba(x_val)
    else:
        import keras
        p_check = keras.models.load_model(FINAL_DIR / "head.keras").predict(x_val, verbose=0)
    diff = float(np.abs(p_check - load_pred(head_id, "val")).max())
    assert diff < 1e-3, f"Экспортталған бас сәйкес емес (айырма {diff})"
    print(f"Экспорт тексерілді: models/final/head{ext}, val болжамдарымен айырма {diff:.1e}")
    labels = df.drop_duplicates("label").sort_values("label")
    save_json({"names": names, "breeds": labels["breed"].tolist(), "synsets": labels["synset"].tolist()},
              FINAL_DIR / "labels.json")
    save_json(final_info, FINAL_DIR / "model_info.json")

    def row(mid):
        r = table[table["id"] == mid]
        return None if r.empty else {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                                     for k, v in r.iloc[0].items() if k in (
                                         "id", "title", "test_accuracy", "test_top5_accuracy", "test_macro_f1",
                                         "test_log_loss", "test_ece", "val_accuracy", "latency_ms", "params")}

    zs = table[table["family"] == "zero-shot"].sort_values("test_accuracy", ascending=False)
    summary = {
        "data_quality": load_json(DATA_QUALITY_JSON) if DATA_QUALITY_JSON.exists() else None,
        "final_model": final_info,
        "article_reported_val_accuracy": ARTICLE_REPORTED_VAL_ACC,
        "article_replica": row(ARTICLE_REPLICA_ID),
        "baseline_cnn": row("baseline_cnn"),
        "dummy": row("dummy"),
        "best_zero_shot": row(zs.iloc[0]["id"]) if not zs.empty else None,
        "worst_classes": per_class.sort_values("f1").head(10).round(3).to_dict(orient="records"),
        "best_classes": per_class.sort_values("f1", ascending=False).head(10).round(3).to_dict(orient="records"),
        "top_confused_pairs": pairs_df.head(10).to_dict(orient="records"),
        "most_confident_errors": examples,
        "error_vs_bbox": bbox_stats,
    }
    save_json(summary, METRICS_DIR / "summary.json")
    print("Сақталды:", METRICS_DIR, FIG_DIR)


if __name__ == "__main__":
    main()
