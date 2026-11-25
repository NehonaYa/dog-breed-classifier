"""Барлау талдауы (EDA): графиктер мен статистика. Ноутбук (01_EDA.ipynb) мен скрипт ортақ қолданады."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src import plot_style
from src.config import DATA_QUALITY_JSON, FIG_DIR, METRICS_DIR, ROOT, SEED
from src.data_utils import load_json, load_splits, pretty_breed, save_json

C = plot_style.C


def load_all() -> pd.DataFrame:
    """Барлық сурет (алынып тасталғандарын қоса) — деректер сапасын талдау үшін."""
    return load_splits(drop_excluded=False)


def basic_stats(df: pd.DataFrame) -> dict:
    kept = df[~df["excluded"]]
    per_class = kept.groupby("breed").size()
    test_pc = kept[kept["split"] == "test"].groupby("breed").size()
    train_pc = kept[kept["split"] == "train"].groupby("breed").size()
    stats = {
        "n_images_total": int(len(df)),
        "n_images_used": int(len(kept)),
        "n_classes": int(df["label"].nunique()),
        "missing_values": int(df[["path", "breed", "label", "width", "height"]].isna().sum().sum()),
        "formats": df["fmt"].value_counts().to_dict(),
        "modes": df["mode"].value_counts().to_dict(),
        "median_width": int(df["width"].median()), "median_height": int(df["height"].median()),
        "min_width": int(df["width"].min()), "max_width": int(df["width"].max()),
        "min_height": int(df["height"].min()), "max_height": int(df["height"].max()),
        "median_aspect": round(float((df["width"] / df["height"]).median()), 3),
        "median_file_kb": round(float(df["file_kb"].median()), 1),
        "per_class_min": {"breed": pretty_breed(per_class.idxmin()), "n": int(per_class.min())},
        "per_class_max": {"breed": pretty_breed(per_class.idxmax()), "n": int(per_class.max())},
        "train_per_class_min": int(train_pc.min()), "train_per_class_max": int(train_pc.max()),
        "test_per_class_min": int(test_pc.min()), "test_per_class_max": int(test_pc.max()),
        "share_multi_dog": round(float((df["n_dogs"] > 1).mean()), 4),
        "median_bbox_frac": round(float(df["bbox_frac_max"].median()), 3),
        "share_dog_lt_25pct": round(float((df["bbox_frac_max"] < 0.25).mean()), 4),
        "median_brightness": round(float(df["brightness"].median()), 1),
    }
    return stats


def fig_class_distribution(df: pd.DataFrame, show: bool = False) -> None:
    kept = df[~df["excluded"]]
    total = kept.groupby("breed").size().sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    ax.bar(np.arange(len(total)), total.values, color=C["blue"], width=0.8, edgecolor=C["surface"], linewidth=0.6)
    ax.set_xticks([])
    ax.set_xlim(-1, len(total))
    ax.set_xlabel("120 тұқым (сурет саны бойынша сұрыпталған)")
    ax.set_ylabel("Сурет саны")
    ax.set_title("Әр тұқымдағы сурет саны (барлық бөлік)")
    ax.grid(axis="x", visible=False)
    for idx in (0, len(total) - 1):
        ax.annotate(f"{pretty_breed(total.index[idx])}: {total.iloc[idx]}", (idx, total.iloc[idx]),
                    xytext=(10 if idx == 0 else -10, 8), textcoords="offset points",
                    ha="left" if idx == 0 else "right", fontsize=9, color=C["ink2"])
    ax = axes[1]
    bins = np.arange(40, 165, 5)
    counts = [kept[kept["split"] == s].groupby("breed").size() for s in ("train", "test")]
    ax.hist(counts, bins=bins, color=[C["blue"], C["orange"]], label=["train", "test"],
            edgecolor=C["surface"], linewidth=1)  # бағаналар қатар тұрады, бірін-бірі жаппайды
    ax.set_xlabel("Бір тұқымдағы сурет саны")
    ax.set_ylabel("Тұқым саны")
    ax.set_title("train теңестірілген, test — жоқ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_class_distribution.png")
    plt.show() if show else plt.close(fig)


def fig_image_sizes(df: pd.DataFrame, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hb = axes[0].hexbin(df["width"], df["height"], gridsize=40, cmap=plot_style.seq_cmap(), mincnt=1,
                        bins="log", extent=(0, 1000, 0, 1000))
    axes[0].set_xlabel("Ені, пиксель")
    axes[0].set_ylabel("Биіктігі, пиксель")
    axes[0].set_title("Сурет өлшемдері (1000 px дейін)")
    fig.colorbar(hb, ax=axes[0], label="сурет саны (log)")
    aspect = (df["width"] / df["height"]).clip(0.3, 3)
    axes[1].hist(aspect, bins=50, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
    axes[1].axvline(1.0, color=C["ink2"], linewidth=1)
    axes[1].set_xlabel("Пропорция (ені / биіктігі)")
    axes[1].set_ylabel("Сурет саны")
    axes[1].set_title("Көбі көлденең (ені > биіктігі)")
    axes[2].hist(df["file_kb"].clip(upper=300), bins=50, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
    axes[2].set_xlabel("Файл көлемі, КБ (300+ біріктірілген)")
    axes[2].set_ylabel("Сурет саны")
    axes[2].set_title("Файл көлемі")
    for ax in axes[1:]:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_image_sizes.png")
    plt.show() if show else plt.close(fig)


def fig_bbox(df: pd.DataFrame, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(df["bbox_frac_max"] * 100, bins=40, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
    med = df["bbox_frac_max"].median() * 100
    axes[0].axvline(med, color=C["ink2"], linewidth=1)
    axes[0].text(0.02, 0.95, f"медиана (тік сызық): {med:.0f}%", transform=axes[0].transAxes,
                 fontsize=9, color=C["ink2"], va="top")
    axes[0].set_xlabel("Ең үлкен иттің кадрдағы ауданы, %")
    axes[0].set_ylabel("Сурет саны")
    axes[0].set_title("Ит кадрдың қанша бөлігін алады?")
    nd = df["n_dogs"].clip(upper=4).value_counts().sort_index()
    labels = [str(i) if i < 4 else "4+" for i in nd.index]
    axes[1].bar(labels, nd.values, color=C["blue"], width=0.6, edgecolor=C["surface"], linewidth=2)
    for x_, v in enumerate(nd.values):
        axes[1].text(x_, v, f"{v} ({v / len(df):.1%})", ha="center", va="bottom", fontsize=9, color=C["ink2"])
    axes[1].set_xlabel("Суреттегі ит саны")
    axes[1].set_ylabel("Сурет саны")
    axes[1].set_title("Бір суретте бірнеше ит болуы мүмкін")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_bbox.png")
    plt.show() if show else plt.close(fig)


def fig_brightness(df: pd.DataFrame, show: bool = False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].hist(df["brightness"], bins=50, color=C["blue"], edgecolor=C["surface"], linewidth=0.6)
    axes[0].set_xlabel("Орташа жарықтық (0 — қара, 255 — ақ)")
    axes[0].set_ylabel("Сурет саны")
    axes[0].set_title("Жарықтықтың таралуы")
    bins = np.linspace(0, 255, 52)
    for col, color, name in [("mean_r", C["red"], "R (қызыл)"), ("mean_g", C["green"], "G (жасыл)"),
                             ("mean_b", C["blue"], "B (көк)")]:
        axes[1].hist(df[col], bins=bins, histtype="step", color=color, linewidth=2, label=name)
    axes[1].set_xlabel("Арнаның орташа мәні")
    axes[1].set_ylabel("Сурет саны")
    axes[1].set_title("Түс арналары")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_brightness.png")
    plt.show() if show else plt.close(fig)


def fig_samples(df: pd.DataFrame, n: int = 18, show: bool = False) -> None:
    kept = df[(~df["excluded"]) & (df["split"] == "train")]
    rng = np.random.default_rng(SEED)
    breeds = rng.choice(kept["breed"].unique(), size=n, replace=False)
    fig, axes = plt.subplots(3, n // 3, figsize=(n // 3 * 2.4, 3 * 2.6))
    for ax, b in zip(axes.ravel(), breeds):
        row = kept[kept["breed"] == b].sample(1, random_state=SEED).iloc[0]
        with Image.open(ROOT / row["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(pretty_breed(b), fontsize=9)
        ax.axis("off")
    fig.suptitle("Деректерден мысалдар: әр сурет — әртүрлі тұқым", x=0.01, ha="left", fontsize=12,
                 fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_samples.png")
    plt.show() if show else plt.close(fig)


def fig_duplicates(n_pairs: int = 4, show: bool = False) -> list[dict]:
    quality = load_json(DATA_QUALITY_JSON)
    conflicts = [e for e in quality["duplicate_examples"] if len(set(e["breeds"])) > 1][:n_pairs]
    fig, axes = plt.subplots(1, 2 * len(conflicts), figsize=(2.6 * 2 * len(conflicts), 3.2))
    for k, e in enumerate(conflicts):
        for j in range(2):
            ax = axes[2 * k + j]
            with Image.open(ROOT / "data/raw/stanford_dogs/Images" / e["paths"][j]) as im:
                ax.imshow(im.convert("RGB"))
            ax.set_title(f"{pretty_breed(e['breeds'][j])}\n({e['splits'][j]})", fontsize=9)
            ax.axis("off")
    fig.suptitle("Бірдей сурет — екі түрлі тұқым: деректердегі белгі қателері", x=0.01, ha="left",
                 fontsize=12, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eda_duplicates.png")
    plt.show() if show else plt.close(fig)
    return conflicts


def run_all(show: bool = False) -> dict:
    plot_style.apply()
    df = load_all()
    stats = basic_stats(df)
    fig_class_distribution(df, show)
    fig_image_sizes(df, show)
    fig_bbox(df, show)
    fig_brightness(df, show)
    fig_samples(df, show=show)
    fig_duplicates(show=show)
    save_json(stats, METRICS_DIR / "eda_stats.json")
    return stats
