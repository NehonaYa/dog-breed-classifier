"""1-қадам: Stanford Dogs деректерін ашу, тексеру, тазалау және train / val / test бөлу.

Не істейді:
  1) images.tar, annotation.tar, lists.tar архивтерін ашады;
  2) ресми train/test тізімдерін (.mat) оқиды;
  3) әр суретті тексереді: бұзылған файл, формат (JPEG емес), түс режимі (RGB емес);
  4) стандартқа сай емес суреттерді RGB JPEG-ке айналдырады (data/processed/fixed_images);
  5) толық және жуық көшірмелерді (MD5 + dHash) іздейді, train/val мен test арасындағы
     "деректердің ағып кетуін" (leakage) болдырмау үшін көшірмені train/val-дан алып тастайды;
  6) аннотациядан (bounding box) суреттегі ит саны мен иттің кадрдағы үлесін есептейді;
  7) ресми train-нен әр класс бойынша 15 суретті валидацияға бөледі (stratified).

Іске қосу (жобаның түбірінен):
    .venv\\Scripts\\python -I scripts/01_prepare_data.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import hashlib  # noqa: E402
import tarfile  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.io import loadmat  # noqa: E402

from src.config import (ANNOT_DIR, DATA_QUALITY_JSON, DATA_RAW, FIXED_IMAGES_DIR,  # noqa: E402
                        IMAGES_DIR, SEED, SPLITS_CSV, VAL_PER_CLASS)
from src.data_utils import breed_from_folder, save_json  # noqa: E402

DHASH_THRESHOLD = 2      # dHash Хэмминг қашықтығы <= 2 болса, жуық көшірме деп санаймыз
ASPECT_TOLERANCE = 0.05  # көшірме жұбының пропорциясы (ені/биіктігі) да ұқсас болуы керек


def extract_archives() -> None:
    targets = [("images.tar", IMAGES_DIR), ("annotation.tar", ANNOT_DIR),
               ("lists.tar", DATA_RAW / "train_list.mat")]
    for name, marker in targets:
        if marker.exists():
            continue
        tar_path = DATA_RAW / name
        if not tar_path.exists():
            sys.exit(f"{tar_path} табылмады. Алдымен scripts/00_download_data.py іске қосыңыз.")
        print(f"Архив ашылуда: {name} ...", flush=True)
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA_RAW, filter="data")  # filter="data" — қауіпсіз ашу (path traversal-дан қорғау)


def read_official_lists() -> pd.DataFrame:
    rows = []
    for split, fname in [("train", "train_list.mat"), ("test", "test_list.mat")]:
        m = loadmat(DATA_RAW / fname)
        files = [str(np.ravel(x)[0]) for x in np.ravel(m["file_list"])]
        labels = np.ravel(m["labels"]).astype(int) - 1  # .mat-та 1..120, бізге 0..119
        for f, lab in zip(files, labels):
            rows.append({"orig_rel": f.replace("\\", "/"), "mat_label": int(lab), "official_split": split})
    return pd.DataFrame(rows)


def inspect_image(abspath: str) -> dict:
    """Бір суретті тексеру: MD5, формат, режим, өлшем, dHash, орташа түс."""
    p = Path(abspath)
    data = p.read_bytes()
    info = {"md5": hashlib.md5(data).hexdigest(), "file_kb": round(len(data) / 1024, 1)}
    try:
        with Image.open(p) as im:
            im.verify()  # бұзылған файлды анықтайды
        with Image.open(p) as im:
            info.update(fmt=im.format, mode=im.mode, width=im.width, height=im.height)
            gray = np.asarray(im.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
            bits = (gray[:, 1:] > gray[:, :-1]).flatten()
            info["dhash"] = format(int(np.packbits(bits).view(">u8")[0]), "016x")
            rgb = np.asarray(im.convert("RGB").resize((32, 32)), dtype=np.float32).reshape(-1, 3)
            info["mean_r"], info["mean_g"], info["mean_b"] = (round(float(v), 2) for v in rgb.mean(0))
            info["brightness"] = round(float(rgb.mean()), 2)
            info["ok"] = True
    except Exception as exc:  # бұзылған немесе оқылмайтын файл
        info.update(ok=False, error=str(exc))
    return info


def parse_annotation(path: Path) -> dict:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        x1, y1, x2, y2 = (float(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        boxes.append((x1, y1, x2, y2))
    return {"n_dogs": len(boxes), "boxes": boxes}


def fix_image(src: Path, dst: Path) -> None:
    """JPEG емес немесе RGB/L емес суретті RGB JPEG-ке айналдыру."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im.convert("RGB").save(dst, "JPEG", quality=95)


def find_near_duplicates(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    """dHash арқылы жуық көшірме жұптарын табу (numpy векторлық есептеу)."""
    h = np.array([int(x, 16) for x in df["dhash"]], dtype=np.uint64)
    aspect = (df["width"] / df["height"]).to_numpy()
    pairs = []
    for i0 in range(0, len(h), 512):
        block = h[i0:i0 + 512]
        dist = np.bitwise_count(block[:, None] ^ h[None, :])
        ii, jj = np.nonzero(dist <= DHASH_THRESHOLD)
        for i, j in zip(ii, jj):
            gi = i0 + i
            if gi < j and abs(aspect[gi] - aspect[j]) < ASPECT_TOLERANCE:
                pairs.append((int(gi), int(j), int(dist[i, j])))
    return pairs


def cluster_pairs(n: int, pairs) -> list[list[int]]:
    """Көшірме жұптарын топтарға біріктіру (union-find)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in pairs:
        parent[find(a)] = find(b)
    members: dict[int, set] = {}
    for a, b, _ in pairs:
        members.setdefault(find(a), set()).update((a, b))
    return [sorted(m) for m in members.values()]


def main() -> None:
    extract_archives()
    df = read_official_lists()
    df["folder"] = df["orig_rel"].str.split("/").str[0]
    df["synset"] = df["folder"].str.split("-").str[0]
    df["breed"] = df["folder"].map(breed_from_folder)
    folders = sorted(df["folder"].unique())
    df["label"] = df["folder"].map({f: i for i, f in enumerate(folders)})
    assert (df["label"] == df["mat_label"]).all(), "Класс индекстері .mat файлымен сәйкес келмейді"
    print(f"Барлығы {len(df)} сурет, {len(folders)} класс", flush=True)

    # --- 1. Әр суретті тексеру (параллель) ---
    abspaths = [str(IMAGES_DIR / r) for r in df["orig_rel"]]
    with ProcessPoolExecutor(max_workers=8) as ex:
        infos = list(ex.map(inspect_image, abspaths, chunksize=64))
    info_df = pd.DataFrame(infos)
    df = pd.concat([df.reset_index(drop=True), info_df], axis=1)
    n_corrupt = int((~df["ok"]).sum())
    print(f"Бұзылған файлдар: {n_corrupt}", flush=True)

    # --- 2. Аннотация: ит саны және кадрдағы үлесі ---
    n_dogs, frac_max, frac_sum = [], [], []
    for rel, w, h in zip(df["orig_rel"], df["width"], df["height"]):
        ann = parse_annotation(ANNOT_DIR / rel.removesuffix(".jpg"))
        areas = [max(0.0, x2 - x1) * max(0.0, y2 - y1) for x1, y1, x2, y2 in ann["boxes"]]
        img_area = float(w * h) if w and h else np.nan
        n_dogs.append(ann["n_dogs"])
        frac_max.append(min(1.0, max(areas) / img_area) if areas else np.nan)
        frac_sum.append(min(1.0, sum(areas) / img_area) if areas else np.nan)
    df["n_dogs"], df["bbox_frac_max"], df["bbox_frac_sum"] = n_dogs, frac_max, frac_sum

    # --- 3. Стандартқа сай емес суреттерді түзету ---
    df["path"] = ("data/raw/stanford_dogs/Images/" + df["orig_rel"])
    need_fix = df["ok"] & ((df["fmt"] != "JPEG") | (~df["mode"].isin(["RGB", "L"])))
    fix_reasons = Counter()
    for idx in df.index[need_fix]:
        rel = df.at[idx, "orig_rel"]
        fix_reasons[f"{df.at[idx, 'fmt']}/{df.at[idx, 'mode']}"] += 1
        dst = FIXED_IMAGES_DIR / rel
        fix_image(IMAGES_DIR / rel, dst)
        df.at[idx, "path"] = dst.relative_to(ROOT).as_posix()
    df["fixed"] = need_fix
    print(f"Түзетілген суреттер: {int(need_fix.sum())} {dict(fix_reasons)}", flush=True)

    # --- 4. Көшірмелер ---
    df["excluded"] = ~df["ok"]
    df["exclude_reason"] = np.where(~df["ok"], "corrupt", "")
    good = df[df["ok"]].reset_index()  # 'index' бағаны — бастапқы индекс
    pairs = find_near_duplicates(good)
    clusters = cluster_pairs(len(good), pairs)
    label_conflicts, examples = 0, []
    for cl in clusters:
        rows = good.loc[cl]
        orig_idx = rows["index"].tolist()
        conflict = rows["label"].nunique() > 1
        label_conflicts += int(conflict)
        test_members = [i for i, s in zip(orig_idx, rows["official_split"]) if s == "test"]
        trainval_members = [i for i, s in zip(orig_idx, rows["official_split"]) if s == "train"]
        if test_members or conflict:
            drop = trainval_members  # test-ті өзгертпейміз, train көшірмесін алып тастаймыз
            reason = "label_conflict" if conflict else "duplicate_of_test"
        else:
            drop = trainval_members[1:]  # train ішіндегі көшірмеден біреуін ғана қалдырамыз
            reason = "duplicate_in_train"
        for i in drop:
            df.at[i, "excluded"] = True
            df.at[i, "exclude_reason"] = reason
        if len(examples) < 20:
            examples.append({"paths": rows["orig_rel"].tolist(), "breeds": rows["breed"].tolist(),
                             "splits": rows["official_split"].tolist()})
    print(f"Жуық көшірме жұптары: {len(pairs)}, топтар: {len(clusters)}, "
          f"белгі қайшылығы: {label_conflicts}", flush=True)

    # --- 5. train / val / test ---
    df["split"] = df["official_split"]
    rng = np.random.default_rng(SEED)
    cand = df[(df["official_split"] == "train") & (~df["excluded"])]
    for _, grp in cand.groupby("label"):
        val_idx = rng.choice(grp.index.to_numpy(), size=min(VAL_PER_CLASS, len(grp)), replace=False)
        df.loc[val_idx, "split"] = "val"

    cols = ["path", "orig_rel", "synset", "breed", "label", "official_split", "split", "excluded",
            "exclude_reason", "fixed", "fmt", "mode", "width", "height", "file_kb", "n_dogs",
            "bbox_frac_max", "bbox_frac_sum", "mean_r", "mean_g", "mean_b", "brightness", "md5", "dhash"]
    df[cols].to_csv(SPLITS_CSV, index=False)

    kept = df[~df["excluded"]]
    quality = {
        "total_images": int(len(df)),
        "classes": int(df["label"].nunique()),
        "corrupt_files": n_corrupt,
        "fixed_images": int(need_fix.sum()),
        "fixed_reasons": dict(fix_reasons),
        "near_duplicate_pairs": len(pairs),
        "duplicate_clusters": len(clusters),
        "label_conflict_clusters": label_conflicts,
        "excluded_by_reason": df.loc[df["excluded"], "exclude_reason"].value_counts().to_dict(),
        "split_counts": kept["split"].value_counts().to_dict(),
        "missing_annotation": int(df["n_dogs"].isna().sum()),
        "duplicate_examples": examples,
    }
    save_json(quality, DATA_QUALITY_JSON)
    print("Бөлініс:", quality["split_counts"])
    print("Сақталды:", SPLITS_CSV)


if __name__ == "__main__":
    main()
