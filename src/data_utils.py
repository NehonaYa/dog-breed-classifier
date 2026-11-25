"""Деректермен жұмыс істеуге арналған көмекші функциялар."""
import json

import pandas as pd

from src.config import ROOT, SPLITS_CSV


def breed_from_folder(folder: str) -> str:
    """'n02085620-Chihuahua' -> 'Chihuahua' (тек бірінші сызықша бойынша бөлеміз: 'Shih-Tzu' сақталады)."""
    return folder.split("-", 1)[1]


def pretty_breed(raw: str) -> str:
    """'golden_retriever' -> 'Golden retriever' (сайтта және графиктерде көрсету үшін)."""
    s = raw.replace("_", " ")
    return s[:1].upper() + s[1:]


def load_splits(drop_excluded: bool = True) -> pd.DataFrame:
    """splits.csv оқу. path бағаны жобаның түбіріне қатысты жол, abspath қосамыз."""
    df = pd.read_csv(SPLITS_CSV)
    if drop_excluded and "excluded" in df.columns:
        df = df[~df["excluded"]].reset_index(drop=True)
    df["abspath"] = df["path"].map(lambda p: str(ROOT / p))
    return df


def class_names(pretty: bool = True) -> list[str]:
    """Кластар тізімі label индексі ретімен (0..119)."""
    df = pd.read_csv(SPLITS_CSV, usecols=["label", "breed"]).drop_duplicates().sort_values("label")
    names = df["breed"].tolist()
    return [pretty_breed(n) for n in names] if pretty else names


def _to_builtin(o):
    """numpy сандарын JSON-ға жазуға болатын Python типтеріне айналдыру."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"JSON-ға жазылмайды: {type(o)}")


def save_json(obj, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_to_builtin)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
