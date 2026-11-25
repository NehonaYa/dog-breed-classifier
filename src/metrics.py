"""Бағалау метрикалары.

Тек accuracy жеткіліксіз, сондықтан бірнеше көрсеткіш есептейміз:
  accuracy        — дұрыс жауаптар үлесі;
  top5_accuracy   — дұрыс тұқым модельдің алғашқы 5 болжамының ішінде бар ма;
  macro_f1        — әр тұқым бойынша F1-дің орташасы (сирек тұқымдар да тең ескеріледі);
  log_loss        — ықтималдықтардың сапасы (сенімді қате жауап қатаң жазаланады);
  ece             — калибрлеу қатесі: модель "90% сенімдімін" десе, шынымен 90% дұрыс па?
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, log_loss,
                             precision_recall_fscore_support)


def top_k_accuracy(y: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(y: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return rows


def classification_metrics(y: np.ndarray, probs: np.ndarray) -> dict:
    probs = np.clip(probs.astype(np.float64), 1e-7, 1.0)
    probs = probs / probs.sum(1, keepdims=True)
    pred = probs.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "top5_accuracy": top_k_accuracy(y, probs, 5),
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, probs, labels=np.arange(probs.shape[1]))),
        "ece": expected_calibration_error(y, probs),
    }


def per_class_table(y: np.ndarray, pred: np.ndarray, names: list[str]):
    import pandas as pd
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=np.arange(len(names)), zero_division=0)
    return pd.DataFrame({"breed": names, "precision": p, "recall": r, "f1": f, "support": s})


def confused_pairs(y: np.ndarray, pred: np.ndarray, names: list[str], top: int = 15):
    import pandas as pd
    wrong = y != pred
    pairs = pd.Series(list(zip(y[wrong], pred[wrong]))).value_counts().head(top)
    support = np.bincount(y, minlength=len(names))
    return pd.DataFrame([{"true": names[t], "predicted": names[p_], "count": int(c),
                          "share_of_true_class": round(c / support[t], 3)} for (t, p_), c in pairs.items()])
