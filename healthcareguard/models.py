"""Model builders, training, thresholding and metrics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from .features import RuleFeatures
from .lexicon import normalize_text

KINDS = ("v01", "tfidf", "rules", "hybrid")


def prep(s):
    return normalize_text(s).lower()


def _word():
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, preprocessor=prep)


def _char():
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2,
                           preprocessor=prep)


def make_pipeline(kind: str, C: float = 3.0, multiclass: bool = False) -> Pipeline:
    """v01   : the original prototype (word 1-2gram TF-IDF + LR)
       tfidf : word + char TF-IDF + LR
       rules : interpretable cue features only
       hybrid: TF-IDF + cue features (the healthcare-specific detector)"""
    lr = LogisticRegression(max_iter=3000, class_weight="balanced", C=C)
    if kind == "v01":
        return Pipeline([("tfidf", _word()), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])
    if kind == "tfidf":
        feats = FeatureUnion([("word", _word()), ("char", _char())])
    elif kind == "rules":
        feats = FeatureUnion([("rules", RuleFeatures())])
    elif kind == "hybrid":
        feats = FeatureUnion([("word", _word()), ("char", _char()), ("rules", RuleFeatures())])
    else:
        raise ValueError(f"unknown kind {kind}")
    return Pipeline([("features", feats), ("clf", lr)])


def fit_binary(kind: str, df: pd.DataFrame, **kw) -> Pipeline:
    d = df[["text", "label"]].dropna()
    if d.label.nunique() != 2:
        raise ValueError("Need both benign (0) and attack (1) examples.")
    return make_pipeline(kind, **kw).fit(d.text.astype(str).tolist(), d.label.astype(int).values)


def fit_category(df: pd.DataFrame) -> Pipeline:
    a = df[(df.label == 1)]
    return make_pipeline("hybrid").fit(a.text.tolist(), a.category.tolist())


def proba(pipe: Pipeline, texts) -> np.ndarray:
    return pipe.predict_proba(list(texts))[:, 1]


# ------------------------------------------------------------------ metrics
def binary_metrics(y, p, thr=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    auc = float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "fpr": fpr, "f1": f1, "accuracy": (tp + tn) / max(len(y), 1), "auroc": auc}


def bootstrap_ci(y, p, thr, key, n=400, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(binary_metrics(y[i], p[i], thr)[key])
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2)))


def choose_threshold(y, p, beta=2.0) -> float:
    """Pick the threshold maximising F-beta (recall-weighted: for safety screening a miss is costlier
    than a review). Ties resolve to the median of tied thresholds."""
    y = np.asarray(y).astype(int)
    best, grid = -1.0, np.round(np.arange(0.05, 0.96, 0.01), 2)
    scores = []
    for t in grid:
        m = binary_metrics(y, p, t)
        pr, rc = m["precision"], m["recall"]
        fb = (1 + beta**2) * pr * rc / (beta**2 * pr + rc) if (pr + rc) else 0.0
        scores.append(fb)
    best = max(scores)
    tied = [t for t, s in zip(grid, scores) if abs(s - best) < 1e-9]
    return float(np.median(tied))
