"""
Soft-vote ensemble: calibrated TF-IDF/SVM + sentence embeddings LogReg.

Often beats either alone on ambiguous prompt-injection text.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from src.embedder import SentenceEmbedder

# old joblibs may have pickled SentenceEmbedder as __main__.*
sys.modules["__main__"].SentenceEmbedder = SentenceEmbedder  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]


def soft_vote(proba_a: np.ndarray, classes_a, proba_b: np.ndarray, classes_b, weight_a: float = 0.55):
    """Average class probabilities (aligned by class name)."""
    classes = list(classes_a)
    idx_b = {c: i for i, c in enumerate(classes_b)}
    aligned_b = np.stack([proba_b[:, idx_b[c]] for c in classes], axis=1)
    return weight_a * proba_a + (1.0 - weight_a) * aligned_b, classes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--tfidf", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    parser.add_argument("--emb", type=Path, default=ROOT / "models" / "embedding_classifier.joblib")
    parser.add_argument("--weight-tfidf", type=float, default=0.6)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    if not args.emb.exists():
        raise SystemExit(f"missing {args.emb} — run: python -m src.train_embeddings")

    df = pd.read_csv(args.test)
    texts = df["text"].astype(str)
    y_true = df["intent"].astype(str)

    tfidf = joblib.load(args.tfidf)
    emb = joblib.load(args.emb)

    p_a = tfidf.predict_proba(texts)
    p_b = emb.predict_proba(texts)
    blended, classes = soft_vote(p_a, tfidf.classes_, p_b, emb.classes_, args.weight_tfidf)
    y_pred = np.array(classes)[np.argmax(blended, axis=1)]

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    report = classification_report(y_true, y_pred)
    print(report)
    print(f"ensemble acc={acc:.4f} macro_f1={f1m:.4f} (tfidf_w={args.weight_tfidf})")

    metrics = {
        "macro_f1": round(float(f1m), 4),
        "weighted_f1": round(float(f1w), 4),
        "accuracy": round(float(acc), 4),
        "n_test": int(len(df)),
        "model": f"soft-vote TF-IDF({args.weight_tfidf:.2f}) + MiniLM({1 - args.weight_tfidf:.2f})",
        "report": report,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "ensemble_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # persist blend weights for demo
    cfg = {"weight_tfidf": args.weight_tfidf, "classes": list(classes)}
    (args.out_dir / "ensemble_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    old = json.loads((args.out_dir / "metrics.json").read_text(encoding="utf-8"))
    if f1m > float(old.get("macro_f1", 0)):
        (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (args.out_dir / "active_model.txt").write_text("ensemble\n", encoding="utf-8")
        print("ensemble is new best — set active_model=ensemble")
    else:
        print("ensemble did not beat current best — metrics saved only")


if __name__ == "__main__":
    main()
