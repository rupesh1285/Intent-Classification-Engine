"""
Stronger baseline: sentence embeddings + linear classifier.

Uses all-MiniLM-L6-v2 (fast). Saves:
  models/embedding_classifier.joblib
  models/embedding_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.embedder import SentenceEmbedder

ROOT = Path(__file__).resolve().parents[1]


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("emb", SentenceEmbedder()),
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    C=2.0,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "clean" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    X_train, y_train = train_df["text"].astype(str), train_df["intent"].astype(str)
    X_test, y_test = test_df["text"].astype(str), test_df["intent"].astype(str)

    pipe = build_pipeline()
    print(f"embedding+logreg on {len(train_df)} rows...")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, pred)
    f1m = f1_score(y_test, pred, average="macro")
    f1w = f1_score(y_test, pred, average="weighted")
    report = classification_report(y_test, pred)
    print(report)
    print(f"acc={acc:.4f} macro_f1={f1m:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "embedding_classifier.joblib"
    metrics_path = args.out_dir / "embedding_metrics.json"
    joblib.dump(pipe, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "macro_f1": round(float(f1m), 4),
                "weighted_f1": round(float(f1w), 4),
                "accuracy": round(float(acc), 4),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "model": "all-MiniLM-L6-v2 embeddings + LogisticRegression",
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", model_path)

    # if better than tfidf model, also promote to default intent_classifier.joblib
    tfidf_metrics = args.out_dir / "metrics.json"
    promote = True
    if tfidf_metrics.exists():
        old = json.loads(tfidf_metrics.read_text(encoding="utf-8"))
        promote = f1m >= float(old.get("macro_f1", 0))
    if promote:
        joblib.dump(pipe, args.out_dir / "intent_classifier.joblib")
        tfidf_metrics.write_text(
            json.dumps(
                {
                    "macro_f1": round(float(f1m), 4),
                    "weighted_f1": round(float(f1w), 4),
                    "accuracy": round(float(acc), 4),
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "model": "all-MiniLM-L6-v2 embeddings + LogisticRegression",
                    "report": report,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("promoted embedding model to intent_classifier.joblib")
    else:
        print("tfidf model still better; kept as default")


if __name__ == "__main__":
    main()
