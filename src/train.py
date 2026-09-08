"""Train the classifier and dump metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DEFAULT = ROOT / "data" / "clean" / "train.csv"
TEST_DEFAULT = ROOT / "data" / "clean" / "test.csv"
MODELS_DIR = ROOT / "models"
SEED = 42


def build_pipeline() -> Pipeline:
    # word + char ngrams helped more than plain logreg on this data
    return Pipeline(
        [
            (
                "feats",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                ngram_range=(1, 3),
                                min_df=2,
                                max_df=0.9,
                                sublinear_tf=True,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 6),
                                min_df=2,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                CalibratedClassifierCV(
                    LinearSVC(
                        class_weight="balanced",
                        C=0.5,
                        random_state=SEED,
                        dual="auto",
                    ),
                    cv=3,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=TRAIN_DEFAULT)
    parser.add_argument("--test", type=Path, default=TEST_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    X_train, y_train = train_df["text"].astype(str), train_df["intent"].astype(str)
    X_test, y_test = test_df["text"].astype(str), test_df["intent"].astype(str)

    pipe = build_pipeline()
    print(f"training on {len(train_df)} rows...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)

    print(report)
    print(f"acc={acc:.4f} macro_f1={f1_macro:.4f} weighted_f1={f1_weighted:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "intent_classifier.joblib"
    metrics_path = args.out_dir / "metrics.json"
    joblib.dump(pipe, model_path)
    metrics_path.write_text(
        json.dumps(
            {
                "macro_f1": round(float(f1_macro), 4),
                "weighted_f1": round(float(f1_weighted), 4),
                "accuracy": round(float(acc), 4),
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "model": "word(1-3)+char(3-6) LinearSVC C=0.5 calibrated",
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("saved", model_path)
    print("saved", metrics_path)


if __name__ == "__main__":
    main()
