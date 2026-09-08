"""Quick eval on the saved model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    args = parser.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.test)
    y_true = df["intent"].astype(str)
    y_pred = pipe.predict(df["text"].astype(str))
    print(classification_report(y_true, y_pred))
    print(f"macro_f1={f1_score(y_true, y_pred, average='macro'):.4f}")


if __name__ == "__main__":
    main()
