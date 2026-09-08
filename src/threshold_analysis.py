"""
Injection-focused threshold sweep on calibrated TF-IDF probs.

For security use-cases you often want higher recall on `injection`
even if precision dips a bit. Writes reports/threshold_analysis.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "threshold_analysis.md")
    args = parser.parse_args()

    pipe = joblib.load(args.model)
    df = pd.read_csv(args.test)
    y_true = df["intent"].astype(str).to_numpy()
    texts = df["text"].astype(str)
    proba = pipe.predict_proba(texts)
    classes = list(pipe.classes_)
    inj_i = classes.index("injection")
    p_inj = proba[:, inj_i]

    rows = []
    for t in np.round(np.arange(0.30, 0.71, 0.05), 2):
        y_pred = np.where(p_inj >= t, "injection", "benign")
        rows.append(
            {
                "threshold": float(t),
                "acc": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro"),
                "inj_precision": precision_score(y_true, y_pred, pos_label="injection"),
                "inj_recall": recall_score(y_true, y_pred, pos_label="injection"),
            }
        )

    # default argmax (~0.5) for reference
    y_default = pipe.predict(texts)
    default = {
        "threshold": 0.5,
        "acc": accuracy_score(y_true, y_default),
        "macro_f1": f1_score(y_true, y_default, average="macro"),
        "inj_precision": precision_score(y_true, y_default, pos_label="injection"),
        "inj_recall": recall_score(y_true, y_default, pos_label="injection"),
    }

    best_f1 = max(rows, key=lambda r: r["macro_f1"])
    best_rec = max(rows, key=lambda r: r["inj_recall"])

    lines = [
        "# Threshold analysis (injection class)",
        "",
        "Calibrated TF-IDF/SVM — sweep `P(injection)` threshold.",
        "",
        "| threshold | accuracy | macro F1 | inj precision | inj recall |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['acc']:.4f} | {r['macro_f1']:.4f} | "
            f"{r['inj_precision']:.4f} | {r['inj_recall']:.4f} |"
        )
    lines += [
        "",
        f"- default predict (~0.5): acc={default['acc']:.4f}, macro_f1={default['macro_f1']:.4f}, "
        f"inj_recall={default['inj_recall']:.4f}",
        f"- best macro F1 in sweep: t={best_f1['threshold']:.2f} → F1={best_f1['macro_f1']:.4f}",
        f"- highest injection recall in sweep: t={best_rec['threshold']:.2f} → recall={best_rec['inj_recall']:.4f}",
        "",
        "Takeaway: lower the threshold if you want fewer missed jailbreaks (more false alarms).",
        "",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
