"""Dump mistakes + confusion matrix for the saved sklearn model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "intent_classifier.joblib")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pipe = joblib.load(args.model)
    df = pd.read_csv(args.test)
    texts = df["text"].astype(str)
    y_true = df["intent"].astype(str)
    y_pred = pipe.predict(texts)

    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion matrix (test)")
    fig.tight_layout()
    cm_path = args.out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=140)
    plt.close(fig)

    err = df.copy()
    err["pred"] = y_pred
    err = err[err["intent"] != err["pred"]].copy()
    # confidence if available
    if hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(texts)
        classes = list(pipe.classes_)
        conf = []
        for i, p in enumerate(y_pred):
            conf.append(float(proba[i][classes.index(p)]))
        err["pred_confidence"] = [conf[i] for i in err.index]
    err = err.sort_values("pred_confidence", ascending=True) if "pred_confidence" in err.columns else err
    err_path = args.out_dir / "errors.csv"
    err.to_csv(err_path, index=False)

    summary = {
        "n_test": int(len(df)),
        "n_errors": int(len(err)),
        "error_rate": round(len(err) / max(len(df), 1), 4),
        "report": classification_report(y_true, y_pred, digits=4),
    }
    (args.out_dir / "error_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = [
        "# Error analysis",
        "",
        f"- test size: {summary['n_test']}",
        f"- mistakes: {summary['n_errors']} ({summary['error_rate']*100:.2f}%)",
        f"- confusion matrix: `{cm_path.name}`",
        f"- full error rows: `{err_path.name}`",
        "",
        "## Classification report",
        "```",
        summary["report"],
        "```",
        "",
        "## Sample mistakes (up to 12)",
        "",
    ]
    for _, r in err.head(12).iterrows():
        text = str(r["text"]).replace("\n", " ")
        if len(text) > 180:
            text = text[:180] + "..."
        md.append(f"- true=`{r['intent']}` pred=`{r['pred']}` :: {text}")
    (args.out_dir / "ERROR_ANALYSIS.md").write_text("\n".join(md), encoding="utf-8")
    print("wrote", cm_path)
    print("wrote", err_path)
    print("wrote", args.out_dir / "ERROR_ANALYSIS.md")


if __name__ == "__main__":
    main()
