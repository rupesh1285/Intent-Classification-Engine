"""
Stacked generalization (OOF probs → meta LogReg).

Bases: TF-IDF/SVM + MiniLM (+ DistilBERT if --with-distilbert and weights exist).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.embedder import SentenceEmbedder
from src.train import build_pipeline
from src.train_embeddings import build_pipeline as build_emb_pipeline

sys.modules["__main__"].SentenceEmbedder = SentenceEmbedder  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["benign", "injection"]
SEED = 42


def align_proba(proba: np.ndarray, classes_from) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes_from)}
    return np.stack([proba[:, idx[c]] for c in CLASSES], axis=1)


def encode_texts(texts, batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return np.asarray(
        model.encode(
            list(map(str, texts)),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    )


def distilbert_proba(texts, model_dir: Path, batch_size: int = 16, max_len: int = 128) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    outs = []
    texts = list(map(str, texts))
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(batch, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            col = {id2label[j]: probs[:, j] for j in range(probs.shape[1])}
            outs.append(np.stack([col[c] for c in CLASSES], axis=1))
    return np.vstack(outs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "clean" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--with-distilbert", action="store_true")
    parser.add_argument("--distil-dir", type=Path, default=ROOT / "models" / "distilbert")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    X = train_df["text"].astype(str)
    y = train_df["intent"].astype(str)
    X_test = test_df["text"].astype(str)
    y_test = test_df["intent"].astype(str)

    # class order from a quick fit
    probe = build_pipeline()
    probe.fit(X.iloc[:300], y.iloc[:300])
    tfidf_classes = list(probe.classes_)

    print(f"OOF TF-IDF ({args.cv}-fold) on {len(X)} rows...")
    oof_tfidf = align_proba(
        cross_val_predict(build_pipeline(), X, y, cv=args.cv, method="predict_proba"),
        tfidf_classes,
    )

    print("encoding train with MiniLM (once)...")
    E_train = encode_texts(X)
    emb_head = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True)),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight="balanced", C=2.0, random_state=SEED
                ),
            ),
        ]
    )
    print(f"OOF MiniLM-LogReg ({args.cv}-fold)...")
    oof_emb_raw = cross_val_predict(emb_head, E_train, y, cv=args.cv, method="predict_proba")
    emb_head.fit(E_train, y)
    oof_emb = align_proba(oof_emb_raw, emb_head.classes_)

    parts = [oof_tfidf, oof_emb]
    base_names = ["tfidf", "minilm"]

    use_distil = args.with_distilbert and (args.distil_dir / "config.json").exists()
    if use_distil:
        print("adding DistilBERT train probs to meta features...")
        parts.append(distilbert_proba(X, args.distil_dir))
        base_names.append("distilbert")
    elif args.with_distilbert:
        print("WARNING: DistilBERT weights missing — stacking without it")

    meta = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=SEED)
    meta.fit(np.hstack(parts), y)
    print(f"fitted meta-LogReg on bases={base_names}")

    print("fitting final TF-IDF + embedding pipelines on full train...")
    tfidf_full = build_pipeline()
    tfidf_full.fit(X, y)
    emb_full = build_emb_pipeline()
    emb_full.fit(X, y)

    p_test = [
        align_proba(tfidf_full.predict_proba(X_test), tfidf_full.classes_),
        align_proba(emb_full.predict_proba(X_test), emb_full.classes_),
    ]
    if use_distil:
        p_test.append(distilbert_proba(X_test, args.distil_dir))

    y_pred = meta.predict(np.hstack(p_test))
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(y_test, y_pred)
    print(report)
    print(f"stacking acc={acc:.4f} macro_f1={f1m:.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(tfidf_full, args.out_dir / "intent_classifier.joblib")
    joblib.dump(emb_full, args.out_dir / "embedding_classifier.joblib")
    joblib.dump(meta, args.out_dir / "stack_meta.joblib")
    (args.out_dir / "stacking_config.json").write_text(
        json.dumps({"bases": base_names, "classes": CLASSES, "cv": args.cv}, indent=2),
        encoding="utf-8",
    )

    metrics = {
        "macro_f1": round(float(f1m), 4),
        "weighted_f1": round(float(f1w), 4),
        "accuracy": round(float(acc), 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "model": f"stacking meta-LogReg on {('+').join(base_names)} (OOF cv={args.cv})",
        "bases": base_names,
        "report": report,
    }
    (args.out_dir / "stacking_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_path = args.out_dir / "metrics.json"
    old_f1 = 0.0
    if metrics_path.exists():
        old_f1 = float(json.loads(metrics_path.read_text(encoding="utf-8")).get("macro_f1", 0))
    if f1m > old_f1:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (args.out_dir / "active_model.txt").write_text("stacking\n", encoding="utf-8")
        print("stacking is new best — active_model=stacking")
    else:
        print("stacking did not beat current best — artifacts saved anyway")


if __name__ == "__main__":
    main()
