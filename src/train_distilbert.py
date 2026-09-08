"""
Fine-tune DistilBERT for benign vs injection.

Default: 1 epoch, max_len=128, CPU-safe Trainer flags.
Saves to models/distilbert/ and updates active_model only if it beats metrics.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
LABEL2ID = {"benign": 0, "injection": 1}
ID2LABEL = {0: "benign", 1: "injection"}


class TextClsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = list(texts)
        self.labels = [LABEL2ID[str(y)] for y in labels]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "clean" / "train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data" / "clean" / "test.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models" / "distilbert")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=128)
    parser.add_argument("--freeze-base", action="store_true", help="train classifier head only (faster CPU)")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    if args.freeze_base:
        for p in model.distilbert.parameters():
            p.requires_grad = False
        print("frozen DistilBERT encoder — training head only")

    train_ds = TextClsDataset(train_df["text"], train_df["intent"], tokenizer, args.max_len)
    test_ds = TextClsDataset(test_df["text"], test_df["intent"], tokenizer, args.max_len)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targs = TrainingArguments(
        output_dir=str(args.out_dir / "runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        learning_rate=2e-5 if not args.freeze_base else 1e-3,
        weight_decay=0.01,
        logging_steps=50,
        report_to=[],
        fp16=False,
        dataloader_pin_memory=False,
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
    )

    print(f"fine-tuning {args.model_name} for {args.epochs} epochs on {len(train_df)} rows...")
    trainer.train()
    pred_out = trainer.predict(test_ds)
    preds = np.argmax(pred_out.predictions, axis=-1)
    y_true = [LABEL2ID[str(y)] for y in test_df["intent"]]
    y_lbl = [ID2LABEL[i] for i in preds]
    y_true_lbl = [ID2LABEL[i] for i in y_true]

    acc = accuracy_score(y_true_lbl, y_lbl)
    f1m = f1_score(y_true_lbl, y_lbl, average="macro")
    f1w = f1_score(y_true_lbl, y_lbl, average="weighted")
    report = classification_report(y_true_lbl, y_lbl)
    print(report)
    print(f"acc={acc:.4f} macro_f1={f1m:.4f}")

    trainer.save_model(str(args.out_dir))
    tokenizer.save_pretrained(str(args.out_dir))
    tag = f"{args.model_name} fine-tuned ({args.epochs} epochs"
    if args.freeze_base:
        tag += ", frozen base"
    tag += ")"
    metrics = {
        "macro_f1": round(float(f1m), 4),
        "weighted_f1": round(float(f1w), 4),
        "accuracy": round(float(acc), 4),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "model": tag,
        "report": report,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (ROOT / "models" / "distilbert_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    metrics_path = ROOT / "models" / "metrics.json"
    old_f1 = 0.0
    old_active = "ensemble"
    if metrics_path.exists():
        old = json.loads(metrics_path.read_text(encoding="utf-8"))
        old_f1 = float(old.get("macro_f1", 0))
    active_path = ROOT / "models" / "active_model.txt"
    if active_path.exists():
        old_active = active_path.read_text(encoding="utf-8").strip() or old_active

    if f1m > old_f1:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        active_path.write_text("distilbert\n", encoding="utf-8")
        print("DistilBERT beats previous — set as active model")
    else:
        active_path.write_text(f"{old_active}\n", encoding="utf-8")
        print(f"kept previous active_model={old_active} (DistilBERT macro_f1={f1m:.4f} vs best={old_f1:.4f})")


if __name__ == "__main__":
    main()
