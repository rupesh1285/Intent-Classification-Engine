# DistilBERT fine-tune (Colab / GPU)

Upload `data/clean/train.csv` + `test.csv`, then run. Usually 1–2 epochs on a free T4 is enough.

```python
# !pip -q install transformers datasets accelerate scikit-learn pandas
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

LABEL2ID = {"benign": 0, "injection": 1}
ID2LABEL = {0: "benign", 1: "injection"}

class DS(Dataset):
    def __init__(self, df, tok, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.labels = [LABEL2ID[y] for y in df["intent"].astype(str)]
        self.tok, self.max_len = tok, max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, i):
        enc = self.tok(self.texts[i], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    return {"accuracy": float(accuracy_score(p.label_ids, preds)),
            "macro_f1": float(f1_score(p.label_ids, preds, average="macro"))}

args = TrainingArguments(
    output_dir="runs",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    learning_rate=2e-5,
    weight_decay=0.01,
    fp16=torch.cuda.is_available(),
    report_to=[],
)
trainer = Trainer(model=model, args=args, train_dataset=DS(train_df, tok), eval_dataset=DS(test_df, tok), compute_metrics=compute_metrics)
trainer.train()
pred = trainer.predict(DS(test_df, tok))
y_hat = [ID2LABEL[i] for i in np.argmax(pred.predictions, axis=-1)]
print(classification_report(test_df["intent"], y_hat))
trainer.save_model("distilbert_export")
tok.save_pretrained("distilbert_export")
# zip + download distilbert_export → drop into local models/distilbert/
```

Then locally:

```powershell
.\.venv\Scripts\python.exe -m src.train_stacking --with-distilbert
```
