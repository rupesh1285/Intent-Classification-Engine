# Intent Classification Engine

Binary text classifier for prompts: **benign** vs **injection** (jailbreak / prompt injection style attacks).

Portfolio project — messy public data → clean split → model comparison → stacking → transformer fine-tune → error analysis → Gradio demo.

## Results (held-out test, n=4996)

| model | accuracy | macro F1 |
|------|----------|----------|
| MiniLM embeddings + LogReg | 0.8389 | 0.8359 |
| word+char TF-IDF + LinearSVC (calibrated) | 0.8885 | 0.8850 |
| OOF stacking (TF-IDF + MiniLM) | 0.8897 | 0.8870 |
| soft-vote ensemble (0.7 TF-IDF + 0.3 MiniLM) | 0.8933 | 0.8901 |
| DistilBERT fine-tuned (1 epoch) | 0.9111 | 0.9079 |
| **OOF stacking (TF-IDF + MiniLM + DistilBERT)** | **0.9159** | **0.9134** |

Active demo model: `models/active_model.txt` → **stacking**.

## Extra analysis

- Error dump + confusion matrix → `reports/`
- Injection threshold sweep → `reports/threshold_analysis.md`
- Model write-up → `MODEL_COMPARISON.md`
- Colab DistilBERT (GPU, 2 epochs) → `notebooks/COLAB_DISTILBERT.md`

## Project layout

```
data/raw/     raw + messy csv (large files gitignored)
data/clean/   cleaned csv + train/test split
src/          preprocess, train*, stacking, distilbert, analysis
models/       sklearn / stack meta / distilbert / metrics
reports/      mistakes, confusion matrix, thresholds
app/demo.py   gradio (tfidf | ensemble | stacking | distilbert)
notebooks/    Colab fine-tune notes
```

## Setup (Windows)

```powershell
cd "path\to\Intent Classification Engine"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run

```powershell
.\.venv\Scripts\python.exe -m src.preprocess
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe -m src.train_embeddings
.\.venv\Scripts\python.exe -m src.train_ensemble --weight-tfidf 0.7
.\.venv\Scripts\python.exe -m src.train_stacking
.\.venv\Scripts\python.exe -m src.train_distilbert --epochs 1
.\.venv\Scripts\python.exe -m src.train_stacking --with-distilbert
.\.venv\Scripts\python.exe -m app.demo
```

Demo (premium Gradio UI): http://127.0.0.1:7860

```powershell
.\.venv\Scripts\python.exe -m app.demo
```

## Data

| text | intent |
|------|--------|
| prompt string | `benign` or `injection` |

Sources (`data/raw/SOURCE.md`): neuralchemy, deepset, Radda v2.

## Deploy (Hugging Face Spaces)

Package ready in `deploy/hf-space/`. Steps: [DEPLOY_HF.md](DEPLOY_HF.md).

## Local artifacts

Heavy MiniLM/DistilBERT weights are gitignored. After clone, restore or retrain - see [RESTORE.md](RESTORE.md).

## License

MIT
