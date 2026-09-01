# Local stacking artifacts

| path | required for |
|------|----------------|
| `data/clean/train.csv` | retrain |
| `data/clean/test.csv` | retrain / eval |
| `models/intent_classifier.joblib` | TF-IDF base (tracked) |
| `models/embedding_classifier.joblib` | MiniLM base (local) |
| `models/distilbert/` | DistilBERT base (local) |
| `models/stack_meta.joblib` | meta learner (tracked) |

Generate/update checksums:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
```

This writes `models/ARTIFACT_MANIFEST.json`.

After restore, keep this folder tracked manifest in sync with verify_artifacts.
