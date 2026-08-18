# Local artifact restore

Heavy files are gitignored (size). If `embedding_classifier.joblib` or `models/distilbert/` go missing:

## Fast path (backup snap)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\restore_from_backup.ps1
```

Default backup: `%TEMP%\ice-final-snap` (created during history rebuild).

## Retrain path

```powershell
.\.venv\Scripts\python.exe -m src.train_embeddings
.\.venv\Scripts\python.exe -m src.train_distilbert --epochs 1
.\.venv\Scripts\python.exe -m src.train_stacking --with-distilbert
```

## Verify

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_artifacts
.\.venv\Scripts\python.exe -m app.demo
```

Active model should stay `stacking` when all three bases are present.
