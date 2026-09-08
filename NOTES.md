# notes

- pipeline: preprocess → train → embeddings → ensemble/stacking → distilbert → stack --with-distilbert → demo
- active model: `models/active_model.txt` → **stacking** (tfidf+minilm+distilbert)
- best metrics: ~0.916 acc / ~0.913 macro F1
- DistilBERT alone: ~0.911 acc (weights under `models/distilbert/`, gitignored — retrain or Colab export)
- Colab GPU for 2 epochs: `notebooks/COLAB_DISTILBERT.md`
- premium Gradio UI: `python -m app.demo` → http://127.0.0.1:7860

- if embedding/DistilBERT weights missing: see RESTORE.md / scripts/restore_from_backup.ps1

- local heavy models restored; verify with: python -m scripts.verify_artifacts
