"""Reusable sentence-transformer wrapper for sklearn pipelines."""

from __future__ import annotations

import numpy as np


class SentenceEmbedder:
    """sklearn-friendly wrapper around sentence-transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None

    def fit(self, X, y=None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        if self.model is None:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
        texts = list(map(str, X))
        emb = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(emb)
