"""Verify local gitignored artifacts needed for stacking demo."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, limit_mb: float | None = 8.0) -> str:
    """Hash whole file, or first limit_mb for very large weights (fast check)."""
    h = hashlib.sha256()
    max_bytes = None if limit_mb is None else int(limit_mb * 1024 * 1024)
    with path.open("rb") as f:
        read = 0
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
            if max_bytes is not None and read >= max_bytes:
                break
    return h.hexdigest()


def main() -> None:
    checks = {
        "data/clean/train.csv": ROOT / "data" / "clean" / "train.csv",
        "data/clean/test.csv": ROOT / "data" / "clean" / "test.csv",
        "models/intent_classifier.joblib": ROOT / "models" / "intent_classifier.joblib",
        "models/embedding_classifier.joblib": ROOT / "models" / "embedding_classifier.joblib",
        "models/stack_meta.joblib": ROOT / "models" / "stack_meta.joblib",
        "models/distilbert/config.json": ROOT / "models" / "distilbert" / "config.json",
        "models/distilbert/model.safetensors": ROOT / "models" / "distilbert" / "model.safetensors",
    }
    rows = []
    missing = []
    for name, path in checks.items():
        ok = path.exists()
        size = path.stat().st_size if ok else 0
        digest = sha256_file(path) if ok else ""
        rows.append({"path": name, "ok": ok, "bytes": size, "sha256_prefix": digest[:16]})
        if not ok:
            missing.append(name)
        print(f"{'OK' if ok else 'MISSING':7} {name:42} {size:>12}  {digest[:16]}")

    out = ROOT / "models" / "ARTIFACT_MANIFEST.json"
    out.write_text(json.dumps({"artifacts": rows}, indent=2), encoding="utf-8")
    print("wrote", out)
    if missing:
        raise SystemExit(f"missing {len(missing)} artifacts — see RESTORE.md")
    print("all stacking artifacts present")


if __name__ == "__main__":
    main()
