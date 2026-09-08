"""Clean raw csv and make train/test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW_DEFAULT = ROOT / "data" / "raw" / "prompt_injection_messy.csv"
CLEAN_DIR = ROOT / "data" / "clean"
SEED = 42
TEST_SIZE = 0.2

# random casing / aliases that showed up in messy dumps
LABEL_MAP = {
    "benign": "benign",
    "injection": "injection",
    "inj": "injection",
    "malicious": "injection",
    "jailbreak": "injection",
    "safe": "benign",
    "0": "benign",
    "1": "injection",
}


def clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    text = str(text).strip().lower()
    if text in {"", "n/a", "na", "null", "none", "nan"}:
        return ""
    # leftover from when I injected fake chat markers into messy data
    if text.startswith("### user"):
        text = text[len("### user") :].lstrip("\n :")
    text = " ".join(text.split())
    return text


def normalize_label(label: object) -> str | None:
    if label is None or (isinstance(label, float) and pd.isna(label)):
        return None
    key = str(label).strip().lower()
    if not key:
        return None
    return LABEL_MAP.get(key, key if key in {"benign", "injection"} else None)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns or "intent" not in df.columns:
        raise ValueError("need text + intent columns")

    out = df.copy()
    out["text"] = out["text"].map(clean_text)
    out["intent"] = out["intent"].map(normalize_label)

    before = len(out)
    out = out[out["text"].str.len() > 0]
    out = out[out["intent"].isin(["benign", "injection"])]
    out = out.drop_duplicates(subset=["text", "intent"])

    # same text with two different labels = junk, drop
    conflict = out.groupby("text")["intent"].nunique()
    bad = set(conflict[conflict > 1].index)
    out = out[~out["text"].isin(bad)].reset_index(drop=True)

    print(f"rows: {before} -> {len(out)} after cleaning")
    print(out["intent"].value_counts().to_string())
    return out


def split_and_save(df: pd.DataFrame, clean_dir: Path, test_size: float, seed: int) -> None:
    clean_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(clean_dir / "prompt_injection_clean.csv", index=False)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["intent"],
    )
    train_df.to_csv(clean_dir / "train.csv", index=False)
    test_df.to_csv(clean_dir / "test.csv", index=False)

    print(f"train={len(train_df)} test={len(test_df)} (test_size={test_size}, seed={seed})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--out-dir", type=Path, default=CLEAN_DIR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("reading", args.input)
    clean = clean_dataframe(pd.read_csv(args.input))
    split_and_save(clean, args.out_dir, args.test_size, args.seed)


if __name__ == "__main__":
    main()
