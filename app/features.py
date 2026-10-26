"""
Feature engineering: the single canonical transformation pipeline (DDIA Ch. 4,
Ch. 10 — "the same code must run in batch and in serving", otherwise you get
training/serving skew, the classic correctness bug of ML systems).

Design decisions:
- The raw dataset schema (what clients send) is a SUPERSET of the numeric-only
  v1 schema, so old requests remain valid (backward compatibility, Ch. 4:
  "make it easy to do the right thing"). New categorical fields are optional;
  missing categoricals map to an explicit 'unknown' category rather than being
  dropped — silently imputing numerics hides data problems; an explicit
  unknown bucket makes them visible in the audit log.
- One-hot encoding is computed from a FIXED vocabulary captured at training
  time and stored with the model artifact. At serving time unseen categories
  fall into the 'unknown' column instead of shifting the feature vector
  (schema-on-read stability, Ch. 4/6).
- Ordinal categoricals (checking_status, savings_status, employment, ...) use
  their natural monetary/duration order, which keeps monotonic meaning for the
  trees instead of treating levels as unordered nominal values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Raw input columns accepted by the API / present in the dataset
# ---------------------------------------------------------------------
NUMERIC_FEATURES = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]

# Columns that carry genuine extra signal about creditworthiness
ORDINAL_CATEGORICALS = {
    # level order -> integer codes (missing -> -1, handled downstream as 'unknown')
    "checking_status": ["no checking account", "<0 DM", "0<=X<200", ">=200 DM", "unknown"],
    "savings_status": ["unknown", "<100 DM", "100<=X<500", "500<=X<1000", ">=1000 DM"],
    "employment": ["unemployed", "<1 year", "1<=X<4", "4<=X<7", ">=7 years", "unknown"],
}

NOMINAL_CATEGORICALS = [
    "credit_history", "purpose", "personal_status", "other_parties",
    "property_magnitude", "other_payment_plans", "housing", "job",
    "foreign_worker",
]

CATEGORICAL_COLUMNS = list(ORDINAL_CATEGORICALS) + NOMINAL_CATEGORICALS

ALL_RAW_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_COLUMNS

TARGET = "class"
LABEL_MAP = {"good": 1, "bad": 0}


def _to_num(series: pd.Series) -> pd.Series:
    """Coerce raw strings/objects to float, NaN on failure."""
    return pd.to_numeric(series, errors="coerce")


def ordinalize(df: pd.DataFrame) -> pd.DataFrame:
    """Map ordinal categoricals to their rank; unknown level -> last rank."""
    out = df.copy()
    for col, levels in ORDINAL_CATEGORICALS.items():
        if col not in out.columns:
            out[col] = np.nan
        rank = {lvl: i for i, lvl in enumerate(levels)}
        # normalize whitespace/case defensively (Ch. 4: tolerate sloppy producers)
        norm = out[col].astype("string").str.strip().str.lower()
        known = {k.lower(): v for k, v in rank.items()}
        unknown_rank = len(levels) - 1
        out[col] = norm.map(lambda v: known.get(v, unknown_rank)).astype(float)
    return out


def onehot_encode(df: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """One-hot nominal categoricals against a frozen vocabulary.

    vocab: {column: [level, ...]} captured at training time. Unseen levels are
    encoded as all-zeros across that column's dummies (i.e. 'other'), which is
    safe for tree models and keeps the output width stable forever.
    """
    parts = []
    for col, levels in vocab.items():
        s = df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        # fillna BEFORE string ops: a missing/None categorical must encode as
        # all-zeros (the 'other' bucket), not raise pandas' NA-string error.
        s = s.astype("string").fillna("na").str.strip().str.lower()
        for lvl in levels:
            parts.append((s == lvl.lower()).astype(np.int8).rename(f"{col}={lvl}"))
    return pd.concat(parts, axis=1)


def build_vocab(df: pd.DataFrame, min_frequency: int = 25) -> dict[str, list[str]]:
    """Frozen one-hot vocabulary from the training frame (frequency-pruned so
    rare noisy levels don't explode dimensionality)."""
    vocab = {}
    for col in NOMINAL_CATEGORICALS:
        counts = df[col].astype("string").str.strip().str.lower().value_counts()
        vocab[col] = [lvl for lvl, n in counts.items() if n >= min_frequency]
    return vocab


def transform(df: pd.DataFrame, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """The full feature pipeline: raw frame -> model matrix. THIS SAME FUNCTION
    RUNS IN retrain.py (batch) AND AT SERVE TIME (single rows)."""
    num = df[NUMERIC_FEATURES].apply(_to_num).fillna(0.0).reset_index(drop=True)
    ord_df = ordinalize(df).reset_index(drop=True)
    ord_cols = ord_df[list(ORDINAL_CATEGORICALS)].astype(float)
    oh = onehot_encode(ord_df, vocab).reset_index(drop=True)
    X = pd.concat([num, ord_cols, oh], axis=1)
    X.columns = [c.replace(" ", "_") for c in X.columns]
    # Uniform float64: sklearn keeps a single dtype across chunks; mixed
    # int8/float columns break single-row predict at serve time.
    return X.astype("float64")


def transform_row(payload: dict, vocab: dict[str, list[str]]) -> pd.DataFrame:
    """Serve-time entry point: one request dict -> one-row feature matrix."""
    row = {col: payload.get(col) for col in ALL_RAW_COLUMNS}
    df = pd.DataFrame([row])
    return transform(df, vocab)
