"""Shared training configuration so train_model.py and retrain.py agree.

v2: the canonical feature definition now lives in app/features.py (raw input
schema + transformation pipeline). This module re-exports it so existing
imports keep working (Ch. 4: evolve interfaces without breaking consumers).
"""
from app.features import (ALL_RAW_COLUMNS, CATEGORICAL_COLUMNS, LABEL_MAP,
                          NOMINAL_CATEGORICALS, NUMERIC_FEATURES, TARGET)

# Raw schema the API accepts / batch expects — superset of the v1 numeric list.
FEATURES = ALL_RAW_COLUMNS

__all__ = ["FEATURES", "NUMERIC_FEATURES", "CATEGORICAL_COLUMNS",
           "NOMINAL_CATEGORICALS", "TARGET", "LABEL_MAP"]
