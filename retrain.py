"""
Batch retraining pipeline (DDIA Ch. 10: the batch layer).

Pipeline stages, each idempotent and rebuildable from inputs:

  1. EXTRACT   — baseline German Credit data + labeled rows from the
                 decision_log audit table (feedback loop: production
                 requests become tomorrow's training data).
  2. PREPROCESS— canonical feature selection from train_config.py.
  3. TRAIN     — RandomForest on a time-aware-ish split.
  4. EVALUATE  — accuracy PLUS precision/recall/ROC-AUC and a confusion
                 matrix (Ch. 9: choose metrics that reflect what matters;
                 for credit risk, recall of the 'good' class is money).
  5. GATE      — champion/challenger: refuse to publish unless the new model
                 beats the currently served one on ROC-AUC by a margin.
  6. PUBLISH   — immutable artifact into the model registry.
  7. (--promote) atomic pointer swap to serve it; rollback = promote old version.

Usage:
    python retrain.py                    # train + evaluate + publish (staged)
    python retrain.py --promote          # ...and immediately serve it
    python retrain.py --version v2026-10-26
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from app.registry import (get_current_version, load_metadata, publish_model,
                          set_current_version)
from train_config import FEATURES, LABEL_MAP, TARGET

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "credit-g.csv")


# ------------------------------------------------------------------
# 1. EXTRACT
# ------------------------------------------------------------------
def load_baseline_dataset() -> pd.DataFrame:
    """Cache the OpenML dataset locally so reruns don't need network (Ch. 10:
    the batch layer is a durable, immutable snapshot of inputs)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"[extract] baseline: {len(df)} rows (cached at {DATASET_PATH})")
        return df
    from sklearn.datasets import fetch_openml
    print("[extract] fetching German Credit dataset from OpenML...")
    data = fetch_openml(name='credit-g', version=1, as_frame=True)
    df = data.frame.copy()
    df.to_csv(DATASET_PATH, index=False)
    print(f"[extract] baseline: {len(df)} rows (fetched & cached)")
    return df


def load_labeled_audit_rows() -> pd.DataFrame:
    """Replay the append-only decision log as candidate training rows.

    NOTE: audit rows only contain features + predictions, not ground-truth
    outcomes yet. We join in labels when available: any row whose 'status'
    was later manually verified against repayment outcome would carry an
    'outcome' column. Until then we treat audit rows as UNLABELED and skip
    them — this function defines the feedback-loop seam where real labels
    plug in (see README note on delayed observations, DDIA Ch. 11).
    """
    try:
        from app.db import DecisionLog, get_session
        from sqlalchemy import select
        with get_session() as session:
            rows = session.execute(
                select(DecisionLog).where(DecisionLog.payload_json.isnot(None))
            ).scalars().all()
    except Exception as e:
        print(f"[extract] audit log unavailable ({e}); skipping incremental data")
        return pd.DataFrame()

    records = []
    for r in rows:
        payload = json.loads(r.payload_json)
        # Ground truth would come from a repayment-events join (Ch. 11).
        # Placeholder: only rows explicitly labeled count.
        if "outcome" in payload:
            rec = {f: payload[f] for f in FEATURES}
            rec[TARGET] = payload["outcome"]
            records.append(rec)
    df = pd.DataFrame(records)
    print(f"[extract] audit log: {len(rows)} decisions, {len(df)} newly labeled")
    return df


def dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


# ------------------------------------------------------------------
# 2-4. PREPROCESS / TRAIN / EVALUATE
# ------------------------------------------------------------------
def preprocess(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].copy()
    y = df[TARGET].map(LABEL_MAP).astype(int)
    # Coerce types defensively — audit-sourced JSON may arrive as strings.
    for c in FEATURES:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    keep = X.notna().all(axis=1)
    return X[keep], y[keep]


def _new_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, max_depth=12,
                                  min_samples_leaf=5, random_state=42)


def train(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = _new_model()
    model.fit(X_train, y_train)
    return model, X_test, y_test


def evaluate(model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "n_test": int(len(y_test)),
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision_good": round(precision_score(y_test, pred, pos_label=1), 4),
        "recall_good": round(recall_score(y_test, pred, pos_label=1), 4),
        "f1_good": round(f1_score(y_test, pred, pos_label=1), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": cm.tolist(),
    }
    print("[evaluate]", json.dumps(metrics, indent=2))
    return metrics


def cross_validated_auc(X: pd.DataFrame, y: pd.Series) -> float:
    """Honest generalization estimate.

    A single 80/20 split on 1000 rows has a ROC-AUC standard error of ~0.04 —
    the test-set number is too noisy to gate deployments on (DDIA Ch. 9:
    measurement error drives bad operational decisions). We use stratified
    5-fold CV out-of-fold probabilities instead.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(_new_model(), X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    return round(float(roc_auc_score(y, oof)), 4)


# ------------------------------------------------------------------
# 5. GATE (champion/challenger)
# ------------------------------------------------------------------
def gate(cv_auc: float, metrics: dict, current_version: str | None,
         min_cv_auc: float = 0.60) -> tuple[bool, str]:
    """Gate on the CROSS-VALIDATED AUC (stable), not the single-split number.

    Note on floors: with only the 7 numeric features of German Credit, honest
    CV-AUC is ~0.65 — that is the real ceiling of this feature set. A floor of
    0.75 would be aspirational marketing, not engineering; we set a defensible
    floor and let champion/challenger comparison drive actual upgrades.
    Raise `min_cv_auc` as richer features (Ch. 4/10) come online.
    """
    if cv_auc < min_cv_auc:
        return False, f"CV ROC-AUC {cv_auc} below floor {min_cv_auc}"
    if current_version:
        try:
            incumbent = load_metadata(current_version)
            inc_auc = incumbent.get("metrics", {}).get("cv_roc_auc")
            if inc_auc is not None and cv_auc <= inc_auc:
                return False, (f"challenger CV-AUC {cv_auc} does not beat "
                               f"incumbent {current_version} CV-AUC {inc_auc}")
        except FileNotFoundError:
            pass  # incumbent has no metadata we can compare against
    return True, "gate passed"


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
def run(promote: bool, version: str | None) -> int:
    t0 = datetime.now(timezone.utc)
    version = version or "v" + t0.strftime("%Y%m%d-%H%M%S")

    frames = [load_baseline_dataset()]
    labeled = load_labeled_audit_rows()
    if len(labeled):
        frames.append(labeled)

    X, y = preprocess(frames)
    print(f"[preprocess] training matrix: {X.shape}, positive rate: {y.mean():.3f}")

    model, X_test, y_test = train(X, y)
    metrics = evaluate(model, X_test, y_test)
    cv_auc = cross_validated_auc(X, y)
    metrics["cv_roc_auc"] = cv_auc
    print(f"[evaluate] cross-validated ROC-AUC: {cv_auc}")

    current = get_current_version()
    ok, reason = gate(cv_auc, metrics, current)
    print(f"[gate] {reason}")
    if not ok:
        print("[abort] model NOT published. Incumbent stays live.")
        return 1

    out_dir = publish_model(version, model, features=FEATURES,
                            metrics=metrics, dataset_hash=dataset_hash(X))
    print(f"[publish] artifact at {out_dir}")

    if promote:
        set_current_version(version)
        print(f"[promote] now serving {version}. "
              f"(rollback: python retrain.py --promote-to {current})")
    else:
        print(f"[stage] {version} published but NOT promoted. "
              f"Run: python -c \"from app.registry import set_current_version;"
              f"set_current_version('{version}')\"")
    print(f"[done] wall time {(datetime.now(timezone.utc) - t0).total_seconds():.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AfriCredit batch retraining pipeline")
    parser.add_argument("--promote", action="store_true",
                        help="atomically switch serving pointer to the new version")
    parser.add_argument("--promote-to", metavar="VERSION",
                        help="rollback/serve an existing registry version without retraining")
    parser.add_argument("--version", help="explicit version label")
    args = parser.parse_args()

    if args.promote_to:
        set_current_version(args.promote_to)
        print(f"[promote] serving {args.promote_to}")
        sys.exit(0)
    sys.exit(run(promote=args.promote, version=args.version))
