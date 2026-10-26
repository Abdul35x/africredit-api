"""
Model registry & serving (DDIA Ch. 10 — the dataworks/universe pattern).

- Every trained model is an IMMUTABLE artifact: model_registry/<version>/model.pkl
  plus metadata.json (features, metrics, dataset hash, sklearn version).
- Serving reads a "current version" pointer; promotion and rollback are atomic
  pointer swaps, so you can always roll back to a known-good artifact.
- The API validates that incoming requests match the served model's expected
  feature schema before predicting (Ch. 4: making the right thing the easy
  thing when schemas evolve).
"""
import hashlib
import json
import logging
import os
from datetime import datetime, timezone

import joblib

from app.config import CURRENT_MODEL_POINTER, MODEL_REGISTRY_DIR

logger = logging.getLogger("africredit.registry")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------
# Publishing artifacts (used by retrain.py)
# ------------------------------------------------------------------
def publish_model(version: str, model, *, features: list[str], metrics: dict,
                  dataset_hash: str) -> str:
    """Write an immutable model artifact + metadata into the registry."""
    out_dir = os.path.join(MODEL_REGISTRY_DIR, version)
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "model.pkl")
    joblib.dump(model, model_path)

    import sklearn
    meta = {
        "version": version,
        "published_at": datetime.now(timezone.utc).isoformat(),
        "features": features,
        "metrics": metrics,
        "dataset_sha256": dataset_hash,
        "sklearn_version": sklearn.__version__,
        "artifact_sha256": _sha256_file(model_path),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Published model %s (%s)", version, meta["artifact_sha256"][:12])
    return out_dir


def set_current_version(version: str) -> None:
    """Atomic pointer swap: write temp file then rename (atomic on POSIX)."""
    payload = {"version": version,
               "switched_at": datetime.now(timezone.utc).isoformat()}
    tmp = CURRENT_MODEL_POINTER + ".tmp"
    os.makedirs(os.path.dirname(CURRENT_MODEL_POINTER), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, CURRENT_MODEL_POINTER)
    logger.info("Serving pointer -> %s", version)


def get_current_version() -> str | None:
    if not os.path.exists(CURRENT_MODEL_POINTER):
        return None
    with open(CURRENT_MODEL_POINTER) as f:
        return json.load(f)["version"]


def load_metadata(version: str) -> dict:
    path = os.path.join(MODEL_REGISTRY_DIR, version, "metadata.json")
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Serving view (used by main.py at startup)
# ------------------------------------------------------------------
class ServedModel:
    def __init__(self, version: str, model, metadata: dict):
        self.version = version
        self.model = model
        self.metadata = metadata
        self.features = metadata["features"]

    def predict(self, application: dict):
        # Schema guard (Ch. 4): refuse inputs that don't match what we trained on
        # rather than silently mispredicting.
        missing = [f for f in self.features if f not in application]
        if missing:
            raise ValueError(f"Request missing features required by model "
                             f"{self.version}: {missing}")
        X = [[application[f] for f in self.features]]
        prediction = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]
        probability = float(proba[1])
        return prediction, probability


def load_served_model() -> ServedModel:
    """Load the model pointed at by the registry pointer.

    Falls back to the legacy single-file model so existing deployments keep
    working during migration (Ch. 4: compatibility when evolving systems).
    """
    version = get_current_version()
    if version:
        model_path = os.path.join(MODEL_REGISTRY_DIR, version, "model.pkl")
        metadata = load_metadata(version)
        # Integrity check: artifact must match the hash we published.
        if _sha256_file(model_path) != metadata.get("artifact_sha256"):
            raise RuntimeError(
                f"Artifact checksum mismatch for model {version}! Refusing to serve.")
        return ServedModel(version, joblib.load(model_path), metadata)

    # Legacy fallback path
    legacy = os.path.join(os.path.dirname(CURRENT_MODEL_POINTER), "credit_model.pkl")
    logger.warning("No registry pointer found; falling back to legacy %s", legacy)
    model = joblib.load(legacy)
    from train_config import FEATURES  # canonical feature list
    return ServedModel("legacy-1.0", model, {"features": FEATURES})
