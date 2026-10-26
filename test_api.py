import os

# Force an isolated SQLite DB for tests (must be set before importing app)
os.environ["DATABASE_URL"] = "sqlite:///./data/test_africredit.db"

from fastapi.testclient import TestClient
from app.main import app  # noqa: E402

client = TestClient(app)

VALID_APP = {
    "duration": 24, "credit_amount": 1500.0, "installment_commitment": 3,
    "residence_since": 2, "age": 35, "existing_credits": 2, "num_dependents": 1,
}


def test_read_main():
    assert client.get("/").status_code == 200


def test_docs_available():
    assert client.get("/docs").status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_predict_ok_and_audited():
    r = client.post("/predict", json=VALID_APP)
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("Approved", "Rejected")
    assert 0 <= body["approval_probability"] <= 1
    assert body["request_id"].startswith("req-")

    # The decision must appear in the append-only audit log
    audit = client.get("/audit/recent?limit=5").json()
    assert any(a["request_id"] == body["request_id"] for a in audit)


def test_predict_handles_missing_and_null_categoricals():
    """Regression: single-row serve-time encoding used to raise pandas'
    'NA values are not allowed in this context' when categorical fields were
    omitted or sent as null. They must encode as the all-zeros 'other' bucket
    and still return 200."""
    r_omit = client.post("/predict", json=VALID_APP)
    assert r_omit.status_code == 200, r_omit.text

    nulls = {**VALID_APP, "credit_history": None, "purpose": None, "job": None}
    r_null = client.post("/predict", json=nulls)
    assert r_null.status_code == 200, r_null.text

    # Unseen levels must also be tolerated (encoded as 'other')
    unseen = {**VALID_APP, "credit_history": "some-never-seen-level"}
    r_unseen = client.post("/predict", json=unseen)
    assert r_unseen.status_code == 200, r_unseen.text


def test_predict_rejects_invalid_age():
    bad = dict(VALID_APP, age=15)
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_negative_amount():
    bad = dict(VALID_APP, credit_amount=-100)
    assert client.post("/predict", json=bad).status_code == 422


def test_audit_log_is_append_only():
    from app.db import DecisionLog, get_session
    with get_session() as s:
        n_before = s.query(DecisionLog).count()
    client.post("/predict", json=VALID_APP)
    with get_session() as s:
        n_after = s.query(DecisionLog).count()
    assert n_after == n_before + 1
