from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from app.db import init_db, record_decision, replay_spool, get_session, DecisionLog
from app.registry import load_served_model

# ---------------------------------------------------------
# 1. SETUP & LIFECYCLE (DDIA Ch. 9: reliability — degrade gracefully)
# ---------------------------------------------------------
served_model = None


def _bootstrap():
    """Idempotent startup: safe to call from lifespan AND TestClient contexts."""
    global served_model, _bootstrapped
    if _bootstrapped:
        return
    # Init storage; if the DB is unavailable we still serve traffic,
    # audit rows just spool to disk until replay.
    try:
        init_db()
        replay_spool()
    except Exception as e:
        print(f"[startup] DB unavailable ({e}); running in degraded mode")
    try:
        served_model = load_served_model()
        print(f"[startup] serving model {served_model.version}")
    except FileNotFoundError:
        raise RuntimeError("Model file not found! Did you run train_model.py or retrain.py?")
    _bootstrapped = True


_bootstrapped = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap()
    yield


app = FastAPI(title="AfriCredit API", version="2.0", lifespan=lifespan)
# Belt-and-braces: some ASGI test harnesses skip lifespan events entirely.
_bootstrap()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# ---------------------------------------------------------
# 2. DATA VALIDATION (The "Bouncer" — stricter, with domain bounds)
# ---------------------------------------------------------
class LoanApplication(BaseModel):
    """v2 raw schema: the 7 numeric fields (required, unchanged — old clients
    keep working, Ch. 4 backward compatibility) plus optional categorical
    fields that unlock the richer model. Missing categoricals are treated as
    'unknown' by the feature pipeline, never silently dropped."""
    duration: int = Field(gt=0, le=120, description="Loan term in months")
    credit_amount: float = Field(gt=0, le=1_000_000)
    installment_commitment: int = Field(ge=1, le=24)
    residence_since: int = Field(ge=0, le=4)
    age: int = Field(ge=18, le=120)
    existing_credits: int = Field(ge=1, le=4)
    num_dependents: int = Field(ge=0, le=5)

    # Optional categorical signals (values follow the German Credit vocabulary;
    # matching is case/whitespace-insensitive in app/features.py)
    checking_status: str | None = Field(None, examples=["<0 DM", "no checking account"])
    savings_status: str | None = Field(None, examples=["<100 DM", ">=1000 DM"])
    employment: str | None = Field(None, examples=["1<=X<4", ">=7 years"])
    credit_history: str | None = Field(None, examples=["existing paid"])
    purpose: str | None = None
    personal_status: str | None = None
    other_parties: str | None = None
    property_magnitude: str | None = None
    other_payment_plans: str | None = None
    housing: str | None = None
    job: str | None = None
    foreign_worker: str | None = None

    @model_validator(mode="after")
    def check_plausible_ratios(self):
        # Domain sanity checks that a plain type system can't express.
        if self.credit_amount / max(self.duration, 1) > 50_000:
            raise ValueError("credit_amount/duration ratio implausibly high")
        return self


# ---------------------------------------------------------
# 3. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")


@app.get("/health")
def health():
    """Liveness + readiness for load balancers (Ch. 5/9)."""
    db_ok = True
    try:
        with get_session() as session:
            session.execute(DecisionLog.__table__.select().limit(1))
    except Exception:
        db_ok = False
    if served_model is None:
        return JSONResponse({"status": "unavailable"}, status_code=503)
    return JSONResponse(
        {"status": "ok", "model_version": served_model.version,
         "database": "up" if db_ok else "down (auditing to spool)"},
        status_code=200)


@app.post("/predict")
def predict_credit_risk(application: LoanApplication):
    if served_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = f"req-{uuid4().hex[:12]}"
    payload = application.model_dump()  # .dict() is deprecated in Pydantic v2

    try:
        prediction, probability = served_model.predict(payload)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    label = "Low Risk" if prediction == 1 else "High Risk"
    status = "Approved" if prediction == 1 else "Rejected"

    # Append-only audit trail (never blocks/fails the user request)
    record_decision(request_id=request_id, application=payload,
                    model_version=served_model.version,
                    prediction_label=label, probability=round(probability, 4),
                    status=status)

    return {
        "request_id": request_id,
        "risk_assessment": label,
        "approval_probability": round(probability, 2),
        "status": status,
        "model_version": served_model.version,
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/audit/recent")
def recent_decisions(limit: int = 20):
    """Read-side over the audit log for ops/debugging."""
    try:
        with get_session() as session:
            rows = (session.query(DecisionLog)
                    .order_by(DecisionLog.id.desc()).limit(min(limit, 100)).all())
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Audit store unavailable: {e}")
    return [{
        "id": r.id, "request_id": r.request_id, "timestamp_utc": r.timestamp_utc,
        "model_version": r.model_version, "status": r.status,
        "approval_probability": r.approval_probability,
    } for r in rows]
