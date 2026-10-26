"""
Database layer (DDIA Ch. 3: transactional storage; Ch. 12: the auditability
guarantee that operational data is never lost or silently mutated).

Design decisions:
- The OLTP request path uses short-lived sessions (connection-per-request),
  which works identically on SQLite and Postgres.
- Audit writes are append-only: there is no UPDATE/DELETE API anywhere.
- If the DB is down, predictions still succeed; the audit record is written
  to a local spool file and replayed later (Ch. 9: degrade gracefully,
  never lose the audit trail).
"""
import json
import logging
import os
from contextlib import contextmanager
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from app.config import DATABASE_URL, IS_POSTGRES

logger = logging.getLogger("africredit.db")


class Base(DeclarativeBase):
    pass


class DecisionLog(Base):
    """Append-only log of every credit decision.

    Stores the FULL input payload as JSON text alongside the normalized
    columns. This makes the log immune to feature-schema evolution (Ch. 4):
    old rows remain readable even after the Pydantic model gains/renames
    fields, and it is the raw material for future batch retraining.
    """
    __tablename__ = "decision_log"

    # BIGINT autoincrement on Postgres, INTEGER on SQLite — SQLAlchemy handles it.
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(index=True)
    timestamp_utc: Mapped[str] = mapped_column(index=True)
    model_version: Mapped[str] = mapped_column()
    # Input features (normalized columns for querying/reporting)
    duration: Mapped[int] = mapped_column()
    credit_amount: Mapped[float] = mapped_column()
    installment_commitment: Mapped[int] = mapped_column()
    residence_since: Mapped[int] = mapped_column()
    age: Mapped[int] = mapped_column()
    existing_credits: Mapped[int] = mapped_column()
    num_dependents: Mapped[int] = mapped_column()
    # Outcome
    prediction_label: Mapped[str] = mapped_column()
    approval_probability: Mapped[float] = mapped_column()
    status: Mapped[str] = mapped_column()
    # Full immutable snapshot of the request (schema-evolution safety net)
    payload_json: Mapped[str] = mapped_column()


# ------------------------------------------------------------------
# Engine / session factory
# ------------------------------------------------------------------
_connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    _db_file = DATABASE_URL.replace("sqlite:///", "")
    os.makedirs(os.path.dirname(_db_file), exist_ok=True)
    # Needed because audit writes may happen from FastAPI's threadpool.
    _connect_args["check_same_thread"] = False

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_connect_args)
# expire_on_commit=False: callers may read ORM attributes after the session's
# commit (e.g. /audit/recent serializes rows after the context exits).
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False,
                           expire_on_commit=False)


def init_db() -> None:
    """Create tables if they don't exist.

    DDIA would tell you to use Alembic migrations in production; this is the
    pragmatic bootstrap that keeps dev frictionless. On Postgres we also add
    a DB-level guard enforcing append-only semantics.
    """
    Base.metadata.create_all(engine)
    if IS_POSTGRES:
        from sqlalchemy import text
        with engine.begin() as conn:
            # A rule-based trigger that refuses UPDATE/DELETE on the audit log.
            conn.execute(text("""
                CREATE OR REPLACE FUNCTION africredit_forbid_mutation()
                RETURNS trigger AS $$
                BEGIN
                    RAISE EXCEPTION 'decision_log is append-only (%) is forbidden', TG_OP;
                END;
                $$ LANGUAGE plpgsql;
            """))
            conn.execute(text("""
                DROP TRIGGER IF EXISTS decision_log_append_only ON decision_log;
            """))
            conn.execute(text("""
                CREATE TRIGGER decision_log_append_only
                BEFORE UPDATE OR DELETE ON decision_log
                FOR EACH ROW EXECUTE FUNCTION africredit_forbid_mutation();
            """))
    logger.info("Database initialized at %s", DATABASE_URL.split("@")[-1])


@contextmanager
def get_session():
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ------------------------------------------------------------------
# Append-only audit writer with spool-file fallback
# ------------------------------------------------------------------
SPOOL_FILE = os.path.join(os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
                          if DATABASE_URL.startswith("sqlite") else ".",
                          "audit_spool.jsonl")


def record_decision(*, request_id: str, application: dict, model_version: str,
                    prediction_label: str, probability: float, status: str) -> None:
    """Write one audit row. Never raises — falls back to the spool file."""
    row = {
        "request_id": request_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        **application,
        "prediction_label": prediction_label,
        "approval_probability": probability,
        "status": status,
    }
    try:
        with get_session() as session:
            session.add(DecisionLog(
                request_id=row["request_id"],
                timestamp_utc=row["timestamp_utc"],
                model_version=model_version,
                duration=application["duration"],
                credit_amount=application["credit_amount"],
                installment_commitment=application["installment_commitment"],
                residence_since=application["residence_since"],
                age=application["age"],
                existing_credits=application["existing_credits"],
                num_dependents=application["num_dependents"],
                prediction_label=prediction_label,
                approval_probability=probability,
                status=status,
                payload_json=json.dumps(application),
            ))
    except Exception as e:
        # Graceful degradation: the audit trail survives in a durable spool.
        logger.error("Audit DB write failed (%s); spooling to %s", e, SPOOL_FILE)
        try:
            with open(SPOOL_FILE, "a") as f:
                f.write(json.dumps(row) + "\n")
        except Exception as e2:
            logger.critical("Audit spool also failed: %s", e2)


def replay_spool() -> int:
    """Re-insert spooled audit rows into the DB once it is healthy again."""
    if not os.path.exists(SPOOL_FILE):
        return 0
    replayed = 0
    remaining = []
    with open(SPOOL_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                with get_session() as session:
                    payload = json.loads(row.get("payload_json", "{}")) or \
                        {k: row[k] for k in ("duration", "credit_amount",
                                             "installment_commitment", "residence_since",
                                             "age", "existing_credits", "num_dependents")}
                    session.add(DecisionLog(
                        request_id=row["request_id"],
                        timestamp_utc=row["timestamp_utc"],
                        model_version=row["model_version"],
                        duration=payload["duration"],
                        credit_amount=payload["credit_amount"],
                        installment_commitment=payload["installment_commitment"],
                        residence_since=payload["residence_since"],
                        age=payload["age"],
                        existing_credits=payload["existing_credits"],
                        num_dependents=payload["num_dependents"],
                        prediction_label=row["prediction_label"],
                        approval_probability=row["approval_probability"],
                        status=row["status"],
                        payload_json=json.dumps(payload),
                    ))
                replayed += 1
            except Exception:
                remaining.append(line)
    with open(SPOOL_FILE, "w") as f:
        f.writelines(l + "\n" for l in remaining)
    if replayed:
        logger.info("Replayed %d spooled audit records", replayed)
    return replayed
