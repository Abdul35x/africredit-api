"""
Central configuration (DDIA Ch. 4: schema evolution & config management).

Everything is driven by environment variables so the same code runs against
SQLite locally and PostgreSQL in production without any code changes.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
# Production: DATABASE_URL=postgresql+psycopg://user:pass@host:5432/africredit
# Local/dev: falls back to a SQLite file so the app runs with zero setup.
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'africredit.db')}")

# ---------------------------------------------------------
# Model registry (DDIA Ch. 10: batch layer keeps immutable artifacts)
# ---------------------------------------------------------
MODEL_REGISTRY_DIR = os.getenv("MODEL_REGISTRY_DIR", os.path.join(BASE_DIR, "model_registry"))
# Pointer to the currently serving version ("latest alias"), stored as a small
# JSON file so promotion/rollback is an atomic pointer swap.
CURRENT_MODEL_POINTER = os.getenv("CURRENT_MODEL_POINTER", os.path.join(BASE_DIR, "model", "current_model.json"))

# Is this a PostgreSQL backend? (affects SQL we emit for auditing)
IS_POSTGRES = DATABASE_URL.startswith("postgres")
