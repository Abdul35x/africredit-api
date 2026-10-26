# AfriCredit-API: Alternative Credit Scoring Engine

A Machine Learning API designed to assess creditworthiness in emerging markets using alternative data proxies.

![AfriCredit Dashboard](dashboard_screenshot.png)

# Project Overview
In many developing economies, traditional credit bureau coverage is low, leaving millions of creditworthy individuals "invisible" to formal banking. AfriCredit attempts to bridge this gap by using a Random Forest Classifier to predict loan repayment probability based on non-traditional metrics like loan duration, installment commitment, and asset stability.

This project was built as a 10-Day Engineering Sprint to demonstrate full-stack Data Science skills: from model training to containerized deployment.

# Key Features
Machine Learning Core: Trained on the German Credit Dataset using Scikit-Learn (Random Forest).
High-Performance API: Built with FastAPI for asynchronous, rapid inference.
Dockerized: Fully containerized application ensuring consistency across environments.
Robust Validation: Implements Pydantic models to prevent invalid data ingestion.
User Dashboard: A clean, accessible HTML/JS frontend for non-technical users like Loan Officers.

# Tech Stack
Language: Python 3.11
ML Libraries: Scikit-Learn, Pandas, Joblib
Backend: FastAPI, Uvicorn
DevOps: Docker, Docker Compose
Frontend: HTML5, CSS3, JavaScript

# Architecture
The system follows a micro-service pattern:
1.Data Ingestion: User inputs data via the Dashboard or API.
2.Validation: Pydantic ensures data types match the schema.
3.Inference: The input is converted to a DataFrame and fed into the serialized Random Forest model (`.pkl`).
4.Response: The API returns a Risk Label (Low/High) and a Probability Score.

# How to Run

# Option 1 (Recommended): Using Docker 
This ensures you run the exact environment I developed in.

1.  Build the Image:
    ```bash
    docker build -t africredit-app .
    ```
2.  Run the Container:
    ```bash
    docker run -d -p 8000:80 africredit-app
    ```
3.  Access the Dashboard: Open `http://localhost:8000`
4.  Access API Docs: Open `http://localhost:8000/docs`

# Option 2: Local Python Setup
1.  Create Virtual Environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\venv\Scripts\activate
    ```
2.  Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Train the Model:
    ```bash
    python train_model.py
    ```
4.  Start the Server:
    ```bash
    uvicorn app.main:app --reload
    ```

# Challenges & Solutions (Learning Journey)
As a student project, I encountered and solved several real-world engineering hurdles:

1. Cross-Platform Virtualization: Resolved kernel compatibility issues between Windows (WSL2) and Linux containers. This reinforced the importance of containerization for ensuring consistent runtime environments across different operating systems.

2. Dependency Hell: I encountered a version conflict where `numpy 2.3.5` (installed on my local Python 3.12) refused to install on the Docker container's Python 3.9 image. I resolved this by pinning the Docker base image to `python:3.11-slim` to match my development environment.

3. Model serialization: Understanding how to persist a trained model using `joblib` so the API doesn't need to retrain on every request.

# Limitations & Future Roadmap

1. Data Imbalance: The training data is skewed towards "Good" credit. Future versions will implement SMOTE (Synthetic Minority Over-sampling Technique) to better detect high-risk cases.

2. Explainability: The current model gives a score but not the "Why." I plan to integrate SHAP (SHapley Additive exPlanations) to show users exactly which factor (e.g., Age vs. Duration) lowered their score.

3. Database Integration: Currently, data is transient. I plan to add PostgreSQL to store loan applications for auditing and retraining.


Author: Abdurrahman Ibrahim
Role: Student Software Engineer & Data Science Enthusiast

---

## Architecture Upgrade (Oct 2026) — DDIA-inspired

### Append-only audit log (`app/db.py`)
Every `/predict` decision is written to a `decision_log` table (SQLite locally,
PostgreSQL in prod via `DATABASE_URL`). The log is append-only: on Postgres a
trigger forbids UPDATE/DELETE. If the DB is down, predictions still succeed and
audit rows spool to `data/audit_spool.jsonl`, replayed on next startup.

```bash
export DATABASE_URL=postgresql+psycopg://user:pass@host:5432/africredit
```

### Model registry + batch retraining (`retrain.py`, `app/registry.py`)
Immutable artifacts under `model_registry/<version>/` (model.pkl + metadata with
metrics, dataset hash, checksums). Serving uses an atomic pointer swap
(`model/current_model.json`) — promotion and rollback are one command:

```bash
python retrain.py                 # extract → train → evaluate → gate → publish (staged)
python retrain.py --promote       # ...and serve it
python retrain.py --promote-to v2026-10-26-baseline   # rollback/re-serve
```

The gate compares cross-validated ROC-AUC against the incumbent champion; a
challenger that doesn't beat it is never published.

### New endpoints
- `GET /health` — liveness/readiness (model version + DB status) for LBs.
- `GET /audit/recent?limit=N` — ops view over the decision log.

### Local production-parity stack
```bash
docker compose up --build   # Postgres 16 + 2 web replicas
```
