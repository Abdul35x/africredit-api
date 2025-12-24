from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles 
from fastapi.responses import FileResponse 
import joblib
import pandas as pd

# ---------------------------------------------------------
# 1. SETUP & MODEL LOADING
# ---------------------------------------------------------
app = FastAPI(title="AfriCredit API", version="1.0")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

try:
    model = joblib.load('model/credit_model.pkl')
except FileNotFoundError:
    # Fail fast if the model isn't there
    raise RuntimeError("Model file not found! Did you run train_model.py?")

# ---------------------------------------------------------
# 2. DATA VALIDATION (The "Bouncer")
# ---------------------------------------------------------
class LoanApplication(BaseModel):
    duration: int               
    credit_amount: float        
    installment_commitment: int 
    residence_since: int        
    age: int                    
    existing_credits: int       
    num_dependents: int         

# ---------------------------------------------------------
# 3. API ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def read_root():
    return FileResponse("app/static/index.html")

@app.post("/predict")
def predict_credit_risk(application: LoanApplication):
    data = pd.DataFrame([application.dict()])
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    return {
        "risk_assessment": "Low Risk" if prediction == 1 else "High Risk",
        "approval_probability": round(probability, 2),
        "status": "Approved" if prediction == 1 else "Rejected"
    }